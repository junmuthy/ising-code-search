"""Sparse categorical sum-product decoding for the BB64 D3 factor graph."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numba
import numpy as np

from .factor_graph import D3FactorGraph


@dataclass(frozen=True)
class CompiledFactorGraph:
    state_offsets: np.ndarray
    state_codes: np.ndarray
    state_log_priors: np.ndarray
    state_detector_words: np.ndarray
    variable_edge_offsets: np.ndarray
    edge_checks: np.ndarray
    check_edge_offsets: np.ndarray
    check_edges: np.ndarray
    component_offsets: np.ndarray

    @property
    def num_variables(self) -> int:
        return len(self.state_offsets) - 1

    @property
    def num_checks(self) -> int:
        return len(self.check_edge_offsets) - 1


@dataclass(frozen=True)
class BPResult:
    hard_codes: np.ndarray
    state_log_posteriors: np.ndarray
    component_llrs: np.ndarray
    iterations: int
    converged: bool
    maximum_message_change: float
    hard_detector_mask: int
    residual_detector_mask: int


def _words(value: int, count: int = 4) -> tuple[int, ...]:
    return tuple((value >> (64 * index)) & ((1 << 64) - 1) for index in range(count))


def compile_factor_graph(graph: D3FactorGraph) -> CompiledFactorGraph:
    state_offsets = [0]
    state_codes: list[int] = []
    state_log_priors: list[float] = []
    state_words: list[tuple[int, ...]] = []
    variable_edge_offsets = [0]
    edge_checks: list[int] = []
    check_edges: list[list[int]] = [[] for _ in range(graph.parsed_circuit.num_detectors)]
    for variable_index, variable in enumerate(graph.variables):
        union = 0
        for code, log_prior, signature in zip(
            variable.state_codes,
            variable.log_priors,
            variable.signatures,
            strict=True,
        ):
            state_codes.append(code)
            state_log_priors.append(log_prior)
            state_words.append(_words(signature.detector_mask))
            union |= signature.detector_mask
        state_offsets.append(len(state_codes))
        checks = []
        while union:
            least = union & -union
            checks.append(least.bit_length() - 1)
            union ^= least
        for check in checks:
            edge = len(edge_checks)
            edge_checks.append(check)
            check_edges[check].append(edge)
        variable_edge_offsets.append(len(edge_checks))
    flat_check_edges: list[int] = []
    check_offsets = [0]
    for edges in check_edges:
        flat_check_edges.extend(edges)
        check_offsets.append(len(flat_check_edges))
    return CompiledFactorGraph(
        state_offsets=np.asarray(state_offsets, dtype=np.int32),
        state_codes=np.asarray(state_codes, dtype=np.uint8),
        state_log_priors=np.asarray(state_log_priors, dtype=np.float64),
        state_detector_words=np.asarray(state_words, dtype=np.uint64),
        variable_edge_offsets=np.asarray(variable_edge_offsets, dtype=np.int32),
        edge_checks=np.asarray(edge_checks, dtype=np.int16),
        check_edge_offsets=np.asarray(check_offsets, dtype=np.int32),
        check_edges=np.asarray(flat_check_edges, dtype=np.int32),
        component_offsets=np.asarray(graph.component_offsets, dtype=np.int32),
    )


@numba.njit(cache=True)
def _logadd(left: float, right: float) -> float:
    if left == -np.inf:
        return right
    if right == -np.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


@numba.njit(cache=True)
def _state_parity(words: np.ndarray, state: int, check: int) -> int:
    return int((words[state, check // 64] >> np.uint64(check % 64)) & np.uint64(1))


@numba.njit(cache=True)
def _initialize_variable_messages(
    state_offsets: np.ndarray,
    log_priors: np.ndarray,
    words: np.ndarray,
    edge_offsets: np.ndarray,
    edge_checks: np.ndarray,
) -> np.ndarray:
    messages = np.empty(len(edge_checks), dtype=np.float64)
    for variable in range(len(state_offsets) - 1):
        start = state_offsets[variable]
        stop = state_offsets[variable + 1]
        for edge in range(edge_offsets[variable], edge_offsets[variable + 1]):
            check = edge_checks[edge]
            even = -np.inf
            odd = -np.inf
            for state in range(start, stop):
                if _state_parity(words, state, check):
                    odd = _logadd(odd, log_priors[state])
                else:
                    even = _logadd(even, log_priors[state])
            messages[edge] = even - odd
    return messages


@numba.njit(cache=True)
def _check_update(
    syndrome: np.ndarray,
    check_offsets: np.ndarray,
    check_edges: np.ndarray,
    variable_to_check: np.ndarray,
    old_check_to_variable: np.ndarray,
    damping: float,
    clip: float,
) -> tuple[np.ndarray, float]:
    output = np.empty_like(old_check_to_variable)
    maximum_change = 0.0
    for check in range(len(check_offsets) - 1):
        start = check_offsets[check]
        stop = check_offsets[check + 1]
        product = 1.0
        zero_count = 0
        for item in range(start, stop):
            edge = check_edges[item]
            value = math.tanh(0.5 * variable_to_check[edge])
            if abs(value) < 1e-300:
                zero_count += 1
            else:
                product *= value
        sign = -1.0 if syndrome[check] else 1.0
        for item in range(start, stop):
            edge = check_edges[item]
            value = math.tanh(0.5 * variable_to_check[edge])
            if zero_count == 0:
                excluded = product / value
            elif zero_count == 1 and abs(value) < 1e-300:
                excluded = product
            else:
                excluded = 0.0
            excluded = min(1.0 - 1e-15, max(-1.0 + 1e-15, sign * excluded))
            raw = min(clip, max(-clip, 2.0 * math.atanh(excluded)))
            updated = damping * old_check_to_variable[edge] + (1.0 - damping) * raw
            output[edge] = updated
            maximum_change = max(maximum_change, abs(updated - old_check_to_variable[edge]))
    return output, maximum_change


@numba.njit(cache=True)
def _variable_update(
    state_offsets: np.ndarray,
    log_priors: np.ndarray,
    words: np.ndarray,
    edge_offsets: np.ndarray,
    edge_checks: np.ndarray,
    check_to_variable: np.ndarray,
    old_variable_to_check: np.ndarray,
    damping: float,
    clip: float,
) -> tuple[np.ndarray, float]:
    output = np.empty_like(old_variable_to_check)
    maximum_change = 0.0
    scores = np.empty(16, dtype=np.float64)
    for variable in range(len(state_offsets) - 1):
        start = state_offsets[variable]
        stop = state_offsets[variable + 1]
        edge_start = edge_offsets[variable]
        edge_stop = edge_offsets[variable + 1]
        for local_state, state in enumerate(range(start, stop)):
            score = log_priors[state]
            for edge in range(edge_start, edge_stop):
                if _state_parity(words, state, edge_checks[edge]):
                    score -= check_to_variable[edge]
            scores[local_state] = score
        for edge in range(edge_start, edge_stop):
            check = edge_checks[edge]
            even = -np.inf
            odd = -np.inf
            for local_state, state in enumerate(range(start, stop)):
                parity = _state_parity(words, state, check)
                score = scores[local_state]
                if parity:
                    score += check_to_variable[edge]
                    odd = _logadd(odd, score)
                else:
                    even = _logadd(even, score)
            raw = min(clip, max(-clip, even - odd))
            updated = damping * old_variable_to_check[edge] + (1.0 - damping) * raw
            output[edge] = updated
            maximum_change = max(maximum_change, abs(updated - old_variable_to_check[edge]))
    return output, maximum_change


@numba.njit(cache=True)
def _posterior(
    state_offsets: np.ndarray,
    state_codes: np.ndarray,
    log_priors: np.ndarray,
    words: np.ndarray,
    edge_offsets: np.ndarray,
    edge_checks: np.ndarray,
    check_to_variable: np.ndarray,
    component_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_logs = np.full(len(log_priors), -np.inf, dtype=np.float64)
    hard_codes = np.zeros(len(state_offsets) - 1, dtype=np.uint8)
    component_llrs = np.zeros(component_offsets[-1], dtype=np.float64)
    for variable in range(len(state_offsets) - 1):
        start = state_offsets[variable]
        stop = state_offsets[variable + 1]
        maximum = -np.inf
        best_state = start
        for state in range(start, stop):
            score = log_priors[state]
            for edge in range(edge_offsets[variable], edge_offsets[variable + 1]):
                if _state_parity(words, state, edge_checks[edge]):
                    score -= check_to_variable[edge]
            state_logs[state] = score
            if score > maximum:
                maximum = score
                best_state = state
        normalization = 0.0
        for state in range(start, stop):
            normalization += math.exp(state_logs[state] - maximum)
        log_normalization = maximum + math.log(normalization)
        for state in range(start, stop):
            state_logs[state] -= log_normalization
        hard_codes[variable] = state_codes[best_state]
        width = component_offsets[variable + 1] - component_offsets[variable]
        for bit in range(width):
            zero = -np.inf
            one = -np.inf
            for state in range(start, stop):
                if (state_codes[state] >> bit) & 1:
                    one = _logadd(one, state_logs[state])
                else:
                    zero = _logadd(zero, state_logs[state])
            component_llrs[component_offsets[variable] + bit] = zero - one
    return state_logs, hard_codes, component_llrs


def _mask_from_codes(graph: D3FactorGraph, codes: np.ndarray) -> int:
    return graph.detector_mask(tuple(map(int, codes)))


class CategoricalBPDecoder:
    def __init__(
        self,
        graph: D3FactorGraph,
        *,
        maximum_iterations: int = 40,
        damping: float = 0.45,
        convergence_tolerance: float = 1e-5,
        message_clip: float = 30.0,
    ):
        if maximum_iterations <= 0 or not 0.0 <= damping < 1.0:
            raise ValueError("invalid BP iteration or damping setting")
        self.graph = graph
        self.compiled = compile_factor_graph(graph)
        self.maximum_iterations = int(maximum_iterations)
        self.damping = float(damping)
        self.convergence_tolerance = float(convergence_tolerance)
        self.message_clip = float(message_clip)

    def decode(self, observed_detector_mask: int) -> BPResult:
        syndrome = np.fromiter(
            (
                (int(observed_detector_mask) >> check) & 1
                for check in range(self.compiled.num_checks)
            ),
            dtype=np.uint8,
        )
        variable_to_check = _initialize_variable_messages(
            self.compiled.state_offsets,
            self.compiled.state_log_priors,
            self.compiled.state_detector_words,
            self.compiled.variable_edge_offsets,
            self.compiled.edge_checks,
        )
        check_to_variable = np.zeros_like(variable_to_check)
        maximum_change = math.inf
        converged = False
        iterations = 0
        for iterations in range(1, self.maximum_iterations + 1):
            check_to_variable, check_change = _check_update(
                syndrome,
                self.compiled.check_edge_offsets,
                self.compiled.check_edges,
                variable_to_check,
                check_to_variable,
                self.damping,
                self.message_clip,
            )
            variable_to_check, variable_change = _variable_update(
                self.compiled.state_offsets,
                self.compiled.state_log_priors,
                self.compiled.state_detector_words,
                self.compiled.variable_edge_offsets,
                self.compiled.edge_checks,
                check_to_variable,
                variable_to_check,
                self.damping,
                self.message_clip,
            )
            maximum_change = max(check_change, variable_change)
            if maximum_change < self.convergence_tolerance:
                converged = True
                break
        state_logs, hard_codes, component_llrs = _posterior(
            self.compiled.state_offsets,
            self.compiled.state_codes,
            self.compiled.state_log_priors,
            self.compiled.state_detector_words,
            self.compiled.variable_edge_offsets,
            self.compiled.edge_checks,
            check_to_variable,
            self.compiled.component_offsets,
        )
        hard_mask = _mask_from_codes(self.graph, hard_codes)
        return BPResult(
            hard_codes=hard_codes,
            state_log_posteriors=state_logs,
            component_llrs=component_llrs,
            iterations=iterations,
            converged=converged,
            maximum_message_change=float(maximum_change),
            hard_detector_mask=hard_mask,
            residual_detector_mask=hard_mask ^ int(observed_detector_mask),
        )
