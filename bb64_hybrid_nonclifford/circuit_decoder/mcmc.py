"""Detector-conditioned nullspace MCMC for D3 action posteriors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from bb64_hybrid_nonclifford.model import HybridModel

from .action_model import RepairActionBuilder, ScheduledRepairAction
from .belief_propagation import BPResult
from .factor_graph import D3FactorGraph
from .osd import FaultHypothesis, _solve, _weighted_basis


@dataclass(frozen=True)
class MCMCChainSummary:
    attempted_moves: int
    accepted_moves: int
    acceptance_fraction: float
    retained_samples: int
    distinct_retained_states: int
    distinct_retained_actions: int
    retained_action_transitions: int


@dataclass(frozen=True)
class MCMCPosterior:
    leading_action: ScheduledRepairAction | None
    leading_probability: float
    runner_up_probability: float
    leading_probability_lower_bound: float
    effective_sample_size: float
    maximum_chain_fraction_spread: float
    total_samples: int
    distinct_actions: int
    total_action_transitions: int
    action_counts: tuple[tuple[ScheduledRepairAction, int], ...]
    chains: tuple[MCMCChainSummary, ...]


def _component_variables(graph: D3FactorGraph) -> tuple[np.ndarray, np.ndarray]:
    variables = np.empty(graph.num_components, dtype=np.int32)
    bits = np.empty(graph.num_components, dtype=np.uint8)
    for variable in range(graph.num_variables):
        for bit, component in enumerate(
            range(graph.component_offsets[variable], graph.component_offsets[variable + 1])
        ):
            variables[component] = variable
            bits[component] = bit
    return variables, bits


def _kernel_generators(
    graph: D3FactorGraph,
    bp: BPResult,
) -> tuple[tuple[int, ...], np.ndarray]:
    hard_bits = np.zeros(graph.num_components, dtype=np.uint8)
    for variable in range(graph.num_variables):
        code = int(bp.hard_codes[variable])
        for bit, component in enumerate(
            range(graph.component_offsets[variable], graph.component_offsets[variable + 1])
        ):
            hard_bits[component] = (code >> bit) & 1
    signed_cost = np.where(hard_bits == 0, bp.component_llrs, -bp.component_llrs)
    reliability_order = np.argsort(signed_cost, kind="stable")
    basis, selected = _weighted_basis(graph.component_detector_masks, reliability_order)
    generators: list[int] = []
    costs: list[float] = []
    for component in range(graph.num_components):
        if component in selected:
            continue
        correction = _solve(basis, graph.component_detector_masks[component])
        if correction is None:
            continue
        generator = correction ^ (1 << component)
        if not generator:
            continue
        generators.append(generator)
        value = generator
        cost = 0.0
        while value:
            least = value & -value
            index = least.bit_length() - 1
            cost += max(0.0, float(signed_cost[index]))
            value ^= least
        costs.append(cost)
    if not generators:
        raise ValueError("factor graph has no detector-preserving kernel generators")
    # A fixed, state-independent proposal distribution keeps each involutory
    # toggle symmetric.  The small floor preserves irreducibility while BP
    # concentrates most proposals on plausible low-cost generators.
    scaled = np.exp(-0.2 * np.minimum(np.asarray(costs), 100.0))
    scaled += 1e-8 * float(np.max(scaled))
    scaled /= np.sum(scaled)
    return tuple(generators), scaled


def _toggle_codes(
    codes: np.ndarray,
    generator: int,
    variables: np.ndarray,
    bits: np.ndarray,
) -> list[tuple[int, int]]:
    changed: dict[int, int] = {}
    value = int(generator)
    while value:
        least = value & -value
        component = least.bit_length() - 1
        variable = int(variables[component])
        changed[variable] = changed.get(variable, 0) ^ (1 << int(bits[component]))
        value ^= least
    changes = [(variable, delta) for variable, delta in changed.items() if delta]
    return changes


def _action_for_codes(
    graph: D3FactorGraph,
    model: HybridModel,
    builder: RepairActionBuilder,
    codes: np.ndarray,
) -> ScheduledRepairAction:
    values = tuple(map(int, codes))
    physical = graph.physical_signature(values)
    class_id = graph.branch_class_id(model, values)
    action, _probability = builder.from_signature(
        class_id,
        final_x_mask=physical.final_x_mask,
        final_z_mask=physical.final_z_mask,
        rotation_sign_mask=physical.rotation_sign_mask,
    )
    return action


def _indicator_ess(values: np.ndarray, maximum_lag: int = 100) -> float:
    samples = np.asarray(values, dtype=np.float64)
    count = len(samples)
    if count <= 1:
        return float(count)
    centered = samples - np.mean(samples)
    variance = float(np.dot(centered, centered) / count)
    if variance <= 1e-15:
        # A chain that never leaves one action supplies no empirical mixing
        # evidence, even when that action ultimately is correct.
        return 0.0
    correlation_sum = 0.0
    for lag in range(1, min(maximum_lag, count - 1) + 1):
        correlation = float(np.dot(centered[:-lag], centered[lag:]) / ((count - lag) * variance))
        if correlation <= 0:
            break
        correlation_sum += correlation
    return count / (1.0 + 2.0 * correlation_sum)


def sample_action_posterior(
    graph: D3FactorGraph,
    model: HybridModel,
    builder: RepairActionBuilder,
    observed_detector_mask: int,
    bp: BPResult,
    starts: Sequence[FaultHypothesis],
    *,
    chains: int = 4,
    burn_in: int = 1000,
    retained_per_chain: int = 500,
    thin: int = 5,
    seed: int = 20260904,
) -> MCMCPosterior:
    """Sample the exact categorical posterior conditioned on all detectors."""

    if not starts or chains <= 0 or burn_in < 0 or retained_per_chain <= 0 or thin <= 0:
        raise ValueError("invalid MCMC configuration")
    generators, proposal = _kernel_generators(graph, bp)
    component_variables, component_bits = _component_variables(graph)
    rng = np.random.default_rng(seed)
    action_counts: dict[ScheduledRepairAction, int] = {}
    chain_actions: list[list[ScheduledRepairAction]] = []
    summaries: list[MCMCChainSummary] = []
    total_steps = burn_in + retained_per_chain * thin
    for chain in range(chains):
        codes = np.asarray(starts[chain % len(starts)].state_codes, dtype=np.uint8).copy()
        if graph.detector_mask(tuple(map(int, codes))) != int(observed_detector_mask):
            raise ValueError("MCMC starting state does not satisfy the observed detectors")
        accepted = 0
        retained: list[ScheduledRepairAction] = []
        retained_state_hashes: set[int] = set()
        transitions = 0
        for step in range(total_steps):
            generator_index = int(rng.choice(len(generators), p=proposal))
            changes = _toggle_codes(
                codes, generators[generator_index], component_variables, component_bits
            )
            delta_log_probability = 0.0
            for variable, delta in changes:
                old = int(codes[variable])
                new = old ^ delta
                delta_log_probability += (
                    graph.variables[variable].log_priors[new]
                    - graph.variables[variable].log_priors[old]
                )
            if delta_log_probability >= 0 or math.log(rng.random()) < delta_log_probability:
                for variable, delta in changes:
                    codes[variable] ^= np.uint8(delta)
                accepted += 1
            if step >= burn_in and (step - burn_in) % thin == 0:
                action = _action_for_codes(graph, model, builder, codes)
                if retained and action != retained[-1]:
                    transitions += 1
                retained.append(action)
                retained_state_hashes.add(hash(codes.tobytes()))
                action_counts[action] = action_counts.get(action, 0) + 1
        chain_actions.append(retained)
        summaries.append(
            MCMCChainSummary(
                attempted_moves=total_steps,
                accepted_moves=accepted,
                acceptance_fraction=accepted / total_steps,
                retained_samples=len(retained),
                distinct_retained_states=len(retained_state_hashes),
                distinct_retained_actions=len(set(retained)),
                retained_action_transitions=transitions,
            )
        )
    ordered = sorted(action_counts.items(), key=lambda item: (-item[1], repr(item[0])))
    total_samples = sum(action_counts.values())
    if not ordered:
        return MCMCPosterior(None, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, (), tuple(summaries))
    leading_action, leading_count = ordered[0]
    leading_probability = leading_count / total_samples
    runner_probability = ordered[1][1] / total_samples if len(ordered) > 1 else 0.0
    indicators = np.asarray(
        [action == leading_action for actions in chain_actions for action in actions],
        dtype=np.uint8,
    )
    ess = _indicator_ess(indicators)
    standard_error = math.sqrt(
        max(
            0.0,
            leading_probability * (1.0 - leading_probability) / max(ess, 1.0),
        )
    )
    lower = max(0.0, leading_probability - 1.96 * standard_error)
    fractions = [
        sum(action == leading_action for action in actions) / len(actions)
        for actions in chain_actions
    ]
    spread = max(fractions) - min(fractions)
    return MCMCPosterior(
        leading_action=leading_action,
        leading_probability=leading_probability,
        runner_up_probability=runner_probability,
        leading_probability_lower_bound=lower,
        effective_sample_size=ess,
        maximum_chain_fraction_spread=spread,
        total_samples=total_samples,
        distinct_actions=len(ordered),
        total_action_transitions=sum(chain.retained_action_transitions for chain in summaries),
        action_counts=tuple(ordered),
        chains=tuple(summaries),
    )
