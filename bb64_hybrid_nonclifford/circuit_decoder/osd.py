"""BP-guided ordered-statistics list construction for the D3 graph."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from bb64_hybrid_nonclifford.model import HybridModel

from .belief_propagation import BPResult
from .factor_graph import D3FactorGraph
from .propagate import PauliSignature


@dataclass(frozen=True)
class FaultHypothesis:
    state_codes: tuple[int, ...]
    log_probability: float
    fault_count: int
    branch_class_id: int
    physical_signature: PauliSignature


@dataclass(frozen=True)
class OSDResult:
    candidates: tuple[FaultHypothesis, ...]
    basis_rank: int
    free_window: int
    order: int
    attempted_patterns: int
    exact_syndrome_patterns: int


@dataclass
class _BasisEntry:
    vector: int
    components: int


def _weighted_basis(
    columns: tuple[int, ...],
    order: np.ndarray,
) -> tuple[dict[int, _BasisEntry], set[int]]:
    basis: dict[int, _BasisEntry] = {}
    selected: set[int] = set()
    for raw_component in order:
        component = int(raw_component)
        vector = int(columns[component])
        components = 1 << component
        while vector:
            pivot = vector.bit_length() - 1
            entry = basis.get(pivot)
            if entry is None:
                basis[pivot] = _BasisEntry(vector, components)
                selected.add(component)
                break
            vector ^= entry.vector
            components ^= entry.components
    return basis, selected


def _solve(basis: dict[int, _BasisEntry], syndrome: int) -> int | None:
    vector = int(syndrome)
    components = 0
    while vector:
        pivot = vector.bit_length() - 1
        entry = basis.get(pivot)
        if entry is None:
            return None
        vector ^= entry.vector
        components ^= entry.components
    return components


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


def _apply_delta(
    hard_codes: np.ndarray,
    delta: int,
    component_variables: np.ndarray,
    component_bits: np.ndarray,
) -> tuple[int, ...]:
    codes = hard_codes.astype(np.uint8, copy=True)
    value = int(delta)
    while value:
        least = value & -value
        component = least.bit_length() - 1
        codes[component_variables[component]] ^= np.uint8(1 << int(component_bits[component]))
        value ^= least
    return tuple(map(int, codes))


def _hypothesis(
    graph: D3FactorGraph,
    model: HybridModel,
    codes: tuple[int, ...],
) -> FaultHypothesis:
    return FaultHypothesis(
        state_codes=codes,
        log_probability=graph.log_probability(codes),
        fault_count=graph.fault_count(codes),
        branch_class_id=graph.branch_class_id(model, codes),
        physical_signature=graph.physical_signature(codes),
    )


def osd_candidate_list(
    graph: D3FactorGraph,
    model: HybridModel,
    observed_detector_mask: int,
    bp: BPResult,
    *,
    order: int = 2,
    free_window: int = 64,
    maximum_candidates: int = 512,
) -> OSDResult:
    """Return exact-syndrome candidates near one BP hard decision."""

    if order not in (0, 1, 2, 3):
        raise ValueError("OSD order must lie in 0..3")
    if free_window < 0 or maximum_candidates <= 0:
        raise ValueError("OSD window and candidate limit must be positive")
    hard = np.asarray(bp.hard_codes, dtype=np.uint8)
    hard_bits = np.zeros(graph.num_components, dtype=np.uint8)
    for variable in range(graph.num_variables):
        code = int(hard[variable])
        for bit, component in enumerate(
            range(graph.component_offsets[variable], graph.component_offsets[variable + 1])
        ):
            hard_bits[component] = (code >> bit) & 1
    signed_cost = np.where(hard_bits == 0, bp.component_llrs, -bp.component_llrs)
    reliability_order = np.argsort(signed_cost, kind="stable")
    basis, selected = _weighted_basis(graph.component_detector_masks, reliability_order)
    free = [
        int(component)
        for component in reliability_order
        if int(component) not in selected
    ][:free_window]
    component_variables, component_bits = _component_variables(graph)
    residual = int(observed_detector_mask) ^ graph.detector_mask(tuple(map(int, hard)))
    deltas: set[int] = set()
    attempted = 0
    for local_order in range(order + 1):
        for chosen in itertools.combinations(free, local_order):
            attempted += 1
            free_delta = 0
            adjusted = residual
            for component in chosen:
                free_delta ^= 1 << component
                adjusted ^= graph.component_detector_masks[component]
            pivot_delta = _solve(basis, adjusted)
            if pivot_delta is not None:
                deltas.add(pivot_delta ^ free_delta)
    candidates: dict[tuple[int, ...], FaultHypothesis] = {}
    for delta in deltas:
        codes = _apply_delta(hard, delta, component_variables, component_bits)
        if graph.detector_mask(codes) != int(observed_detector_mask):
            raise AssertionError("OSD emitted a candidate with the wrong detector history")
        candidates[codes] = _hypothesis(graph, model, codes)
    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.log_probability, candidate.state_codes),
    )[:maximum_candidates]
    return OSDResult(
        candidates=tuple(ordered),
        basis_rank=len(basis),
        free_window=len(free),
        order=order,
        attempted_patterns=attempted,
        exact_syndrome_patterns=len(candidates),
    )
