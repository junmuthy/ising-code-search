"""Exact classical labeled trajectories for the D3 Pauli-plus-TMR model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bb64_hybrid_nonclifford.model import HybridModel

from .action_model import RepairActionBuilder, ScheduledRepairAction
from .factor_graph import D3FactorGraph
from .propagate import PauliSignature


@dataclass(frozen=True)
class LabeledTrajectory:
    state_codes: tuple[int, ...]
    observed_detector_mask: int
    fault_count: int
    branch_class_id: int
    physical_signature: PauliSignature
    correct_action: ScheduledRepairAction
    log_probability: float


def sample_state_codes(graph: D3FactorGraph, rng: np.random.Generator) -> tuple[int, ...]:
    codes = []
    for variable in graph.variables:
        probabilities = np.exp(np.asarray(variable.log_priors, dtype=np.float64))
        codes.append(int(rng.choice(len(probabilities), p=probabilities)))
    return tuple(codes)


def labeled_trajectory(
    graph: D3FactorGraph,
    model: HybridModel,
    action_builder: RepairActionBuilder,
    state_codes: tuple[int, ...],
) -> LabeledTrajectory:
    codes = graph.validate_codes(state_codes)
    physical = graph.physical_signature(codes)
    class_id = graph.branch_class_id(model, codes)
    action, _branch_probability = action_builder.from_signature(
        class_id,
        final_x_mask=physical.final_x_mask,
        final_z_mask=physical.final_z_mask,
        rotation_sign_mask=physical.rotation_sign_mask,
    )
    return LabeledTrajectory(
        state_codes=codes,
        observed_detector_mask=graph.detector_mask(codes),
        fault_count=graph.fault_count(codes),
        branch_class_id=class_id,
        physical_signature=physical,
        correct_action=action,
        log_probability=graph.log_probability(codes),
    )


def sample_labeled_trajectory(
    graph: D3FactorGraph,
    model: HybridModel,
    action_builder: RepairActionBuilder,
    rng: np.random.Generator,
) -> LabeledTrajectory:
    return labeled_trajectory(
        graph,
        model,
        action_builder,
        sample_state_codes(graph, rng),
    )
