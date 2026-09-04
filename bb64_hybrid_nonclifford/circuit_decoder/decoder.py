"""Action-posterior decoder for scheduled BB64 detector histories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from bb64_hybrid_nonclifford.model import HybridModel

from .branches import branch_action_data
from .catalog import FaultCatalog
from .frames import BoundaryFrame, bits_to_int


@dataclass(frozen=True)
class ScheduledRepairAction:
    physical_x_correction: tuple[int, ...]
    physical_z_correction: tuple[int, ...]
    logical_x_frame: tuple[int, ...]
    logical_z_frame: tuple[int, ...]
    signed_residual_angles: tuple[float, ...]


@dataclass(frozen=True)
class ScheduledDecodeResult:
    reset: bool
    reset_reason: str | None
    action: ScheduledRepairAction | None
    modeled_posterior: float
    conservative_posterior_lower_bound: float
    runner_up_modeled_posterior: float
    matching_explanations: int
    distinct_actions: int
    modeled_evidence: float
    omitted_probability_upper_bound: float


def _indices(mask: int, width: int = 64) -> tuple[int, ...]:
    return tuple(index for index in range(width) if (mask >> index) & 1)


def _frame_bits(mask: int) -> tuple[int, ...]:
    return tuple((mask >> index) & 1 for index in range(8))


def _rounded_angles(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 15) for value in values)


class ScheduledActionDecoder:
    """Sum explanation probabilities by repair action and fail closed.

    The modeled catalog contains exact no-fault and single-fault mechanisms,
    plus any explicitly selected pairs.  All omitted probability is assigned
    adversarially against the leading action when computing the conservative
    posterior lower bound.
    """

    def __init__(
        self,
        model: HybridModel,
        catalog: FaultCatalog,
        *,
        theta: float,
        minimum_posterior: float = 0.99,
        maximum_repairs: int = 8,
    ):
        if not 0.0 <= minimum_posterior <= 1.0:
            raise ValueError("minimum_posterior must lie in [0,1]")
        if not 0 <= maximum_repairs <= 8:
            raise ValueError("maximum_repairs must lie in 0..8")
        self.model = model
        self.catalog = catalog
        self.theta = float(theta)
        self.minimum_posterior = float(minimum_posterior)
        self.maximum_repairs = int(maximum_repairs)
        self.class_from_syndrome = {
            bits_to_int(syndrome): class_id
            for class_id, syndrome in enumerate(model.displayed_syndromes)
        }
        used = {
            index
            for group in (*catalog.history.x_detector_groups, *catalog.history.z_detector_groups)
            for index in group
        }
        self.initialization_indices = tuple(
            index for index in range(catalog.circuit.num_detectors) if index not in used
        )

    def _ideal_class(self, detector_mask: int) -> int | None:
        if any((detector_mask >> index) & 1 for index in self.initialization_indices):
            return None
        for group in self.catalog.history.z_detector_groups:
            if any((detector_mask >> index) & 1 for index in group):
                return None
        first_group = self.catalog.history.x_detector_groups[0]
        syndrome = sum(((detector_mask >> index) & 1) << check for check, index in enumerate(first_group))
        for group in self.catalog.history.x_detector_groups[1:]:
            value = sum(((detector_mask >> index) & 1) << check for check, index in enumerate(group))
            if value != syndrome:
                return None
        return self.class_from_syndrome.get(syndrome)

    def _action(
        self,
        class_id: int,
        frame: BoundaryFrame,
        rotation_sign_mask: int,
    ) -> tuple[ScheduledRepairAction, float]:
        branch = branch_action_data(
            self.model,
            theta=self.theta,
            class_id=class_id,
            rotation_sign_mask=rotation_sign_mask,
            logical_x_frame=frame.logical_x_frame,
        )
        branch_z = bits_to_int(self.model.recovery.correction_physical_supports[class_id])
        logical_z = frame.logical_z_frame ^ branch.angle_z_frame
        action = ScheduledRepairAction(
            physical_x_correction=_indices(frame.physical_x_correction),
            physical_z_correction=_indices(frame.physical_z_correction ^ branch_z),
            logical_x_frame=_frame_bits(frame.logical_x_frame),
            logical_z_frame=_frame_bits(logical_z),
            signed_residual_angles=_rounded_angles(branch.signed_residual_angles),
        )
        return action, branch.probability

    def decode_mask(self, observed_detector_mask: int) -> ScheduledDecodeResult:
        p0 = self.catalog.ledger.no_fault_probability
        explanations: list[tuple[object, BoundaryFrame, float]] = [
            (self.catalog.no_fault_group.signature, self.catalog.no_fault_group.frame, p0)
        ]
        explanations.extend(
            (group.signature, group.frame, p0 * group.relative_weight)
            for group in self.catalog.single_groups
        )
        explanations.extend(
            (group.signature, group.frame, p0 * group.relative_weight)
            for group in self.catalog.pair_groups
        )

        action_weights: dict[ScheduledRepairAction, float] = {}
        matching = 0
        for signature, frame, mechanism_probability in explanations:
            corrected = int(observed_detector_mask) ^ signature.detector_mask
            class_id = self._ideal_class(corrected)
            if class_id is None:
                continue
            action, branch_probability = self._action(
                class_id, frame, signature.rotation_sign_mask
            )
            weight = mechanism_probability * branch_probability
            action_weights[action] = action_weights.get(action, 0.0) + weight
            matching += 1
        if not action_weights:
            return ScheduledDecodeResult(
                True,
                "outside_modeled_catalog",
                None,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0.0,
                self.catalog.conservative_tail_probability,
            )
        ordered = sorted(action_weights.items(), key=lambda item: -item[1])
        total = math.fsum(action_weights.values())
        best_action, best_weight = ordered[0]
        runner_weight = ordered[1][1] if len(ordered) > 1 else 0.0
        modeled_posterior = best_weight / total
        tail = self.catalog.conservative_tail_probability
        lower_bound = best_weight / (total + tail)
        repair_count = sum(abs(angle) > 1e-14 for angle in best_action.signed_residual_angles)
        reason = None
        if repair_count > self.maximum_repairs:
            reason = "repair_threshold"
        elif lower_bound < self.minimum_posterior:
            reason = "posterior_lower_bound"
        return ScheduledDecodeResult(
            reset=reason is not None,
            reset_reason=reason,
            action=None if reason is not None else best_action,
            modeled_posterior=modeled_posterior,
            conservative_posterior_lower_bound=lower_bound,
            runner_up_modeled_posterior=runner_weight / total,
            matching_explanations=matching,
            distinct_actions=len(action_weights),
            modeled_evidence=total,
            omitted_probability_upper_bound=tail,
        )

    def decode_detectors(self, detectors: np.ndarray) -> ScheduledDecodeResult:
        values = np.asarray(detectors, dtype=np.uint8)
        if values.shape != (self.catalog.circuit.num_detectors,) or np.any(values > 1):
            raise ValueError(
                f"detectors must be a binary vector of width {self.catalog.circuit.num_detectors}"
            )
        mask = sum(int(bit) << index for index, bit in enumerate(values))
        return self.decode_mask(mask)
