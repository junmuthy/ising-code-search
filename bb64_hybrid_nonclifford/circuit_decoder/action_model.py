"""Canonical STAR repair actions shared by bounded and scalable decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bb64_hybrid_nonclifford.model import HybridModel

from .branches import branch_action_data
from .frames import (
    BoundaryFrame,
    BoundaryFrameDecomposer,
    GF2RowReducer,
    bits_to_int,
)


@dataclass(frozen=True)
class ScheduledRepairAction:
    physical_x_correction: tuple[int, ...]
    physical_z_correction: tuple[int, ...]
    logical_x_frame: tuple[int, ...]
    logical_z_frame: tuple[int, ...]
    signed_residual_angles: tuple[float, ...]


def _indices(mask: int, width: int = 64) -> tuple[int, ...]:
    return tuple(index for index in range(width) if (mask >> index) & 1)


def _frame_bits(mask: int) -> tuple[int, ...]:
    return tuple((mask >> index) & 1 for index in range(8))


def _rounded_angles(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 15) for value in values)


class RepairActionBuilder:
    """Map a latent TMR class and physical fault payload to one action key."""

    def __init__(self, model: HybridModel, *, theta: float):
        self.model = model
        self.theta = float(theta)
        self.decomposer = BoundaryFrameDecomposer(model.code)
        self.x_stabilizers = GF2RowReducer(model.code.matrix_x)
        self.z_stabilizers = GF2RowReducer(model.code.matrix_z)

    def from_frame(
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
        physical_x = self.x_stabilizers.reduce(frame.physical_x_correction)
        physical_z = self.z_stabilizers.reduce(frame.physical_z_correction ^ branch_z)
        logical_z = frame.logical_z_frame ^ branch.angle_z_frame
        action = ScheduledRepairAction(
            physical_x_correction=_indices(physical_x),
            physical_z_correction=_indices(physical_z),
            logical_x_frame=_frame_bits(frame.logical_x_frame),
            logical_z_frame=_frame_bits(logical_z),
            signed_residual_angles=_rounded_angles(branch.signed_residual_angles),
        )
        return action, branch.probability

    def from_signature(
        self,
        class_id: int,
        *,
        final_x_mask: int,
        final_z_mask: int,
        rotation_sign_mask: int,
    ) -> tuple[ScheduledRepairAction, float]:
        return self.from_frame(
            class_id,
            self.decomposer.decompose(final_x_mask, final_z_mask),
            rotation_sign_mask,
        )
