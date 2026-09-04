"""Translate decoder output into reset or signed non-Clifford repair actions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table

from .decoder import DecoderResult


@dataclass(frozen=True)
class RepairAction:
    reset: bool
    reset_reason: str | None
    threshold: int
    alternative_mask: tuple[int, ...]
    logical_x_frame: tuple[int, ...]
    logical_z_frame: tuple[int, ...]
    signed_residual_angles: tuple[float, ...]
    repaired_logicals: tuple[int, ...]


@lru_cache(maxsize=32)
def _angle_parameters(theta: float) -> tuple[float, float]:
    table = angle_table(theta)
    return (
        float(table["alternative"]["logical_angle"]),
        float(theta - table["alternative"]["logical_angle"]),
    )


def repair_action(
    decoded: DecoderResult,
    *,
    theta: float,
    threshold: int,
    logical_x_frame: Sequence[int] = (0,) * 8,
    logical_z_frame: Sequence[int] = (0,) * 8,
) -> RepairAction:
    """Return a frame-aware action that produces all eight target resources."""

    if not 0 <= threshold <= 8:
        raise ValueError("threshold must lie in 0..8")
    frame_x = np.asarray(logical_x_frame, dtype=np.uint8)
    frame_z = np.asarray(logical_z_frame, dtype=np.uint8)
    if frame_x.shape != (8,) or frame_z.shape != (8,) or np.any(frame_x > 1) or np.any(frame_z > 1):
        raise ValueError("logical frames must contain eight binary values")
    mask = np.asarray(decoded.alternative_mask, dtype=np.uint8)
    _, residual = _angle_parameters(float(theta))
    angles = np.zeros(8, dtype=np.float64)
    angles[mask.astype(bool)] = residual
    angles[frame_x.astype(bool)] *= -1

    reason = None
    if decoded.ambiguous:
        reason = "decoder_ambiguity"
    elif decoded.bad_count > threshold:
        reason = "repair_threshold"
    if reason is not None:
        angles[:] = 0.0
    return RepairAction(
        reset=reason is not None,
        reset_reason=reason,
        threshold=int(threshold),
        alternative_mask=tuple(map(int, mask)),
        logical_x_frame=tuple(map(int, frame_x)),
        logical_z_frame=tuple(map(int, frame_z)),
        signed_residual_angles=tuple(map(float, angles)),
        repaired_logicals=tuple(map(int, np.flatnonzero(mask))) if reason is None else (),
    )


def verify_action_angles(decoded: DecoderResult, action: RepairAction, *, theta: float) -> float:
    """Return the maximum noiseless target-angle error before Pauli frames."""

    if action.reset:
        return math.nan
    alternative, _ = _angle_parameters(float(theta))
    target = float(theta)
    initial = np.where(np.asarray(decoded.alternative_mask), alternative, target)
    frame_sign = np.where(np.asarray(action.logical_x_frame), -1.0, 1.0)
    # Moving an existing X frame through a Z rotation changes the physical
    # command sign; in the corrected logical frame the added angle is restored.
    corrected_added = frame_sign * np.asarray(action.signed_residual_angles)
    final = initial + corrected_added
    return float(np.max(np.abs(final - theta)))
