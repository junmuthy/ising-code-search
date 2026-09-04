"""Signed TMR branch probabilities and angles around non-Clifford faults."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from bb64_tmr_postselection.circuit import physical_tmr_angle
from bb64_tmr_postselection.model import BATCHES

from bb64_hybrid_nonclifford.model import HybridModel


@dataclass(frozen=True)
class BranchActionData:
    probability: float
    logical_angles: tuple[float, ...]
    angle_z_frame: int
    signed_residual_angles: tuple[float, ...]


def rotation_layout(model: HybridModel) -> tuple[tuple[int, int, int], ...]:
    """Return ``(logical, piece, pivot)`` in the scheduled R_Z target order."""

    logical_degree = np.count_nonzero(model.code.logical_z, axis=0)
    layout: list[tuple[int, int, int]] = []
    for batch in BATCHES:
        for logical in batch:
            for piece, support in enumerate(model.recovery.partitions[logical]):
                pivot = min(map(int, support), key=lambda qubit: (int(logical_degree[qubit]), qubit))
                layout.append((int(logical), int(piece), int(pivot)))
    if len(layout) != 24 or len({(logical, piece) for logical, piece, _pivot in layout}) != 24:
        raise AssertionError("scheduled rotation layout is not an 8 by 3 bijection")
    return tuple(layout)


def rotation_sign_array(model: HybridModel, rotation_sign_mask: int) -> np.ndarray:
    signs = np.ones((8, 3), dtype=np.int8)
    for index, (logical, piece, _pivot) in enumerate(rotation_layout(model)):
        if (rotation_sign_mask >> index) & 1:
            signs[logical, piece] = -1
    return signs


def _fold_rotation_angle(angle: float) -> tuple[float, int]:
    frame = 0
    while angle >= math.pi / 2:
        angle -= math.pi
        frame ^= 1
    while angle < -math.pi / 2:
        angle += math.pi
        frame ^= 1
    return angle, frame


def signed_local_branch(theta: float, selected: np.ndarray, signs: np.ndarray) -> tuple[float, float, int]:
    """Return probability, angle, and Z-frame for one signed three-piece coset."""

    q = np.asarray(selected, dtype=np.uint8)
    sigma = np.asarray(signs, dtype=np.int8)
    if q.shape != (3,) or sigma.shape != (3,) or np.any((q != 0) & (q != 1)) or np.any(np.abs(sigma) != 1):
        raise ValueError("one local branch requires three selection bits and three signs")
    alpha = physical_tmr_angle(theta, 3)
    angles = sigma.astype(np.float64) * alpha

    def coefficient(mask: np.ndarray) -> complex:
        value = 1.0 + 0.0j
        for bit, angle in zip(mask, angles, strict=True):
            value *= -1j * math.sin(angle / 2) if bit else math.cos(angle / 2)
        return value

    a = coefficient(q)
    b = coefficient(q ^ 1)
    probability = float(abs(a) ** 2 + abs(b) ** 2)
    if probability <= 0 or abs(a + b) == 0:
        raise ValueError("degenerate signed TMR branch")
    angle = float(np.angle((a - b) / (a + b)))
    angle, z_frame = _fold_rotation_angle(angle)
    normalized = np.array((a, b), dtype=np.complex128) / math.sqrt(probability)
    if abs(np.real(normalized[0] * normalized[1].conjugate())) > 1e-12:
        raise AssertionError("signed branch is not proportional to a unitary Z rotation")
    return probability, angle, z_frame


def branch_action_data(
    model: HybridModel,
    *,
    theta: float,
    class_id: int,
    rotation_sign_mask: int,
    logical_x_frame: int,
) -> BranchActionData:
    canonical = model.recovery.canonical_piece_masks[int(class_id)].reshape(8, 3)
    signs = rotation_sign_array(model, rotation_sign_mask)
    probability = 1.0
    angles: list[float] = []
    z_frame = 0
    residuals: list[float] = []
    for logical in range(8):
        local_probability, angle, local_frame = signed_local_branch(
            theta, canonical[logical], signs[logical]
        )
        probability *= local_probability
        angles.append(angle)
        z_frame |= int(local_frame) << logical
        residual = float(theta - angle)
        if (logical_x_frame >> logical) & 1:
            residual = -residual
        residuals.append(residual)
    return BranchActionData(
        probability=float(probability),
        logical_angles=tuple(angles),
        angle_z_frame=z_frame,
        signed_residual_angles=tuple(residuals),
    )
