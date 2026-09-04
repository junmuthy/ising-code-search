"""Frozen inputs and derived branch coordinates for hybrid BB64 recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bb64_syndrome_recovery.algebra import (
    NUM_LOGICALS,
    NUM_SYNDROME_CLASSES,
    RecoveryModel,
    build_recovery_model,
)


PACKAGE_DIR = Path(__file__).resolve().parent
RECOVERY_ARCHIVE = (
    PACKAGE_DIR.parent
    / "bb64_syndrome_recovery"
    / "results"
    / "ideal_recovery_pi32"
    / "syndrome_classes.npz"
)


@dataclass(frozen=True)
class HybridModel:
    """Recovery model plus linear two-bit coordinates for each logical branch."""

    recovery: RecoveryModel
    quotient_coordinates: np.ndarray
    syndrome_to_quotient: np.ndarray
    recovery_archive: Path

    @property
    def code(self):
        return self.recovery.code

    @property
    def displayed_syndromes(self) -> np.ndarray:
        return self.recovery.displayed_syndromes

    def class_ids_from_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        bits = np.asarray(coordinates, dtype=np.uint8)
        if bits.shape[-1] != 16:
            raise ValueError("syndrome coordinates must have width 16")
        powers = (1 << np.arange(16, dtype=np.uint32))
        return np.sum(bits.astype(np.uint32) * powers, axis=-1, dtype=np.uint32)


def _derive_quotient_coordinates(recovery: RecoveryModel) -> np.ndarray:
    """Map each local three-piece mask to two linear quotient coordinates.

    For a local occurrence mask ``(q0,q1,q2)``, the coordinates are
    ``(q0 xor q2, q1 xor q2)``.  Complementing all three pieces leaves these
    coordinates unchanged, and ``00`` is precisely the target-angle branch.
    """

    initial = recovery.initial_piece_masks.reshape(-1, NUM_LOGICALS, 3)
    quotient = np.empty((NUM_SYNDROME_CLASSES, NUM_LOGICALS, 2), dtype=np.uint8)
    quotient[:, :, 0] = initial[:, :, 0] ^ initial[:, :, 2]
    quotient[:, :, 1] = initial[:, :, 1] ^ initial[:, :, 2]
    if not np.array_equal(np.any(quotient, axis=2), recovery.alternative_mask.astype(bool)):
        raise ValueError("two-bit quotient coordinates do not reproduce the alternative mask")
    return quotient


def _derive_linear_map(recovery: RecoveryModel, quotient: np.ndarray) -> np.ndarray:
    """Return the GF(2) map from 16 syndrome coordinates to 16 quotient bits."""

    basis_rows = 1 << np.arange(16, dtype=np.uint32)
    linear_map = quotient[basis_rows].reshape(16, 16).T.copy()
    predicted = (recovery.syndrome_coordinates @ linear_map.T) % 2
    if not np.array_equal(predicted, quotient.reshape(NUM_SYNDROME_CLASSES, 16)):
        raise ValueError("derived quotient-coordinate map is not linear")
    return linear_map.astype(np.uint8)


def _validate_committed_archive(recovery: RecoveryModel, path: Path) -> None:
    archive = np.load(path)
    required = {
        "displayed_syndromes": recovery.displayed_syndromes,
        "local_branch_labels": recovery.local_branch_labels,
        "alternative_mask": recovery.alternative_mask,
        "correction_physical_supports": recovery.correction_physical_supports,
    }
    for name, expected in required.items():
        if name not in archive or not np.array_equal(archive[name], expected):
            raise ValueError(f"committed recovery archive mismatch for {name}")


def load_hybrid_model(*, validate_archive: bool = True) -> HybridModel:
    recovery = build_recovery_model()
    if validate_archive:
        _validate_committed_archive(recovery, RECOVERY_ARCHIVE)
    quotient = _derive_quotient_coordinates(recovery)
    linear_map = _derive_linear_map(recovery, quotient)
    return HybridModel(
        recovery=recovery,
        quotient_coordinates=quotient,
        syndrome_to_quotient=linear_map,
        recovery_archive=RECOVERY_ARCHIVE,
    )
