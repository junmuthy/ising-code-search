"""Canonical physical corrections and logical frames at the repair boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bb64_tmr_postselection.model import CodeArtifact, initialization_correction_map


@dataclass(frozen=True)
class BoundaryFrame:
    physical_x_correction: int
    physical_z_correction: int
    logical_x_frame: int
    logical_z_frame: int


def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.fromiter(((value >> index) & 1 for index in range(width)), dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(np.asarray(bits, dtype=np.uint8)))


def _canonical_correction(
    error: np.ndarray,
    check: np.ndarray,
    rows: tuple[int, ...],
    right_inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    syndrome = (check @ error) % 2
    correction = (right_inverse @ syndrome[list(rows)]) % 2
    corrected = error ^ correction
    if np.any((check @ corrected) % 2):
        raise AssertionError("canonical correction did not return the error to the normalizer")
    return correction.astype(np.uint8), corrected.astype(np.uint8)


class BoundaryFrameDecomposer:
    """Reusable decomposition with the expensive GF(2) maps cached once."""

    def __init__(self, code: CodeArtifact):
        self.code = code
        self.x_rows, _x_qubits, self.x_inverse = initialization_correction_map(code.matrix_z)
        self.z_rows, _z_qubits, self.z_inverse = initialization_correction_map(code.matrix_x)

    def decompose(self, x_mask: int, z_mask: int) -> BoundaryFrame:
        x = int_to_bits(x_mask, self.code.matrix_z.shape[1])
        z = int_to_bits(z_mask, self.code.matrix_x.shape[1])
        x_correction, x_normalizer = _canonical_correction(
            x, self.code.matrix_z, self.x_rows, self.x_inverse
        )
        z_correction, z_normalizer = _canonical_correction(
            z, self.code.matrix_x, self.z_rows, self.z_inverse
        )
        logical_x = (self.code.logical_z @ x_normalizer) % 2
        logical_z = (self.code.logical_x @ z_normalizer) % 2
        return BoundaryFrame(
            physical_x_correction=bits_to_int(x_correction),
            physical_z_correction=bits_to_int(z_correction),
            logical_x_frame=bits_to_int(logical_x),
            logical_z_frame=bits_to_int(logical_z),
        )


def decompose_boundary_pauli(code: CodeArtifact, x_mask: int, z_mask: int) -> BoundaryFrame:
    """Split a data Pauli into canonical corrections and logical frame bits."""

    return BoundaryFrameDecomposer(code).decompose(x_mask, z_mask)
