"""Tests for sparse-F, affine-solved-G row-space utilities."""

from __future__ import annotations

import numpy as np

from reverse_geometry.n192 import LATTICE_ORDER, cancellation_witness
from reverse_geometry.rowspace import (
    affine_g_system,
    checks_from_coefficients,
    displayed_partner_coefficients,
    independent_affine_rows,
    logical_compatibility_constraints,
    rowspace_zx_dual,
)


def _coefficients(indices: list[int]) -> np.ndarray:
    output = np.zeros(6 * LATTICE_ORDER, dtype=np.uint8)
    output[indices] = 1
    return output


def test_affine_reduction_detects_consistent_and_inconsistent_systems() -> None:
    matrix = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    assert independent_affine_rows(matrix, np.asarray([1, 1, 0], dtype=np.uint8)) is not None
    assert independent_affine_rows(matrix, np.asarray([1, 0, 0], dtype=np.uint8)) is None


def test_known_displayed_fold_pair_satisfies_general_affine_system() -> None:
    terms = [0, LATTICE_ORDER]
    ff = _coefficients(terms)
    gg = displayed_partner_coefficients(ff)
    matrix, target = affine_g_system(cancellation_witness(), ff)
    assert not np.any((matrix @ gg) % 2 != target)
    compatibility = logical_compatibility_constraints(cancellation_witness())
    assert not np.any((compatibility @ ff) % 2)
    matrix_x, matrix_z = checks_from_coefficients(ff, gg)
    assert not np.any((matrix_x @ matrix_z.T) % 2)
    assert rowspace_zx_dual(matrix_x, matrix_z)
