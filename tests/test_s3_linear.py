"""Regression tests for the faithful two-dimensional S3 lift."""

from __future__ import annotations

import numpy as np

from gala_search.s3_linear import (
    BLOCK_SIZE,
    build_linear_code,
    build_linear_fold_code,
    expected_zx_fold,
    is_zx_fold,
    linear_fold_active_orthogonality_data,
    linear_fold_permutation,
)
from gala_search.s3_many_copy_result import certified_s3_four_grid_candidate


def test_generic_linear_fold_matches_saved_l12_family() -> None:
    candidate = certified_s3_four_grid_candidate()
    saved = build_linear_code(candidate.entries, "l12-vertex-fold")
    generic = build_linear_fold_code(candidate.entries, active_rows=3, shift=0)
    assert saved.num_qubits == 12 * BLOCK_SIZE
    assert np.array_equal(saved.matrix_x, generic.matrix_x)
    assert np.array_equal(saved.matrix_z, generic.matrix_z)


def test_linear_l12_fold_has_active_orthogonality_and_zx_fold() -> None:
    candidate = certified_s3_four_grid_candidate()
    code = build_linear_fold_code(candidate.entries, active_rows=3, shift=0)
    orthogonality = linear_fold_active_orthogonality_data(
        candidate.entries, active_rows=3, shift=0
    )
    fold = linear_fold_permutation(half_blocks=6, active_rows=3, shift=0)
    assert orthogonality["active_offsets_zero"]
    assert orthogonality["some_latent_offset_nonzero"]
    assert is_zx_fold(code, fold)
    assert np.array_equal(fold, expected_zx_fold(12, "l12-vertex-fold"))
