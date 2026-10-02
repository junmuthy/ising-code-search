"""Regression tests for the isolated natural-S3 reverse search."""

from __future__ import annotations

import numpy as np

from searches.reverse_geometry.n192 import (
    NUM_QUBITS,
    TARGET_LOGICALS,
    analyze_seed,
    cancellation_witness,
    fold_tied_annihilator,
    fold_tied_basis_checks,
    fold_internal_matrix,
    search_sparse_fold_tied_generators,
    standard_transposition_fold,
    matrices_from_terms,
    pairing_kernel_mask,
    top_parity_projections,
)


def test_weight_seven_cancellation_witness_has_ideal_geometry() -> None:
    fold = standard_transposition_fold()
    result = analyze_seed(cancellation_witness(), fold)
    assert fold.fixed_sheets == (0, 3)
    assert result["weight"] == 7
    assert result["sheet_occupancies"] == [1, 2, 2, 0, 1, 1]
    assert result["batching"]["disjoint_within_each_batch"]
    assert not result["batching"]["all_at_once_disjoint"]
    assert result["translation_orbit_rank"] == TARGET_LOGICALS
    assert result["pairing_is_identity"]
    assert pairing_kernel_mask(cancellation_witness(), fold) == 1
    assert [np.count_nonzero(item) for item in top_parity_projections(cancellation_witness())] == [1, 0]


def test_fold_tied_basis_has_exact_displayed_zx_relation() -> None:
    fold = standard_transposition_fold()
    qq = fold_internal_matrix(fold)
    data_fold = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    data_fold[: len(qq), : len(qq)] = qq
    data_fold[len(qq) :, len(qq) :] = qq
    checks_x, checks_z = fold_tied_basis_checks()
    for matrix_x, matrix_z in zip(checks_x[::37], checks_z[::37], strict=True):
        assert np.array_equal(matrix_z, (qq @ matrix_x @ data_fold.T) % 2)


def test_fold_tied_annihilator_is_nonempty_and_rank_capable() -> None:
    result = fold_tied_annihilator(cancellation_witness())
    assert result["coefficient_variables"] == 192
    assert result["constraint_nullity"] > 0
    assert result["exact_displayed_zx_relation"]
    assert result["rank_capacity_survives_k32_target"]


def test_sparse_generator_pilot_respects_weight_twelve_bound() -> None:
    result = search_sparse_fold_tied_generators(
        cancellation_witness(),
        minimum_terms=2,
        maximum_terms=2,
        solutions=2,
        seconds_per_solution=2,
        random_seed=192,
    )
    assert result["solutions_found"] >= 1
    assert all(record["logical_orbit_in_x_kernel"] for record in result["records"])
    assert all(record["maximum_check_weight"] <= 12 for record in result["records"])


def test_saved_term_serialization_reconstructs_checks() -> None:
    terms = [
        {"top": 2, "x": 5, "y": 0},
        {"top": 3, "x": 5, "y": 0},
        {"top": 3, "x": 5, "y": 1},
        {"top": 4, "x": 5, "y": 1},
        {"top": 4, "x": 7, "y": 3},
        {"top": 5, "x": 7, "y": 3},
    ]
    matrix_x, matrix_z = matrices_from_terms(terms)
    assert matrix_x.shape == (96, 192)
    assert matrix_z.shape == (96, 192)
    assert not np.any((matrix_x @ matrix_z.T) % 2)
