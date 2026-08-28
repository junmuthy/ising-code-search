"""Regression tests for the half-checkerboard seed-first search."""

from __future__ import annotations

import numpy as np

from gala_search.half_grid import (
    NUM_LOGICALS,
    NUM_QUBITS,
    StructuredGLFold,
    add_automatic_css_relation,
    analyze_fold_seed,
    build_half_grid_code,
    independent_constraint_rows,
    polynomial_constraint_matrix,
    solve_sparse_polynomial_constraints,
    standard_fold_control,
)
from gala_search.s3_ising import ProductMonomial
from gala_search.s3_linear import linear_internal_translation_orbits


def test_half_grid_has_eight_disjoint_translation_fibres() -> None:
    orbits = linear_internal_translation_orbits(4)
    assert len(orbits) == 8
    assert all(len(orbit) == 32 for orbit in orbits)
    assert len(set().union(*map(set, orbits))) == NUM_QUBITS


def test_coupled_fold_has_full_rank_weight_six_seed() -> None:
    # Deterministic witness found by the cheap fold/seed precheck.  It is not
    # asserted to be a stabilizer-code logical until generators are found.
    fold = StructuredGLFold((0, 1, 3, 2), (1, 2, 0, 3))
    support = (56, 213, 117, 67, 10, 168)
    analysis = analyze_fold_seed(support, fold)
    assert analysis["graph_supported"]
    assert analysis["pairwise_disjoint"]
    assert analysis["physical_orbit_rank"] == NUM_LOGICALS
    assert analysis["zx_pairing_rank"] == NUM_LOGICALS


def test_standard_top_independent_fold_is_rank_deficient_on_witness() -> None:
    fold = StructuredGLFold((1, 0, 3, 2), (1, 0, 3, 2))
    support = (56, 213, 117, 67, 10, 168)
    analysis = analyze_fold_seed(support, fold)
    assert analysis["zx_pairing_rank"] < NUM_LOGICALS


def test_standard_fold_control_does_not_report_full_rank() -> None:
    result = standard_fold_control(seed=11, trials_per_fold=20)
    assert all(
        item["maximum_sampled_rank"] < NUM_LOGICALS
        for item in result["folds"].values()
    )


def test_polynomial_constraints_use_independent_512_coefficient_basis() -> None:
    data_fold = StructuredGLFold((0, 1, 3, 2), (1, 2, 0, 3))
    check_fold = StructuredGLFold((0, 1), (0, 1))
    matrix = polynomial_constraint_matrix(
        (56, 213, 117, 67, 10, 168),
        data_fold=data_fold,
        check_fold=check_fold,
    )
    assert matrix.shape == (128 + 128 * 256, 4 * 4 * 8 * 4)
    basis = independent_constraint_rows(matrix)
    assert basis.shape[1] == 512
    assert not np.any((basis @ np.zeros(512, dtype=np.uint8)) % 2)


def test_known_coupled_fold_has_exact_rank_capacity_obstruction() -> None:
    data_fold = StructuredGLFold((0, 1, 2, 3), (0, 3, 2, 1))
    check_fold = StructuredGLFold((0, 1), (1, 0))
    matrix = polynomial_constraint_matrix(
        (0, 34, 70, 103, 132, 182),
        data_fold=data_fold,
        check_fold=check_fold,
    )
    result = solve_sparse_polynomial_constraints(matrix, maximum_terms=16)
    assert result["search_method"] == "exact_linear_rank_capacity"
    assert result["maximum_possible_rank_x_upper_bound"] == 32
    assert result["minimum_possible_k_lower_bound"] == 192


def test_rank_probe_retains_an_exact_target_rank_candidate() -> None:
    # Witness 31/check fold 0 samples rank 112 frequently.  Keeping this
    # candidate (not merely the maximum-rank one) lets the runner perform all
    # exact CSS, dimension, fold, and logical-grid checks.
    data_fold = StructuredGLFold((0, 1, 2, 3), (2, 3, 0, 1))
    check_fold = StructuredGLFold((0, 1), (0, 1))
    matrix = polynomial_constraint_matrix(
        (64, 105, 155, 172, 211, 228),
        data_fold=data_fold,
        check_fold=check_fold,
    )
    result = solve_sparse_polynomial_constraints(
        matrix,
        rank_probe_trials=500,
        random_seed=260857,
        probe_only=True,
    )
    assert result["rank_probe"]["target_rank_observed"]
    assert result["entries"] is not None
    assert result["objective_terms"] == result["rank_probe"][
        "best_target_rank_coefficient_weight"
    ]


def test_identity_and_swap_relations_are_automatic_css() -> None:
    f0 = (ProductMonomial(1, 0, 0), ProductMonomial(3, 1, 2))
    f1 = (
        ProductMonomial(2, 2, 1),
        ProductMonomial(0, 3, 3),
        ProductMonomial(1, 4, 0),
    )
    for entries in ((f0, f1, f0, f1), (f0, f1, f1, f0)):
        code = build_half_grid_code(entries)
        matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
        matrix_z = np.asarray(code.matrix_z, dtype=np.uint8)
        assert not np.any((matrix_x @ matrix_z.T) % 2)

    shifted_f0 = tuple(
        ProductMonomial(term.top, (term.x + 3) % 8, (term.y + 1) % 4)
        for term in f0
    )
    shifted_f1 = tuple(
        ProductMonomial(term.top, (term.x + 3) % 8, (term.y + 1) % 4)
        for term in f1
    )
    shifted = build_half_grid_code((f0, f1, shifted_f1, shifted_f0))
    matrix_x = np.asarray(shifted.matrix_x, dtype=np.uint8)
    matrix_z = np.asarray(shifted.matrix_z, dtype=np.uint8)
    assert not np.any((matrix_x @ matrix_z.T) % 2)


def test_automatic_css_relation_adds_exact_coefficient_equalities() -> None:
    matrix = np.zeros((0, 512), dtype=np.uint8)
    identity = add_automatic_css_relation(matrix, "identity")
    swap = add_automatic_css_relation(matrix, "swap")
    shifted = add_automatic_css_relation(matrix, "swap", x_shift=3, y_shift=1)
    assert independent_constraint_rows(identity).shape == (256, 512)
    assert independent_constraint_rows(swap).shape == (256, 512)
    assert independent_constraint_rows(shifted).shape == (256, 512)
