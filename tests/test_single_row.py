"""Regression tests for the ``[[32,4,6]]`` single-row search."""

from __future__ import annotations

import numpy as np

from gala_search.single_row import (
    NUM_COEFFICIENTS,
    NUM_LOGICALS,
    NUM_QUBITS,
    TARGET_CHECK_RANK,
    TranslationFold,
    add_automatic_css_relation,
    analyze_seed_fold,
    checks_from_coefficients,
    coefficient_checks,
    fixed_f_affine_system,
    gf2_nullspace,
    gf2_rank,
    iter_structured_gl_folds,
    polynomial_constraint_matrix,
    search_seed_fold_witnesses,
    solve_constraint_space,
    translation_orbit,
)


def test_weight_six_seed_translates_are_disjoint_with_eight_slack_qubits() -> None:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[[0, 4, 8, 12, 16, 20]] = 1
    orbit = translation_orbit(seed)
    assert orbit.shape == (NUM_LOGICALS, NUM_QUBITS)
    assert np.all(np.count_nonzero(orbit, axis=1) == 6)
    assert np.all(np.sum(orbit, axis=0) <= 1)
    assert np.count_nonzero(np.sum(orbit, axis=0) == 0) == 8


def test_seed_search_finds_full_rank_c4_pairing() -> None:
    result = search_seed_fold_witnesses(
        seed=1, trials_per_fold=2, target_witnesses=1
    )
    assert len(result["witnesses"]) == 1
    witness = result["witnesses"][0]
    fold = TranslationFold(
        tuple(witness["fold"]["fibre_images"]),
        tuple(witness["fold"]["fibre_shifts"]),
    )
    assert analyze_seed_fold(witness["seed_support"], fold)["zx_pairing_rank"] == 4


def test_represented_coefficient_basis_has_64_independent_variables() -> None:
    assert NUM_COEFFICIENTS == 64
    lifts = np.asarray(
        [
            np.concatenate([matrix_x.ravel(), matrix_z.ravel()])
            for matrix_x, matrix_z in coefficient_checks()
        ],
        dtype=np.uint8,
    )
    assert gf2_rank(lifts) == NUM_COEFFICIENTS


def test_all_automatic_relations_are_css() -> None:
    rng = np.random.default_rng(7)
    empty = np.zeros((0, NUM_COEFFICIENTS), dtype=np.uint8)
    for q_bits in range(1, 16):
        for a, b in ((1, 0), (0, 1), (1, 1)):
            constraints = add_automatic_css_relation(
                empty, q_bits=q_bits, a=a, b=b
            )
            basis = gf2_nullspace(constraints)
            generators = rng.integers(0, 2, len(basis), dtype=np.uint8)
            coefficients = (generators @ basis) % 2
            matrix_x, matrix_z = checks_from_coefficients(coefficients)
            assert not np.any((matrix_x @ matrix_z.T) % 2)


def test_fixed_f_affine_equations_match_binary_css_checks() -> None:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[[0, 4, 8, 12, 16, 20]] = 1
    f_support = (0, 5, 17, 30)
    _f_matrix_x, _f_matrix_z, matrix, target = fixed_f_affine_system(
        f_support, seed
    )
    assert set(np.unique(matrix)).issubset({0, 1})

    rng = np.random.default_rng(11)
    f_vector = np.zeros(32, dtype=np.uint8)
    f_vector[list(f_support)] = 1
    for _ in range(8):
        g_vector = rng.integers(0, 2, 32, dtype=np.uint8)
        matrix_x, matrix_z = checks_from_coefficients(
            np.concatenate([f_vector, g_vector])
        )
        residual = np.concatenate(
            [((matrix_x @ matrix_z.T) % 2).ravel(), (matrix_x @ seed) % 2]
        )
        assert np.array_equal(((matrix @ g_vector) % 2) ^ target, residual)


def test_known_seed_relation_has_exact_rank_capacity_no_go() -> None:
    result = search_seed_fold_witnesses(
        seed=1, trials_per_fold=2, target_witnesses=1
    )
    witness = result["witnesses"][0]
    data_fold = TranslationFold(
        tuple(witness["fold"]["fibre_images"]),
        tuple(witness["fold"]["fibre_shifts"]),
    )
    check_fold = next(iter_structured_gl_folds(2))
    constraints = polynomial_constraint_matrix(
        witness["seed_support"],
        data_fold=data_fold,
        check_fold=check_fold,
    )
    constraints = add_automatic_css_relation(
        constraints, q_bits=1, a=1, b=0
    )
    synthesis = solve_constraint_space(
        constraints,
        support=witness["seed_support"],
        data_fold=data_fold,
        check_fold=check_fold,
    )
    assert TARGET_CHECK_RANK == 14
    assert synthesis["status"] == "exact_rank_capacity_no_go"
    assert synthesis["rank_capacity_x"] == 4
