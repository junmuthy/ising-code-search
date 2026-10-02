"""Regression tests for the self-dual ``C4 x C5`` search."""

from __future__ import annotations

import numpy as np

from searches.c4xc5_search.search import (
    NUM_QUBITS,
    analyze_logical_orbit,
    build_checks,
    c4_fibres,
    find_graph_seed,
    translation_orbit,
)
from searches.c4xc5_search.general import (
    build_checks as build_general_checks,
    rowspace_zx_dual,
    standard_fold,
)
from searches.c4xc5_search.seeded import (
    analyze_known_seed,
    multiply_supports,
    polynomial_pair_from_seed,
)
from searches.c4xc5_search.solved import (
    compatible_sparse_pairs,
    seed_constraint_syndromes,
)
import random
from searches.c4xc5_search.automorphism_dual import (
    INVOLUTIONS,
    automorphism_fold,
    find_automorphism_seed,
    rowspace_automorphism_dual,
    transformed_partner_support,
)


def test_c4_translation_has_ten_physical_fibres() -> None:
    fibres = c4_fibres()
    assert len(fibres) == 10
    assert all(len(fibre) == 4 for fibre in fibres)
    assert sorted(qubit for fibre in fibres for qubit in fibre) == list(
        range(NUM_QUBITS)
    )


def test_compiled_solver_finds_weight_seven_seed_for_zero_check() -> None:
    check = np.zeros((20, 40), dtype=np.uint8)
    support = find_graph_seed(check)
    assert support is not None
    assert len(support) == 7
    logical = analyze_logical_orbit(check, support)
    assert logical["pairwise_disjoint"]
    assert logical["orbit_in_kernel"]
    assert logical["zx_pairing_is_identity"]


def test_self_dual_checks_are_css() -> None:
    matrix_x, matrix_z = build_checks((0, 1, 6, 12, 17, 19))
    assert np.array_equal(matrix_x, matrix_z)
    assert not np.any((matrix_x @ matrix_z.T) % 2)


def test_general_two_polynomial_checks_are_css_and_zx_dual() -> None:
    matrix_x, matrix_z = build_general_checks((0, 2, 7), (1, 8, 13, 19))
    assert not np.any((matrix_x @ matrix_z.T) % 2)
    assert rowspace_zx_dual(matrix_x, matrix_z)
    fold = standard_fold((0, 0))
    assert np.array_equal(fold[fold], np.arange(40))


def test_seeded_syzygy_forces_a_disjoint_kernel_orbit() -> None:
    seed_support = (0, 6, 12, 18, 20, 26, 32)
    factor = (0,)
    support_a, support_b = polynomial_pair_from_seed(seed_support, factor)
    matrix_x, matrix_z = build_general_checks(support_a, support_b)
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(seed_support)] = 1
    orbit = translation_orbit(seed)
    assert not np.any((matrix_x @ orbit.T) % 2)
    assert multiply_supports(factor, support_a) == support_a
    witness = analyze_known_seed(matrix_x, matrix_z, seed_support)
    if witness is not None:
        assert witness["pairwise_disjoint"]
        assert witness["zx_pairing_rank"] == 4


def test_sparse_solver_returns_only_seed_compatible_pairs() -> None:
    seed_support = (0, 6, 12, 18, 20, 26, 32)
    syndromes = seed_constraint_syndromes(seed_support)
    pairs = compatible_sparse_pairs(
        seed_support,
        rng=random.Random(4),
        restarts=1,
        maximum_pairs=4,
    )
    assert pairs
    for support_a, support_b in pairs:
        syndrome = 0
        for item in support_a:
            syndrome ^= syndromes[item]
        for item in support_b:
            syndrome ^= syndromes[20 + item]
        assert syndrome == 0
        assert len(support_a) + len(support_b) in (6, 8, 10, 12)


def test_half_preserving_automorphism_ansatz_is_zx_dual() -> None:
    support_a = (0, 1, 7, 13)
    for sigma in INVOLUTIONS:
        support_b = transformed_partner_support(support_a, sigma)
        matrix_x, matrix_z = build_general_checks(support_a, support_b)
        assert rowspace_automorphism_dual(matrix_x, matrix_z, sigma)
        fold = automorphism_fold(sigma)
        assert np.array_equal(fold[fold], np.arange(NUM_QUBITS))
