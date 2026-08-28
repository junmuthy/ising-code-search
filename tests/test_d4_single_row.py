"""Regression tests for the faithful ``D4 x C4`` search at ``n=32``."""

from __future__ import annotations

import numpy as np

from d4_single_row.search import (
    D4_REFLECTION,
    D4_ROTATION,
    IDENTITY,
    NUM_COEFFICIENTS,
    NUM_FIBRES,
    NUM_QUBITS,
    TOP_MATRIX_BASIS,
    checks_from_coefficients,
    d4_group_matrices,
    gf2_nullspace,
    gf2_rank,
    identity_fold_constraints,
    d4_involution_permutations,
    exact_fold_constraints,
    lifted_top_permutation,
    select_seed_witnesses,
    select_fold_seed_witnesses,
    translation_orbit,
)


def test_faithful_square_action_and_represented_algebra() -> None:
    assert np.array_equal(np.linalg.matrix_power(D4_ROTATION, 4) % 2, IDENTITY)
    assert not np.array_equal((D4_ROTATION @ D4_ROTATION) % 2, IDENTITY)
    assert np.array_equal((D4_REFLECTION @ D4_REFLECTION) % 2, IDENTITY)
    assert np.array_equal(
        (D4_REFLECTION @ D4_ROTATION @ D4_REFLECTION) % 2,
        np.linalg.matrix_power(D4_ROTATION, 3) % 2,
    )
    assert len(d4_group_matrices()) == 8
    assert len(TOP_MATRIX_BASIS) == 6
    assert gf2_rank(np.asarray([item.ravel() for item in TOP_MATRIX_BASIS])) == 6


def test_compact_problem_has_eight_c4_fibres_at_n32() -> None:
    assert NUM_QUBITS == 32
    assert NUM_FIBRES == 8
    witness = select_seed_witnesses(count=1, random_seed=4)[0]
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[witness["seed_support"]] = 1
    orbit = translation_orbit(seed)
    assert np.all(np.sum(orbit, axis=0) <= 1)
    assert np.array_equal((orbit @ orbit.T) % 2, np.eye(4, dtype=np.uint8))


def test_identity_fold_nullspace_satisfies_seed_and_zx() -> None:
    witness = select_seed_witnesses(count=1, random_seed=5)[0]
    constraints = identity_fold_constraints(witness["seed_support"])
    basis = gf2_nullspace(constraints)
    assert basis.shape[1] == NUM_COEFFICIENTS
    assert len(basis)
    coefficients = basis[0]
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[witness["seed_support"]] = 1
    assert not np.any(matrix_x @ seed % 2)
    assert np.array_equal(matrix_x, matrix_z)


def test_d4_involution_folds_and_exact_constraint() -> None:
    involutions = d4_involution_permutations()
    assert len(involutions) == 6
    for permutation in involutions:
        lifted = lifted_top_permutation(permutation, blocks=2)
        assert np.array_equal(lifted[lifted], np.arange(NUM_QUBITS))
    witness = select_seed_witnesses(count=1, random_seed=8)[0]
    constraints = exact_fold_constraints(
        witness["seed_support"],
        data_top_permutation=involutions[1],
        check_top_permutation=involutions[2],
    )
    assert constraints.shape[1] == NUM_COEFFICIENTS
    witnesses = select_fold_seed_witnesses(
        witnesses_per_data_fold=1, random_seed=9
    )
    assert len(witnesses) == 3
    assert all(
        any(
            source == target
            for source, target in enumerate(item["data_top_permutation"])
        )
        for item in witnesses
    )
