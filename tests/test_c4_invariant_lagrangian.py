"""Tests for direct C4-invariant Lagrangian construction."""

from __future__ import annotations

import random

import numpy as np

from searches.c4_invariant_lagrangian.search import (
    NUM_QUBITS,
    TARGET_STABILIZER_RANK,
    analyze_candidate,
    construct_candidate,
    decompose_free_orbits,
    gf2_rank,
    logical_columns,
    minimum_weight_basis,
    module_partitions,
    nilpotent_translation_power,
    translate_vector,
)


def test_canonical_logicals_are_disjoint_orthonormal_c4_orbit() -> None:
    logicals = logical_columns()
    assert logicals.shape == (4, 28)
    assert np.all(np.count_nonzero(logicals, axis=1) == 7)
    assert np.array_equal((logicals @ logicals.T) % 2, np.eye(4, dtype=np.uint8))
    assert np.array_equal(translate_vector(logicals[0]), logicals[1])
    assert np.max(np.sum(logicals, axis=0)) == 1


def test_random_constructor_produces_valid_lagrangian_code() -> None:
    stabilizer = construct_candidate(
        rng=random.Random(11), maximum_check_weight=12, attempts_per_orbit=2000
    )
    assert stabilizer is not None
    assert stabilizer.shape[1] == NUM_QUBITS
    assert gf2_rank(stabilizer) == TARGET_STABILIZER_RANK
    assert not np.any((stabilizer @ stabilizer.T) % 2)
    assert not np.any((stabilizer @ logical_columns().T) % 2)
    analysis = analyze_candidate(stabilizer)
    assert analysis["n"] == 28
    assert analysis["k"] == 4
    assert analysis["checks"]["c4_invariant_rowspace"]
    assert analysis["maximum_check_weight"] <= 12
    assert 1 <= analysis["certified_distance"] <= 7
    assert len(decompose_free_orbits(stabilizer)) == 3
    low_basis = minimum_weight_basis(stabilizer)
    assert gf2_rank(low_basis) == TARGET_STABILIZER_RANK
    assert np.count_nonzero(low_basis, axis=1).max() <= 12


def test_nilpotent_translation_and_module_partitions() -> None:
    nilpotent = nilpotent_translation_power(1)
    assert not np.any(nilpotent_translation_power(4))
    assert not np.any(
        (nilpotent @ nilpotent_translation_power(3)) % 2
    )
    partitions = module_partitions()
    assert (4, 4, 4) in partitions
    assert (4, 4, 2, 2) in partitions
    assert (1,) * 12 in partitions
    assert all(sum(module_type) == 12 for module_type in partitions)
