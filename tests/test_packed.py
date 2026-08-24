"""Regression tests for the packed ``C_8 x C_4`` logical quotient."""

from __future__ import annotations

import numpy as np

from gala_search.packed import (
    LOGICALS_PER_HALF,
    NUM_LOGICALS,
    PackedCandidate,
    PackedWeightTwelveCandidate,
    analyze_packed_candidate,
    build_packed_code,
    certify_packed_distance,
    iter_packed_candidates,
    packed_automorphism_checks,
    packed_logicals,
)

DISCONNECTED_WEIGHT8 = PackedCandidate(m=7, r=1, s=1, u=1, v=1)
CONNECTED_WEIGHT12 = PackedWeightTwelveCandidate(
    m=7, r0=1, r1=1, r2=1, p1=0, p2=0
)


def test_packed_iteration_has_unique_equivalence_keys() -> None:
    candidates = list(iter_packed_candidates(7))
    keys = [candidate.equivalence_key for candidate in candidates]
    assert candidates
    assert len(keys) == len(set(keys))
    assert DISCONNECTED_WEIGHT8.equivalence_key in set(keys)


def test_packed_logicals_partition_the_physical_qubits() -> None:
    code = build_packed_code(CONNECTED_WEIGHT12)
    logicals = packed_logicals(CONNECTED_WEIGHT12)
    assert logicals.shape == (NUM_LOGICALS, code.num_qubits)
    assert np.all(np.count_nonzero(logicals, axis=1) == CONNECTED_WEIGHT12.m)
    assert np.all(np.count_nonzero(logicals, axis=0) == 1)


def test_connected_packed_candidate_has_target_logical_module() -> None:
    result = analyze_packed_candidate(CONNECTED_WEIGHT12)
    assert result["accepted"]
    assert result["n"] == 448
    assert result["k"] == 64
    assert result["rank_h"] == 192
    assert result["check_weight"] == 12
    assert result["checks"]["self_dual"]
    assert result["checks"]["candidate_logicals_complete"]
    assert result["checks"]["candidate_logicals_partition_qubits"]
    assert result["checks"]["x_translation_is_C8"]
    assert result["checks"]["y_translation_is_C4"]


def test_weight_eight_two_pair_fixture_is_disconnected() -> None:
    result = analyze_packed_candidate(DISCONNECTED_WEIGHT8)
    assert not result["accepted"]
    assert result["rejection_reasons"] == ["tanner_connected"]


def test_physical_translations_preserve_the_stabilizers() -> None:
    assert all(packed_automorphism_checks(CONNECTED_WEIGHT12).values())


def test_two_halves_are_the_distance_representatives() -> None:
    result = certify_packed_distance(
        CONNECTED_WEIGHT12,
        target_distance=6,
        logical_indices=[0, LOGICALS_PER_HALF],
    )
    assert result["complete"]
    assert result["certified_by_translation_symmetry"]
    assert result["threshold_passed"]
    assert result["certified_distance"] == 6
