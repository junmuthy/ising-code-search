"""Regression tests for the paired-polynomial GALA search."""

from __future__ import annotations

import numpy as np

from gala_search.core import (
    WeightEightCandidate,
    analyze_candidate,
    build_code,
    candidate_logicals,
    certify_distance,
    iter_weight_eight_candidates,
)

BASELINE = WeightEightCandidate(m=7, r=3, s=2, t=2)


def test_weight_eight_iteration_has_no_duplicate_supports() -> None:
    candidates = list(iter_weight_eight_candidates(7, quotient_symmetries=False))
    supports = [candidate.support for candidate in candidates]
    assert len(candidates) == 126
    assert len(supports) == len(set(supports))
    assert BASELINE in candidates


def test_symmetry_quotient_retains_baseline_class() -> None:
    candidates = list(iter_weight_eight_candidates(7))
    keys = [candidate.equivalence_key for candidate in candidates]
    assert len(keys) == len(set(keys))
    assert BASELINE.equivalence_key in set(keys)


def test_known_bicycle_chain_baseline() -> None:
    code = build_code(BASELINE)
    logicals = code.field(candidate_logicals(BASELINE))
    assert code.num_qubits == 56
    assert code.dimension == 8
    assert code.code_x.rank == 24
    assert code.get_weight() == 8
    assert np.array_equal(code.matrix_x, code.matrix_z)
    assert not np.any(code.matrix_x @ logicals.T)
    assert np.array_equal(logicals @ logicals.T, np.eye(8, dtype=int))

    result = analyze_candidate(BASELINE)
    assert result["accepted"]
    assert result["distance_upper_bound"] == 7
    assert result["tanner_girth"] >= 4


def test_baseline_exact_distance() -> None:
    result = certify_distance(BASELINE)
    assert result["complete"]
    assert result["certified_distance"] == 6
    assert result["ilp_distance_upper_bound"] == 6
    assert all(item["solver_status"] == "optimal" for item in result["class_results"])


def test_threshold_certification_is_not_mislabeled_exact() -> None:
    result = certify_distance(BASELINE, stop_at=6)
    assert result["threshold_met"]
    assert result["ilp_distance_upper_bound"] == 6
    assert not result["complete"]
    assert result["certified_distance"] is None
    assert len(result["class_results"]) == 1
