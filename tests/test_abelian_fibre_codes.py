"""Regression tests for the two order-seven-fibre abelian targets."""

from __future__ import annotations

import numpy as np

from gala_search.abelian_fibre_codes import (
    analyze_candidate,
    build_code,
    certify_distance,
    fibre_logicals,
    iter_four_term_candidates,
    iter_six_term_candidates,
    make_four_term_candidate,
    make_six_term_candidate,
)


CERTIFIED_QUARTER_GRID = make_six_term_candidate(4, 3, 1, 3, 6, 2)
CERTIFIED_ROW = make_six_term_candidate(1, 1, 1, 2, 0, 2, 1, 0)


def test_known_bicycle_chain_seed_has_required_structure() -> None:
    candidate = make_four_term_candidate(3, 2, 2)
    analysis = analyze_candidate(candidate)
    assert analysis["accepted"]
    assert (analysis["n"], analysis["k"], analysis["check_weight"]) == (56, 8, 8)
    assert np.array_equal(build_code(candidate).matrix_x, build_code(candidate).matrix_z)
    assert np.all(np.count_nonzero(fibre_logicals(candidate), axis=1) == 7)


def test_scaled_packed_seed_has_disjoint_c4xc4_fibres() -> None:
    analysis = analyze_candidate(CERTIFIED_QUARTER_GRID)
    assert analysis["accepted"]
    assert analysis["n"] == 224
    assert analysis["k"] == 32
    assert analysis["check_weight"] == 12
    assert analysis["checks"]["self_dual"]
    assert analysis["checks"]["fibre_logicals_orthonormal"]
    assert analysis["checks"]["fibre_logicals_partition_qubits"]
    assert analysis["checks"]["x_translation_is_C4"]
    assert analysis["checks"]["y_translation_is_C4"]


def test_certified_row_candidate_has_target_logical_module() -> None:
    analysis = analyze_candidate(CERTIFIED_ROW)
    assert analysis["accepted"]
    assert (analysis["n"], analysis["k"], analysis["check_weight"]) == (56, 8, 12)
    assert analysis["checks"]["self_dual"]
    assert analysis["checks"]["fibre_logicals_partition_qubits"]
    assert analysis["checks"]["y_translation_is_C4"]


def test_search_spaces_have_unique_canonical_supports() -> None:
    families = (
        list(iter_four_term_candidates()),
        list(iter_six_term_candidates(1)),
        list(iter_six_term_candidates(4)),
    )
    assert [len(family) for family in families] == [24, 426, 1323]
    for family in families:
        keys = [candidate.equivalence_key for candidate in family]
        assert len(keys) == len(set(keys))


def test_selected_candidates_have_exact_distance_seven() -> None:
    for candidate in (CERTIFIED_ROW, CERTIFIED_QUARTER_GRID):
        certificate = certify_distance(candidate)
        assert certificate["complete"]
        assert certificate["certified_by_translation_symmetry"]
        assert certificate["threshold_passed"]
        assert certificate["certified_distance"] == 7
