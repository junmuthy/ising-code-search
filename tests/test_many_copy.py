"""Regression tests for the scalable many-copy packed-code family."""

from __future__ import annotations

import numpy as np

from gala_search.many_copy import (
    analyze_many_copy_candidate,
    build_many_copy_code,
    certify_many_copy_distance,
    iter_many_copy_candidates,
    many_copy_automorphism_checks,
    many_copy_cnot_schedule,
    many_copy_logicals,
    natural_four_copy_candidate,
)


def test_many_copy_iteration_has_unique_equivalence_keys() -> None:
    candidates = list(iter_many_copy_candidates(m=7, copies_per_half=2))
    keys = [candidate.equivalence_key for candidate in candidates]
    assert candidates
    assert len(keys) == len(set(keys))
    assert natural_four_copy_candidate().equivalence_key in set(keys)


def test_natural_four_copy_candidate_has_complete_logical_module() -> None:
    candidate = natural_four_copy_candidate()
    code = build_many_copy_code(candidate)
    logicals = many_copy_logicals(candidate)
    analysis = analyze_many_copy_candidate(candidate)

    assert logicals.shape == (128, 896)
    assert np.all(np.count_nonzero(logicals, axis=1) == 7)
    assert np.all(np.count_nonzero(logicals, axis=0) == 1)
    assert analysis["accepted"]
    assert (analysis["n"], analysis["k"]) == (896, 128)
    assert analysis["rank_h"] == 384
    assert analysis["check_weight"] == 12
    assert analysis["qubit_degree"] == 6
    assert analysis["num_simulation_copies"] == 4
    assert analysis["minimum_cnot_layers_per_css_type"] == 12
    assert analysis["checks"]["candidate_logicals_orthonormal"]
    assert analysis["checks"]["candidate_logicals_partition_qubits"]
    assert analysis["checks"]["x_power_r_is_C8_on_each_copy"]
    assert analysis["checks"]["y_is_C4_on_each_copy"]
    assert np.array_equal(code.matrix_x, code.matrix_z)


def test_many_copy_physical_automorphisms() -> None:
    assert all(many_copy_automorphism_checks(natural_four_copy_candidate()).values())


def test_many_copy_has_optimal_twelve_layer_schedule() -> None:
    schedule = many_copy_cnot_schedule(natural_four_copy_candidate())
    assert schedule["optimal"]
    assert schedule["covers_each_check_edge_once"]
    assert schedule["all_layers_collision_free"]
    assert schedule["num_layers_per_css_type"] == 12
    assert schedule["lower_bound_from_check_weight"] == 12
    assert all(layer["num_interactions"] == 448 for layer in schedule["layers"])


def test_natural_four_copy_candidate_has_distance_seven() -> None:
    result = certify_many_copy_distance(
        natural_four_copy_candidate(), target_distance=7
    )
    assert result["complete"]
    assert result["certified_by_translation_symmetry"]
    assert result["threshold_passed"]
    assert result["certified_distance"] == 7
