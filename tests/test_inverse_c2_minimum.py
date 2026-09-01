from __future__ import annotations

import pathlib
import random
import sys

import numpy as np

from inverse_c2_minimum import search
from inverse_c2_minimum import run_search
from inverse_c2_minimum import local_refinement


def test_n12_pairing_obstruction_for_random_permutations() -> None:
    logicals = (
        sum(1 << (2 * index) for index in range(6)),
        sum(1 << (2 * index + 1) for index in range(6)),
    )
    generator = random.Random(12)
    for _ in range(100):
        permutation = list(range(12))
        generator.shuffle(permutation)
        pairing = search.pairing_matrix(logicals, permutation)
        assert np.array_equal(
            (pairing @ np.ones(2, dtype=np.uint8)) % 2,
            np.zeros(2, dtype=np.uint8),
        )
        assert search.gf2_rank(pairing) < 2


def test_canonical_fold_catalogs_are_valid_and_nonempty_when_expected() -> None:
    expected = {
        "n14-w6-s2-0-s1-2": 0,
        "n14-w6-s2-1-s1-0": 20,
        "n14-w7-s2-0-s1-0": 40,
    }
    for geometry in search.geometry_catalog(14):
        folds = search.canonical_fold_catalog(geometry)
        assert len(folds) == expected[geometry.name]
        for fold in folds:
            analysis = fold.analyze()
            assert analysis["involution"]
            assert analysis["commutes_with_translation"]
            assert analysis["pairing_is_permutation"]


def test_structural_solver_returns_a_valid_c2_folded_css_model() -> None:
    geometry = search.geometry_catalog(14)[1]
    fold = search.canonical_fold_catalog(geometry)[0]
    solver = search.FoldedCSSSolver(
        fold=fold,
        module_type=(2, 2, 2),
        maximum_check_weight=8,
        timeout_ms=10_000,
    )
    assert str(solver.check()) == "sat"
    analysis = search.analyze_basis(solver.concrete_basis(), fold)
    assert analysis["n"] == 14
    assert analysis["k"] == 2
    assert analysis["rank_x"] == 6
    assert analysis["rank_z"] == 6
    assert analysis["css_orthogonal"]
    assert analysis["logical_z_disjoint"]
    assert analysis["logical_x_disjoint"]
    assert analysis["logical_pairing_is_permutation"]
    assert analysis["translation_swaps_logical_z"]
    assert analysis["translation_swaps_logical_x"]


def test_eager_distance_constraints_match_concrete_distance() -> None:
    geometry = search.geometry_catalog(14)[1]
    fold = search.canonical_fold_catalog(geometry)[0]
    solver = search.FoldedCSSSolver(
        fold=fold,
        module_type=(2, 2, 1, 1),
        maximum_check_weight=8,
        timeout_ms=10_000,
    )
    assert solver.add_full_distance_constraints() > 0
    status = solver.check()
    if str(status) == "sat":
        assert search.analyze_basis(solver.concrete_basis(), fold)["distance"] >= 6
    else:
        assert str(status) in {"unsat", "unknown"}


def test_runner_accepts_exclusive_fold_ranges(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_search.py",
            "--output-root",
            "results",
            "--run-name",
            "range-smoke",
            "--fold-start",
            "5",
            "--fold-stop",
            "10",
        ],
    )
    arguments = run_search.parse_args()
    assert arguments.fold_start == 5
    assert arguments.fold_stop == 10


def test_n18_closest_analogue_geometry() -> None:
    geometry = search.LogicalGeometry(
        n=18,
        logical_weight=7,
        slack_transpositions=2,
        slack_fixed_points=0,
    )
    assert geometry.name == "n18-w7-s2-2-s1-0"
    assert geometry.analyze()["translation_swaps_logical_z"]
    assert not (geometry.logical_z[0] & geometry.logical_z[1])


def test_n20_odd_support_geometry() -> None:
    geometry = search.LogicalGeometry(
        n=20,
        logical_weight=9,
        slack_transpositions=1,
        slack_fixed_points=0,
    )
    assert geometry.name == "n20-w9-s2-1-s1-0"
    assert geometry.analyze()["translation_swaps_logical_z"]
    assert not (geometry.logical_z[0] & geometry.logical_z[1])
    assert geometry.logical_z[0].bit_count() == 9
    assert geometry.logical_z[1].bit_count() == 9


def test_local_refinement_reconstructs_saved_distance_four_seed() -> None:
    path = next(
        pathlib.Path("results/inverse-c2-minimum").glob(
            "n16-w6-s2-2-s1-0-folds00-04-w8-260830-v1/tasks/"
            "task-0000-*-best.json"
        )
    )
    seed = local_refinement.load_seed(path)
    analysis = local_refinement.analyze_generators(
        seed["generators"], seed["module_type"], seed["fold"]
    )
    assert analysis["n"] == 16
    assert analysis["k"] == 2
    assert analysis["distance"] == 4
    assert analysis["css_orthogonal"]
    assert analysis["logical_pairing_is_permutation"]
    assert analysis["translation_swaps_logical_z"]
    assert analysis["translation_swaps_logical_x"]
    assert analysis["presentation_maximum_check_weight"] <= 8


def test_local_refinement_discovers_exact_analogue_distance_three_seeds() -> None:
    seeds = local_refinement.candidate_generators(
        16,
        tuple([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]),
        (
            sum(1 << index for index in range(0, 14, 2)),
            sum(1 << index for index in range(1, 14, 2)),
        ),
        2,
        8,
    )
    assert seeds
    assert all(mask.bit_count() <= 8 for mask in seeds)
