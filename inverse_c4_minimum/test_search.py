from __future__ import annotations

import random

import numpy as np

import search


def test_n24_pairing_augmentation_no_go_for_random_permutations() -> None:
    rng = random.Random(24)
    logicals = []
    for logical in range(4):
        mask = sum(1 << (6 * logical + offset) for offset in range(6))
        logicals.append(mask)
    for _ in range(50):
        permutation = list(range(24))
        rng.shuffle(permutation)
        pairing = np.asarray(
            [
                [
                    (left & search.permute_mask(right, permutation)).bit_count() & 1
                    for right in logicals
                ]
                for left in logicals
            ],
            dtype=np.uint8,
        )
        assert np.all((pairing @ np.ones(4, dtype=np.uint8)) % 2 == 0)
        assert search.gf2_rank(pairing) < 4


def test_fold_catalog_has_clean_hadamard_geometry() -> None:
    for fold in search.canonical_fold_catalog():
        analysis = fold.analyze()
        assert analysis["involution"]
        assert analysis["normalizes_translation"]
        assert analysis["pairing_rank"] == 4
        assert analysis["pairing_is_permutation"]


def test_symbolic_free_module_returns_valid_folded_css_model() -> None:
    geometry = search.canonical_fold_catalog()[0]
    solver = search.FoldedCSSSolver(
        geometry=geometry,
        module_type=(4, 4, 4),
        timeout_ms=10_000,
    )
    assert str(solver.check()) == "sat"
    analysis = search.analyze_basis(solver.concrete_basis(), geometry)
    assert analysis["rank_x"] == 12
    assert analysis["rank_z"] == 12
    assert analysis["css_orthogonal"]
    assert analysis["fold"]["pairing_is_permutation"]
    assert analysis["fold"]["normalizes_translation"]
