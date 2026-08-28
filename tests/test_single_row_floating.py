"""Tests for the floating-dimension single-row search."""

from __future__ import annotations

import numpy as np

from gala_search.single_row_floating import (
    graph_seed_supports,
    identity_fold,
    involutive_gl_folds,
    select_self_dual_seed_witnesses,
)
from gala_search.single_row import analyze_seed_fold


def test_all_structured_involutive_folds_have_order_two() -> None:
    folds = involutive_gl_folds()
    assert len(folds) == 24
    for fold in folds:
        permutation = fold.permutation
        assert np.array_equal(permutation[permutation], np.arange(32))


def test_weight_six_graph_seeds_are_complete_and_disjoint() -> None:
    supports = graph_seed_supports()
    assert len(supports) == 28 * 4**5
    assert all(len(support) == 6 for support in supports[:100])


def test_structured_involutions_are_fixed_point_free_on_c4_fibres() -> None:
    for fold in involutive_gl_folds():
        assert all(
            source != target
            for source, target in enumerate(fold.fibre_images)
        )


def test_identity_fold_has_odd_permutation_pairing_witnesses() -> None:
    witnesses = select_self_dual_seed_witnesses(count=4)
    assert len(witnesses) == 4
    assert all(witness["zx_pairing_rank"] == 4 for witness in witnesses)
    assert all(witness["pairing_is_permutation"] for witness in witnesses)
    fold = identity_fold(8)
    assert analyze_seed_fold(graph_seed_supports(6)[0], fold)[
        "zx_pairing_rank"
    ] == 0
