"""Regression tests for the floating-``k`` self-dual ``C4 x C4`` search."""

from __future__ import annotations

import numpy as np

from c4xc4_floating.search import analyze_logical_orbit, find_graph_seed
from gala_search.abelian_bb32 import build_checks


def test_meet_in_middle_finds_seven_fibre_seed_in_zero_check() -> None:
    check = np.zeros((16, 32), dtype=np.uint8)
    support = find_graph_seed(check)
    assert support is not None
    assert len(support) == 7
    logical = analyze_logical_orbit(check, support)
    assert logical["pairwise_disjoint"]
    assert logical["orbit_in_kernel"]
    assert logical["zx_pairing_is_identity"]


def test_meet_in_middle_result_is_exactly_validated() -> None:
    check, _check_z = build_checks((0, 1, 2, 4, 8, 12))
    support = find_graph_seed(check)
    if support is not None:
        logical = analyze_logical_orbit(check, support)
        assert logical["orbit_in_kernel"]
        assert logical["pairwise_disjoint"]
        assert logical["zx_pairing_rank"] == 4
