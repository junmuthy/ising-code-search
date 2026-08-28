"""Tests for the corrected multi-grid ZX-pairing criterion."""

from __future__ import annotations

from gala_search.s3_fold_analysis import build_certified_compact_code
from gala_search.s3_joint_pairing import (
    analyze_joint_grid_combinations,
    build_candidate_code,
    enumerate_translation_grid_orbits,
    saturation_even_weight_obstruction,
    saved_extended_near_miss_candidate,
)


def test_even_weight_saturation_obstruction() -> None:
    assert saturation_even_weight_obstruction(
        num_qubits=384, num_grids=2, logical_weight=6
    )
    assert saturation_even_weight_obstruction(
        num_qubits=768, num_grids=4, logical_weight=6
    )
    assert not saturation_even_weight_obstruction(
        num_qubits=768, num_grids=3, logical_weight=6
    )


def test_compact_candidate_has_only_its_rank_zero_grid() -> None:
    code = build_certified_compact_code()
    enumeration = enumerate_translation_grid_orbits(code, weight=6)
    assert enumeration["enumeration_exhaustive"]
    assert enumeration["num_orbits"] == 1
    assert enumeration["self_pairing_rank_counts"] == {0: 1}
    result = analyze_joint_grid_combinations(
        code, enumeration["grids"], maximum_grids=2
    )
    assert result["by_num_grids"]["1"]["best_logical_rank"] == 32
    assert result["by_num_grids"]["1"]["best_pairing_rank_any"] == 0
    assert not result["full_rank_hits"]


def test_extended_near_miss_is_exhaustively_rank_60_and_56() -> None:
    candidate = saved_extended_near_miss_candidate()
    assert candidate.candidate_id == "s3-l8-j2-w16-b77391bd2f2fd82b"
    code = build_candidate_code(candidate)
    assert (code.num_qubits, code.dimension) == (768, 388)
    enumeration = enumerate_translation_grid_orbits(code, weight=6)
    assert enumeration["enumeration_exhaustive"]
    assert enumeration["num_orbits"] == 2
    result = analyze_joint_grid_combinations(
        code, enumeration["grids"], maximum_grids=3
    )
    pair = result["by_num_grids"]["2"]
    assert pair["disjoint_combinations"] == 1
    assert pair["best_logical_rank"] == 60
    assert pair["best_pairing_rank_any"] == 56
    assert pair["independent_combinations"] == 0
    assert not result["full_rank_hits"]
