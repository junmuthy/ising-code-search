"""Regression tests for the certified weight-12 four-grid S3 code."""

from __future__ import annotations

from gala_search.s3_many_copy_result import (
    analyze_certified_s3_four_grid_code,
    certified_s3_four_grid_candidate,
    certify_s3_four_grid_distance,
)


def test_certified_s3_four_grid_structure() -> None:
    candidate = certified_s3_four_grid_candidate()
    assert candidate.candidate_id == (
        "s3-l12-j3-w12-vertex-fold-5390517e81fd1dd8"
    )

    result = analyze_certified_s3_four_grid_code()
    structure = result["structure"]
    assert result["parameters"] == {"n": 1152, "k": 580, "d": 6}
    assert structure["accepted"]
    assert structure["checks"]["translation_ZX_fold"]
    assert structure["checks"]["uniform_check_weight_12"]
    assert structure["checks"]["uniform_column_degree_3"]
    assert structure["checks"]["doubly_even_stabilizer_generators"]
    assert structure["orthogonality"]["active_offsets_zero"]
    assert structure["orthogonality"]["latent_offset_3_nonzero"]
    assert not structure["even_syndrome_parity"]

    assert result["four_grids_pairwise_disjoint"]
    assert result["logical_support_union_weight"] == 4 * 32 * 6
    assert result["combined_rank_mod_stabilizers"] == 128
    assert result["combined_zx_pairing_rank"] == 128
    assert result["pairing_is_permutation"]
    assert result["logical_grid_pair_maps"] == [
        [[2, 7, 0]],
        [[3, 7, 1]],
        [[0, 1, 0]],
        [[1, 1, 3]],
    ]
    assert all(
        sector["pairwise_disjoint"]
        and sector["translation_orbit_size"] == 32
        and sector["orbit_rank_mod_stabilizers"] == 32
        for sector in result["sector_results"]
    )
    assert result["schedule"] == {
        "minimum_cnot_layers_complete_css_round": 12,
        "check_ancilla_degree": 12,
        "combined_xz_data_degree": 6,
        "optimal_by_bipartite_edge_coloring": True,
    }


def test_certified_s3_four_grid_exact_distance() -> None:
    result = certify_s3_four_grid_distance(max_nodes=5_000_000)
    assert result["certified"]
    assert result["certified_distance"] == 6
    assert result["logical_up_to_weight_four"] is None
    assert result["weight_five"]["search_exhaustive"]
    assert result["weight_five"]["support"] is None
    assert result["weight_six"]["search_exhaustive"]
    assert result["weight_six"]["is_nontrivial_logical"]
    assert len(result["weight_six"]["support"]) == 6
