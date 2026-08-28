"""Certified four-grid result from the weight-12 nonabelian search."""

from __future__ import annotations

from typing import Any

import numpy as np

from .s3_ising import (
    ProductMonomial,
    _gf2_rank,
    _permute_columns,
    analyze_translation_logical_seed,
    find_logical_up_to_weight_four,
    find_zero_syndrome_by_tanner_search,
    logical_translation_orbit,
)
from .s3_many_copy import (
    S3L12VertexFoldW12Candidate,
    analyze_s3_l12_fold_candidate,
    build_s3_l12_fold_code,
    s3_l12_zx_fold_permutation,
)


def certified_s3_four_grid_candidate() -> S3L12VertexFoldW12Candidate:
    """Return the saved ``[[1152,580,6]]`` four-grid candidate."""
    entries = (
        (ProductMonomial(0, 0, 0),),
        (ProductMonomial(4, 7, 3),),
        (ProductMonomial(1, 7, 0),),
        (ProductMonomial(0, 4, 3),),
        (ProductMonomial(0, 7, 1),),
        (ProductMonomial(4, 7, 0),),
    )
    return S3L12VertexFoldW12Candidate(entries)


def certified_s3_four_grid_supports() -> tuple[tuple[int, ...], ...]:
    """One weight-six seed for each of the four disjoint logical grids."""
    return (
        (32, 452, 485, 764, 829, 915),
        (64, 396, 560, 703, 776, 888),
        (96, 215, 295, 639, 979, 1103),
        (128, 259, 365, 668, 1016, 1083),
    )


def analyze_certified_s3_four_grid_code() -> dict[str, Any]:
    """Recompute the complete structural and four-grid certificate."""
    candidate = certified_s3_four_grid_candidate()
    code = build_s3_l12_fold_code(candidate)
    supports = certified_s3_four_grid_supports()
    fold = s3_l12_zx_fold_permutation(candidate)
    sector_results = [
        analyze_translation_logical_seed(
            code, support, zx_fold_permutation=fold
        )
        for support in supports
    ]
    z_orbits: list[np.ndarray] = []
    for support in supports:
        seed = np.zeros(code.num_qubits, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbits.append(logical_translation_orbit(seed, num_blocks=12))
    z_logicals = np.vstack(z_orbits)
    x_logicals = _permute_columns(z_logicals, fold)
    pairing = (z_logicals @ x_logicals.T) % 2
    combined_rank = (
        _gf2_rank(
            np.vstack([np.asarray(code.matrix_z, dtype=np.uint8), z_logicals]),
            code.field,
        )
        - code.code_z.rank
    )
    pair_maps: list[list[list[int]]] = []
    for grid in range(4):
        mappings: set[tuple[int, int, int]] = set()
        for logical_x in range(8):
            for logical_y in range(4):
                row = grid * 32 + logical_x * 4 + logical_y
                columns = np.flatnonzero(pairing[row])
                if len(columns) != 1:
                    continue
                partner, remainder = divmod(int(columns[0]), 32)
                partner_x, partner_y = divmod(remainder, 4)
                mappings.add(
                    (
                        partner,
                        (partner_x - logical_x) % 8,
                        (partner_y - logical_y) % 4,
                    )
                )
        pair_maps.append([list(mapping) for mapping in sorted(mappings)])
    integer_x = np.asarray(code.matrix_x, dtype=np.uint8)
    integer_z = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weight = int(np.count_nonzero(integer_x, axis=1).max())
    combined_data_degree = int(
        (
            np.count_nonzero(integer_x, axis=0)
            + np.count_nonzero(integer_z, axis=0)
        ).max()
    )
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "polynomials": candidate.polynomial_text,
        "parameters": {"n": code.num_qubits, "k": code.dimension, "d": 6},
        "structure": analyze_s3_l12_fold_candidate(candidate),
        "logical_supports": [list(support) for support in supports],
        "sector_results": sector_results,
        "four_grids_pairwise_disjoint": bool(
            np.all(np.sum(z_logicals, axis=0) <= 1)
        ),
        "logical_support_union_weight": int(
            np.count_nonzero(np.any(z_logicals, axis=0))
        ),
        "combined_rank_mod_stabilizers": combined_rank,
        "combined_zx_pairing_rank": _gf2_rank(pairing, code.field),
        "pairing_is_permutation": bool(
            np.all(np.count_nonzero(pairing, axis=0) == 1)
            and np.all(np.count_nonzero(pairing, axis=1) == 1)
        ),
        "logical_grid_pair_maps": pair_maps,
        "schedule": {
            "minimum_cnot_layers_complete_css_round": max(
                row_weight, combined_data_degree
            ),
            "check_ancilla_degree": row_weight,
            "combined_xz_data_degree": combined_data_degree,
            "optimal_by_bipartite_edge_coloring": True,
        },
    }


def certify_s3_four_grid_distance(*, max_nodes: int = 5_000_000) -> dict[str, Any]:
    """Certify no logical below six and return a weight-six witness."""
    candidate = certified_s3_four_grid_candidate()
    code = build_s3_l12_fold_code(candidate)
    lower = find_logical_up_to_weight_four(code)
    weight_five = find_zero_syndrome_by_tanner_search(
        code, weight=5, max_nodes=max_nodes
    )
    weight_six = find_zero_syndrome_by_tanner_search(
        code, weight=6, max_nodes=max_nodes
    )
    certified = bool(
        lower is None
        and weight_five is not None
        and weight_five.get("search_exhaustive")
        and weight_five.get("support") is None
        and weight_six is not None
        and weight_six.get("search_exhaustive")
        and weight_six.get("is_nontrivial_logical")
    )
    return {
        "certified": certified,
        "certified_distance": 6 if certified else None,
        "logical_up_to_weight_four": lower,
        "weight_five": weight_five,
        "weight_six": weight_six,
        "x_distance_equals_z_distance_by_zx_fold": True,
    }
