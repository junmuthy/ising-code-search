#!/usr/bin/env python3
"""Independent verification bundle for the first relaxed C4 x C7 hit."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from qldpc import codes

from distance import (
    certify_distance_at_least,
    find_logical_milp,
)
from common import (
    atomic_json,
    compose,
    find_regular_permutation_h_grid_css,
    maps_rowspace,
    permutation_order,
    preserves_rowspace,
    rref,
)
from construction import Family


SUPPORT_A = (0, 1, 7, 9)
SIGMA_Y = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matrix-output", required=True, type=Path)
    return parser.parse_args()


def connected(hx: list[int], hz: list[int], width: int) -> bool:
    checks = [*hx, *hz]
    adjacency = [[] for _ in range(width + len(checks))]
    for row_index, row in enumerate(checks):
        check_vertex = width + row_index
        value = row
        while value:
            low = value & -value
            qubit = low.bit_length() - 1
            adjacency[qubit].append(check_vertex)
            adjacency[check_vertex].append(qubit)
            value ^= low
    seen = {0}
    queue = deque([0])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(adjacency)


def matrix(rows: list[int], width: int) -> np.ndarray:
    return np.asarray(
        [[(row >> column) & 1 for column in range(width)] for row in rows],
        dtype=np.uint8,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.matrix_output.exists():
        raise SystemExit("refusing to overwrite verification output")
    family = Family(7)
    support_b = family.transformed_partner(SUPPORT_A, SIGMA_Y)
    hx_raw, hz_raw = family.build_checks(SUPPORT_A, support_b)
    width = family.num_qubits
    hx, _ = rref(hx_raw, width)
    hz, _ = rref(hz_raw, width)
    t4, t2, fold = family.physical_maps(SIGMA_Y)
    diagnostics: dict[str, object] = {}
    grid = find_regular_permutation_h_grid_css(
        hx, hz, width, t4, t2, fold, diagnostics
    )
    if grid is None:
        raise AssertionError(diagnostics)
    hx_matrix = matrix(hx_raw, width)
    hz_matrix = matrix(hz_raw, width)
    code = codes.CSSCode(hx_matrix, hz_matrix)
    lower = certify_distance_at_least(
        code,
        6,
        solver="HIGHS",
        equal_xz_by_permutation=True,
    )
    exact_z = find_logical_milp(code, pauli="Z", solver="HIGHS")
    exact_x = find_logical_milp(code, pauli="X", solver="HIGHS")
    logical_h = grid["hadamard_permutation"]
    logical_h_inverse = [logical_h.index(site) for site in range(8)]
    logical_t4 = [2 * ((site // 2 + 1) % 4) + site % 2 for site in range(8)]
    logical_t4_inverse = [logical_t4.index(site) for site in range(8)]
    logical_t2 = [site ^ 1 for site in range(8)]
    css_commutes = not any((left & right).bit_count() & 1 for left in hx for right in hz)
    relations = {
        "css_commutes": css_commutes,
        "rank_x": len(hx),
        "rank_z": len(hz),
        "k": width - len(hx) - len(hz),
        "tanner_connected": connected(hx_raw, hz_raw, width),
        "uniform_x_check_weight": sorted({row.bit_count() for row in hx_raw}),
        "uniform_z_check_weight": sorted({row.bit_count() for row in hz_raw}),
        "translation_four_physical_order": permutation_order(t4),
        "translation_two_physical_order": permutation_order(t2),
        "translations_commute": compose(t4, t2) == compose(t2, t4),
        "translation_four_preserves_x": preserves_rowspace(hx, t4, width),
        "translation_four_preserves_z": preserves_rowspace(hz, t4, width),
        "translation_two_preserves_x": preserves_rowspace(hx, t2, width),
        "translation_two_preserves_z": preserves_rowspace(hz, t2, width),
        "fold_maps_x_to_z": maps_rowspace(hx, hz, fold, width),
        "fold_maps_z_to_x": maps_rowspace(hz, hx, fold, width),
        "fold_physical_order": permutation_order(fold),
        "logical_orbit_rank": 8,
        "logical_hadamard_is_permutation": True,
        "logical_hadamard_normalizes_t4_by_inversion": (
            compose(logical_h, compose(logical_t4, logical_h_inverse))
            == logical_t4_inverse
        ),
        "logical_hadamard_commutes_with_t2": (
            compose(logical_h, compose(logical_t2, logical_h_inverse))
            == logical_t2
        ),
    }
    if not (
        all(
            value
            for key, value in relations.items()
            if key
            not in {
                "rank_x",
                "rank_z",
                "k",
                "uniform_x_check_weight",
                "uniform_z_check_weight",
                "translation_four_physical_order",
                "translation_two_physical_order",
                "fold_physical_order",
                "logical_orbit_rank",
            }
        )
        and relations["rank_x"] == relations["rank_z"] == 24
        and relations["k"] == 8
        and relations["uniform_x_check_weight"] == [8]
        and relations["uniform_z_check_weight"] == [8]
        and lower["certified"]
        and exact_z["weight"] == exact_x["weight"] == 6
    ):
        raise AssertionError((relations, lower, exact_z, exact_x))

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.matrix_output,
        matrix_x=hx_matrix,
        matrix_z=hz_matrix,
        translation_four=np.asarray(t4, dtype=int),
        translation_two=np.asarray(t2, dtype=int),
        hadamard_fold=np.asarray(fold, dtype=int),
        logical_z_orbit=np.asarray(
            [
                [(column in support) for column in range(width)]
                for support in grid["orbit_supports"]
            ],
            dtype=np.uint8,
        ),
        logical_hadamard_permutation=np.asarray(grid["hadamard_permutation"], dtype=int),
    )
    payload = {
        "complete": True,
        "code": "[[56,8,6]]",
        "family": "C4xC7-automorphism-dual-BB",
        "group_coordinate_order": [4, 7],
        "qubit_index": "half*28 + 7*x + y",
        "a_indices": list(SUPPORT_A),
        "a_support": [list(family.elements[item]) for item in SUPPORT_A],
        "b_indices": list(support_b),
        "b_support": [list(family.elements[item]) for item in support_b],
        "polynomials": {
            "a": "1 + y + x + x*y^2",
            "b": "1 + y^-1 + x + x*y^-2",
        },
        "check_weight": 8,
        "relations": relations,
        "grid": grid,
        "logical_site_order": "site=2*x+y for x in C4 and y in C2",
        "logical_hadamard_affine_map": "(x,y) -> (-x,y+1)",
        "distance_lower_certificate": lower,
        "exact_z_distance": exact_z,
        "exact_x_distance": exact_x,
        "matrix_artifact": str(args.matrix_output),
        "verification_note": (
            "Distance lower bound was independently certified by the repository CVXPY/HIGHS "
            "logical MILP; unconstrained minimizations supplied weight-six X and Z witnesses."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
