#!/usr/bin/env python3
"""Independently validate saved inverse-C4 CSS candidates with qLDPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def matrix_from_supports(supports: list[list[int]], n: int) -> np.ndarray:
    matrix = np.zeros((len(supports), n), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def permutation_from_fold(fold: dict, n: int) -> np.ndarray:
    permutation = np.empty(n, dtype=int)
    epsilon = int(fold["epsilon"])
    for fibre, image_fibre in enumerate(fold["fibre_images"]):
        shift = int(fold["shifts"][fibre])
        for coordinate in range(4):
            source = 4 * fibre + coordinate
            target = 4 * int(image_fibre) + (epsilon * coordinate + shift) % 4
            permutation[source] = target
    return permutation


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    rank_left = gf2_rank(left)
    rank_right = gf2_rank(right)
    return rank_left == rank_right == gf2_rank(np.vstack([left, right]))


def supports_disjoint(matrix: np.ndarray) -> bool:
    overlaps = matrix @ matrix.T
    return bool(np.all(overlaps == np.diag(np.diag(overlaps))))


def validate(path: Path) -> dict:
    source = json.loads(path.read_text())
    analysis = source["analysis"]
    n = int(analysis["n"])
    hx = matrix_from_supports(analysis["basis_x"], n)
    hz = matrix_from_supports(analysis["basis_z"], n)
    logical_x = matrix_from_supports(analysis["logical_x"], n)
    logical_z = matrix_from_supports(analysis["logical_z"], n)
    permutation = permutation_from_fold(analysis["fold"], n)
    translation = np.array(
        [4 * (qubit // 4) + (qubit % 4 + 1) % 4 for qubit in range(n)],
        dtype=int,
    )
    inverse_translation = np.argsort(translation)
    expected_conjugate = translation if analysis["fold"]["epsilon"] == 1 else inverse_translation
    conjugate = permutation[translation[np.argsort(permutation)]]

    code = codes.CSSCode(hx, hz)
    distance_x = code.get_distance_exact(Pauli.X)
    distance_z = code.get_distance_exact(Pauli.Z)

    pairing = (logical_z @ logical_x.T) % 2
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    result = {
        "source": str(path.resolve()),
        "qldpc": {
            "n": int(code.num_qubits),
            "k": int(code.dimension),
            "distance_x": int(distance_x),
            "distance_z": int(distance_z),
            "distance": int(min(distance_x, distance_z)),
            "rank_x": int(code.code_x.rank),
            "rank_z": int(code.code_z.rank),
        },
        "checks": {
            "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
            "rank_x": rank_x,
            "rank_z": rank_z,
            "k": n - rank_x - rank_z,
            "matrices_equal": bool(np.array_equal(hx, hz)),
            "rowspaces_equal": same_rowspace(hx, hz),
            "fold_maps_hx_to_hz_row_for_row": bool(np.array_equal(act_rows(hx, permutation), hz)),
            "fold_maps_hx_rowspace_to_hz": same_rowspace(act_rows(hx, permutation), hz),
            "fold_maps_hz_rowspace_to_hx": same_rowspace(act_rows(hz, permutation), hx),
            "translation_preserves_hx_rowspace": same_rowspace(act_rows(hx, translation), hx),
            "translation_preserves_hz_rowspace": same_rowspace(act_rows(hz, translation), hz),
            "row_weights_x": np.count_nonzero(hx, axis=1).astype(int).tolist(),
            "row_weights_z": np.count_nonzero(hz, axis=1).astype(int).tolist(),
        },
        "fold": {
            "name": analysis["fold"]["name"],
            "permutation": permutation.astype(int).tolist(),
            "involution": bool(np.array_equal(permutation[permutation], np.arange(n))),
            "normalizes_translation": bool(np.array_equal(conjugate, expected_conjugate)),
            "epsilon": int(analysis["fold"]["epsilon"]),
            "logical_pairing": pairing.astype(int).tolist(),
            "logical_pairing_is_permutation": bool(
                np.all(pairing.sum(axis=0) == 1) and np.all(pairing.sum(axis=1) == 1)
            ),
        },
        "logicals": {
            "logical_z_weights": np.count_nonzero(logical_z, axis=1).astype(int).tolist(),
            "logical_x_weights": np.count_nonzero(logical_x, axis=1).astype(int).tolist(),
            "logical_z_supports_disjoint": supports_disjoint(logical_z),
            "logical_x_supports_disjoint": supports_disjoint(logical_x),
            "logical_z_commutes_with_hx": bool(not np.any((logical_z @ hx.T) % 2)),
            "logical_x_commutes_with_hz": bool(not np.any((logical_x @ hz.T) % 2)),
            "logical_z_independent_mod_stabilizers": gf2_rank(np.vstack([hz, logical_z])) - rank_z,
            "logical_x_independent_mod_stabilizers": gf2_rank(np.vstack([hx, logical_x])) - rank_x,
            "translation_cycles_logical_z": bool(
                np.array_equal(act_rows(logical_z, translation), np.roll(logical_z, 1, axis=0))
                or np.array_equal(act_rows(logical_z, translation), np.roll(logical_z, -1, axis=0))
            ),
            "fold_maps_logical_z_to_logical_x": bool(
                np.array_equal(act_rows(logical_z, permutation), logical_x)
            ),
        },
    }
    hard_checks = [
        result["qldpc"]["n"] == 28,
        result["qldpc"]["k"] == 4,
        result["checks"]["css_orthogonal"],
        result["checks"]["fold_maps_hx_to_hz_row_for_row"],
        result["checks"]["fold_maps_hz_rowspace_to_hx"],
        result["checks"]["translation_preserves_hx_rowspace"],
        result["checks"]["translation_preserves_hz_rowspace"],
        result["fold"]["involution"],
        result["fold"]["normalizes_translation"],
        result["fold"]["logical_pairing_is_permutation"],
        result["logicals"]["logical_z_supports_disjoint"],
        result["logicals"]["logical_x_supports_disjoint"],
        result["logicals"]["logical_z_commutes_with_hx"],
        result["logicals"]["logical_x_commutes_with_hz"],
        result["logicals"]["logical_z_independent_mod_stabilizers"] == 4,
        result["logicals"]["logical_x_independent_mod_stabilizers"] == 4,
        result["logicals"]["translation_cycles_logical_z"],
        result["logicals"]["fold_maps_logical_z_to_logical_x"],
    ]
    result["all_structural_checks_pass"] = all(hard_checks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    results = [validate(path) for path in args.paths]
    payload = {"candidates": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
