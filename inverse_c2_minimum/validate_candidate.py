#!/usr/bin/env python3
"""Independently validate saved minimum-C2 candidates with qLDPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

from inverse_c2_minimum.search import gf2_rank


def matrix_from_supports(supports: list[list[int]], n: int) -> np.ndarray:
    matrix = np.zeros((len(supports), n), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    rank_left = gf2_rank(left)
    rank_right = gf2_rank(right)
    return rank_left == rank_right == gf2_rank(np.vstack([left, right]))


def supports_disjoint(matrix: np.ndarray) -> bool:
    return bool(np.all(np.count_nonzero(matrix, axis=0) <= 1))


def validate(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    analysis = source["analysis"]
    n = int(analysis["n"])
    hx = matrix_from_supports(analysis["basis_x"], n)
    hz = matrix_from_supports(analysis["basis_z"], n)
    logical_x = matrix_from_supports(analysis["logical_x"], n)
    logical_z = matrix_from_supports(analysis["logical_z"], n)
    fold = np.asarray(analysis["fold"]["permutation"], dtype=int)
    translation = np.asarray(analysis["translation"], dtype=int)

    code = codes.CSSCode(hx, hz)
    distance_x = int(code.get_distance_exact(Pauli.X))
    distance_z = int(code.get_distance_exact(Pauli.Z))
    pairing = (logical_z @ logical_x.T) % 2
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    result = {
        "source": str(path.resolve()),
        "qldpc": {
            "n": int(code.num_qubits),
            "k": int(code.dimension),
            "distance_x": distance_x,
            "distance_z": distance_z,
            "distance": min(distance_x, distance_z),
        },
        "checks": {
            "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
            "rank_x": rank_x,
            "rank_z": rank_z,
            "fold_maps_hx_to_hz": bool(np.array_equal(act_rows(hx, fold), hz)),
            "translation_preserves_hx": same_rowspace(act_rows(hx, translation), hx),
            "translation_preserves_hz": same_rowspace(act_rows(hz, translation), hz),
        },
        "symmetry": {
            "fold_is_involution": bool(np.array_equal(fold[fold], np.arange(n))),
            "translation_is_involution": bool(
                np.array_equal(translation[translation], np.arange(n))
            ),
            "fold_commutes_with_translation": bool(
                np.array_equal(fold[translation], translation[fold])
            ),
            "translation_swaps_logical_z": bool(
                np.array_equal(act_rows(logical_z, translation), logical_z[::-1])
            ),
            "translation_swaps_logical_x": bool(
                np.array_equal(act_rows(logical_x, translation), logical_x[::-1])
            ),
            "logical_pairing": pairing.astype(int).tolist(),
            "logical_pairing_is_permutation": bool(
                np.all(pairing.sum(axis=0) == 1)
                and np.all(pairing.sum(axis=1) == 1)
            ),
        },
        "logicals": {
            "logical_z_weights": np.count_nonzero(logical_z, axis=1).tolist(),
            "logical_x_weights": np.count_nonzero(logical_x, axis=1).tolist(),
            "logical_z_disjoint": supports_disjoint(logical_z),
            "logical_x_disjoint": supports_disjoint(logical_x),
            "logical_z_commutes_with_hx": bool(not np.any((logical_z @ hx.T) % 2)),
            "logical_x_commutes_with_hz": bool(not np.any((logical_x @ hz.T) % 2)),
            "logical_z_rank_gain": gf2_rank(np.vstack([hz, logical_z])) - rank_z,
            "logical_x_rank_gain": gf2_rank(np.vstack([hx, logical_x])) - rank_x,
        },
    }
    result["all_required_checks_pass"] = all(
        (
            result["qldpc"]["k"] == 2,
            result["qldpc"]["distance"] >= 6,
            *result["checks"].values(),
            *[
                value
                for key, value in result["symmetry"].items()
                if key != "logical_pairing"
            ],
            result["logicals"]["logical_z_disjoint"],
            result["logicals"]["logical_x_disjoint"],
            result["logicals"]["logical_z_commutes_with_hx"],
            result["logicals"]["logical_x_commutes_with_hz"],
            result["logicals"]["logical_z_rank_gain"] == 2,
            result["logicals"]["logical_x_rank_gain"] == 2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = {"candidates": [validate(path) for path in args.paths]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
