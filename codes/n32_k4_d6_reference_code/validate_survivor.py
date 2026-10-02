#!/usr/bin/env python3
"""Independent qLDPC and structural validation of a folded n=32 survivor."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from collections import deque
from typing import Any

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli


NUM_QUBITS = 32
LOGICAL_ORDER = 4
THICKNESS = 8


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray(
        [[int(mask) >> qubit & 1 for qubit in range(NUM_QUBITS)] for mask in masks],
        dtype=np.uint8,
    )


def matrix_from_supports(supports: list[list[int]]) -> np.ndarray:
    output = np.zeros((len(supports), NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(supports):
        output[row, support] = 1
    return output


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    return gf2_rank(left) == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def disjoint_rows(matrix: np.ndarray) -> bool:
    overlap = matrix @ matrix.T
    return bool(np.array_equal(overlap, np.diag(np.diag(overlap))))


def tanner_connected(hx: np.ndarray, hz: np.ndarray) -> bool:
    checks = np.vstack([hx, hz])
    size = len(checks) + NUM_QUBITS
    adjacency = [set() for _ in range(size)]
    for row_index, row in enumerate(checks):
        for qubit in np.flatnonzero(row):
            qnode = len(checks) + int(qubit)
            adjacency[row_index].add(qnode)
            adjacency[qnode].add(row_index)
    reached = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return len(reached) == size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    record = json.loads(args.input.read_text())
    analysis = record["result"]["best"] if "result" in record else record
    hx = matrix_from_masks(analysis["stabilizer_masks_x"])
    hz = matrix_from_masks(analysis["stabilizer_masks_z"])
    logical_z = matrix_from_supports(analysis["logical_supports_z"])
    logical_x = matrix_from_supports(analysis["logical_supports_x"])
    permutation = np.asarray(analysis["permutation"], dtype=int)
    translation = np.asarray(
        [
            ((qubit // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS
            + qubit % THICKNESS
            for qubit in range(NUM_QUBITS)
        ],
        dtype=int,
    )
    inverse_translation = np.argsort(translation)
    conjugated = permutation[translation[np.argsort(permutation)]]
    code = codes.CSSCode(hx, hz)
    distance_x = code.get_distance_exact(Pauli.X)
    distance_z = code.get_distance_exact(Pauli.Z)
    pairing = (logical_z @ logical_x.T) % 2
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    result = {
        "source": str(args.input.resolve()),
        "qldpc": {
            "n": int(code.num_qubits),
            "k": int(code.dimension),
            "rank_x": int(code.code_x.rank),
            "rank_z": int(code.code_z.rank),
            "distance_x": int(distance_x),
            "distance_z": int(distance_z),
            "distance": int(min(distance_x, distance_z)),
        },
        "checks": {
            "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
            "rank_x": rank_x,
            "rank_z": rank_z,
            "matrices_equal": bool(np.array_equal(hx, hz)),
            "rowspaces_equal": same_rowspace(hx, hz),
            "fold_maps_hx_rowspace_to_hz": same_rowspace(
                act_rows(hx, permutation), hz
            ),
            "fold_maps_hz_rowspace_to_hx": same_rowspace(
                act_rows(hz, permutation), hx
            ),
            "translation_preserves_hx": same_rowspace(act_rows(hx, translation), hx),
            "translation_preserves_hz": same_rowspace(act_rows(hz, translation), hz),
            "tanner_connected": tanner_connected(hx, hz),
            "minimum_basis_row_weights_x": analysis["row_weights_x"],
            "maximum_check_weight": int(analysis["maximum_check_weight"]),
        },
        "fold": {
            "permutation": permutation.astype(int).tolist(),
            "involution": bool(np.array_equal(permutation[permutation], np.arange(NUM_QUBITS))),
            "conjugates_translation_to_inverse": bool(
                np.array_equal(conjugated, inverse_translation)
            ),
            "pairing": pairing.astype(int).tolist(),
            "pairing_is_permutation": bool(
                np.all(pairing.sum(axis=0) == 1) and np.all(pairing.sum(axis=1) == 1)
            ),
        },
        "logicals": {
            "z_weights": np.count_nonzero(logical_z, axis=1).astype(int).tolist(),
            "x_weights": np.count_nonzero(logical_x, axis=1).astype(int).tolist(),
            "z_disjoint": disjoint_rows(logical_z),
            "x_disjoint": disjoint_rows(logical_x),
            "z_commutes_with_hx": bool(not np.any((logical_z @ hx.T) % 2)),
            "x_commutes_with_hz": bool(not np.any((logical_x @ hz.T) % 2)),
            "z_independent_mod_stabilizers": gf2_rank(np.vstack([hz, logical_z])) - rank_z,
            "x_independent_mod_stabilizers": gf2_rank(np.vstack([hx, logical_x])) - rank_x,
            "fold_maps_z_to_x": bool(
                np.array_equal(act_rows(logical_z, permutation), logical_x)
            ),
            "translation_cycles_z": bool(
                np.array_equal(act_rows(logical_z, translation), np.roll(logical_z, 1, axis=0))
                or np.array_equal(act_rows(logical_z, translation), np.roll(logical_z, -1, axis=0))
            ),
            "translation_cycles_x": bool(
                np.array_equal(act_rows(logical_x, translation), np.roll(logical_x, 1, axis=0))
                or np.array_equal(act_rows(logical_x, translation), np.roll(logical_x, -1, axis=0))
            ),
        },
    }
    required = [
        result["qldpc"]["n"] == 32,
        result["qldpc"]["k"] == 4,
        result["qldpc"]["distance_x"] == 6,
        result["qldpc"]["distance_z"] == 6,
        result["checks"]["css_orthogonal"],
        not result["checks"]["rowspaces_equal"],
        result["checks"]["fold_maps_hx_rowspace_to_hz"],
        result["checks"]["fold_maps_hz_rowspace_to_hx"],
        result["checks"]["translation_preserves_hx"],
        result["checks"]["translation_preserves_hz"],
        result["checks"]["tanner_connected"],
        result["fold"]["involution"],
        result["fold"]["conjugates_translation_to_inverse"],
        result["fold"]["pairing_is_permutation"],
        result["logicals"]["z_disjoint"],
        result["logicals"]["x_disjoint"],
        result["logicals"]["z_commutes_with_hx"],
        result["logicals"]["x_commutes_with_hz"],
        result["logicals"]["z_independent_mod_stabilizers"] == 4,
        result["logicals"]["x_independent_mod_stabilizers"] == 4,
        result["logicals"]["fold_maps_z_to_x"],
        result["logicals"]["translation_cycles_z"],
        result["logicals"]["translation_cycles_x"],
    ]
    result["all_required_checks_pass"] = all(required)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
