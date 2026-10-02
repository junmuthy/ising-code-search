#!/usr/bin/env python3
"""Save and certify the weight-72 simultaneous basis found in run_002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REFINED_SUPPORTS = [
    [2, 11, 12, 13, 22, 34, 36, 54],
    [6, 18, 20, 38, 43, 44, 45, 50],
    [7, 23, 24, 25, 26, 39, 49, 55],
    [5, 6, 12, 13, 19, 38, 51, 58],
    [0, 6, 16, 32, 48, 57, 58, 59],
    [8, 9, 15, 27, 28, 30, 31, 40, 41, 47, 48, 52, 54, 59, 60, 61],
    [4, 12, 21, 35, 36, 53, 62, 63],
    [1, 3, 17, 33, 49, 56, 62, 63],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def gf2_rank(matrix: np.ndarray) -> int:
    pivots: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def same_row_space(left: np.ndarray, right: np.ndarray) -> bool:
    return gf2_rank(left) == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def permutation_matrix(matrix: np.ndarray) -> bool:
    return bool(np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1))


def permutation_order(mapping: np.ndarray) -> int:
    power = np.arange(len(mapping), dtype=np.int64)
    identity = power.copy()
    for order in range(1, len(mapping) + 1):
        power = mapping[power]
        if np.array_equal(power, identity):
            return order
    raise AssertionError("not a permutation")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_path = args.run_dir / "simultaneous_basis.npz"
    output_path = args.run_dir / "simultaneous_basis_weight72.npz"
    certificate_path = args.run_dir / "weight72_certificate.json"
    if output_path.exists() or certificate_path.exists():
        raise SystemExit("refusing to overwrite an existing weight-72 certificate")

    source = np.load(source_path)
    check_x = np.asarray(source["matrix_x"], dtype=np.uint8)
    check_z = np.asarray(source["matrix_z"], dtype=np.uint8)
    permutation_x = np.asarray(source["grid_x_physical_permutation"], dtype=np.int64)
    permutation_y = np.asarray(source["grid_y_physical_permutation"], dtype=np.int64)
    labels = np.asarray(source["disjoint_batch_of_logical"], dtype=np.int64)
    hadamard_permutation = np.asarray(source["hadamard_permutation"], dtype=np.int64)
    logical_z = np.zeros((8, 64), dtype=np.uint8)
    for logical, support in enumerate(REFINED_SUPPORTS):
        logical_z[logical, support] = 1
    logical_x = logical_z[hadamard_permutation]
    pairing = (logical_z @ logical_z.T) % 2

    rank_x = gf2_rank(check_x)
    rank_z = gf2_rank(check_z)
    assert np.array_equal(check_x, check_z)
    assert rank_x == rank_z == 28
    assert np.count_nonzero((check_x @ check_z.T) % 2) == 0
    assert np.count_nonzero((check_x @ logical_z.T) % 2) == 0
    assert np.count_nonzero((check_z @ logical_x.T) % 2) == 0
    assert gf2_rank(np.vstack([check_z, logical_z])) == 36
    assert np.array_equal((logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert permutation_matrix(pairing)
    assert np.array_equal(np.argmax(pairing, axis=1), hadamard_permutation)
    assert np.array_equal(hadamard_permutation[hadamard_permutation], np.arange(8))

    batches_z = [np.flatnonzero(labels == label).astype(int).tolist() for label in sorted(set(labels))]
    flattened = [logical for batch in batches_z for logical in batch]
    assert sorted(flattened) == list(range(8)) and len(flattened) == len(set(flattened))
    assert all(len(batch) >= 2 for batch in batches_z)
    assert all(np.all(logical_z[batch].sum(axis=0) <= 1) for batch in batches_z)
    # X_j has the support of Z_p(j), so apply p^{-1}=p to each Z batch.
    batches_x = [sorted(hadamard_permutation[batch].astype(int).tolist()) for batch in batches_z]
    flattened_x = [logical for batch in batches_x for logical in batch]
    assert sorted(flattened_x) == list(range(8)) and len(flattened_x) == len(set(flattened_x))
    assert all(np.all(logical_x[batch].sum(axis=0) <= 1) for batch in batches_x)

    assert same_row_space(check_z, permute_rows(check_z, permutation_x))
    assert same_row_space(check_z, permute_rows(check_z, permutation_y))
    action_x_z = (permute_rows(logical_z, permutation_x) @ logical_x.T) % 2
    action_y_z = (permute_rows(logical_z, permutation_y) @ logical_x.T) % 2
    action_x_x = (permute_rows(logical_x, permutation_x) @ logical_z.T) % 2
    action_y_x = (permute_rows(logical_x, permutation_y) @ logical_z.T) % 2
    assert all(permutation_matrix(action) for action in (action_x_z, action_y_z, action_x_x, action_y_x))
    map_x_z = np.argmax(action_x_z, axis=1).astype(np.int64)
    map_y_z = np.argmax(action_y_z, axis=1).astype(np.int64)
    map_x_x = np.argmax(action_x_x, axis=1).astype(np.int64)
    map_y_x = np.argmax(action_y_x, axis=1).astype(np.int64)
    assert permutation_order(map_x_z) == permutation_order(map_x_x) == 4
    assert permutation_order(map_y_z) == permutation_order(map_y_x) == 2
    assert np.array_equal(map_x_z[map_y_z], map_y_z[map_x_z])
    assert np.array_equal(map_x_x[map_y_x], map_y_x[map_x_x])
    orbit = {0}
    for xx in range(4):
        for yy in range(2):
            value = 0
            for _ in range(xx):
                value = int(map_x_z[value])
            for _ in range(yy):
                value = int(map_y_z[value])
            orbit.add(value)
    assert orbit == set(range(8))

    check_weights = np.count_nonzero(check_x, axis=1).astype(int)
    certificate = {
        "code": "Liang-Chen self-dual [[64,8,8]] BB code",
        "source_run": str(source_path),
        "same_basis_claim": True,
        "n": 64,
        "k": 8,
        "d": 8,
        "check_rank_each_css_type": 28,
        "check_weight_set": sorted(set(check_weights.tolist())),
        "logical_z_weights": logical_z.sum(axis=1).astype(int).tolist(),
        "logical_x_weights": logical_x.sum(axis=1).astype(int).tolist(),
        "total_logical_z_weight": int(logical_z.sum()),
        "disjoint_z_batches": batches_z,
        "disjoint_x_batches": batches_x,
        "batch_sizes": [len(batch) for batch in batches_z],
        "exact_cover_each_basis": True,
        "hadamard_logical_permutation": hadamard_permutation.astype(int).tolist(),
        "hadamard_cycles": [[0, 5], [1, 4], [2, 7], [3, 6]],
        "grid_x_z_mapping": map_x_z.astype(int).tolist(),
        "grid_y_z_mapping": map_y_z.astype(int).tolist(),
        "grid_x_x_mapping": map_x_x.astype(int).tolist(),
        "grid_y_x_mapping": map_y_x.astype(int).tolist(),
        "grid_x_order": 4,
        "grid_y_order": 2,
        "grid_actions_commute": True,
        "grid_action_is_regular_and_transitive": True,
        "verification": {
            "self_dual_hx_equals_hz": True,
            "css_commutator_zero": True,
            "logical_kernel_checks": True,
            "eight_independent_logicals_mod_stabilizers": True,
            "canonical_zx_pairing": True,
            "physical_permutations_preserve_check_space": True,
            "z_and_x_grid_actions_are_permutations": True,
            "batch_disjointness_and_exact_cover": True,
        },
        "optimization_note": (
            "Three-batch feasibility is certified. The representative-weight pass found total "
            "Z weight 72; caps 24 and 28 for the remaining three-logical batch timed out, so "
            "global support-weight optimality is not claimed."
        ),
    }
    np.savez_compressed(
        output_path,
        matrix_x=check_x,
        matrix_z=check_z,
        logical_z=logical_z,
        logical_x=logical_x,
        zx_pairing=pairing,
        hadamard_permutation=hadamard_permutation,
        grid_x_physical_permutation=permutation_x,
        grid_y_physical_permutation=permutation_y,
        grid_x_z_logical_action=action_x_z,
        grid_y_z_logical_action=action_y_z,
        grid_x_x_logical_action=action_x_x,
        grid_y_x_logical_action=action_y_x,
        disjoint_batch_of_z_logical=labels,
        disjoint_batch_of_x_logical=np.asarray(
            [next(index for index, batch in enumerate(batches_x) if logical in batch) for logical in range(8)],
            dtype=np.int64,
        ),
    )
    atomic_json(certificate_path, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
