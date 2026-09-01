#!/usr/bin/env python3
"""Certify and save the all-weight-eight BB64 simultaneous basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from certify_weight72 import (
    gf2_rank,
    permutation_matrix,
    permutation_order,
    permute_rows,
    same_row_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--improved-batch-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / "simultaneous_basis_weight64.npz"
    certificate_path = args.output_dir / "weight64_certificate.json"
    if output_path.exists() or certificate_path.exists():
        raise SystemExit("refusing to overwrite an existing weight-64 certificate")

    source = np.load(args.source_npz)
    improvement = np.load(args.improved_batch_npz)
    check_x = np.asarray(source["matrix_x"], dtype=np.uint8)
    check_z = np.asarray(source["matrix_z"], dtype=np.uint8)
    logical_z = np.asarray(improvement["logical_z"], dtype=np.uint8)
    hadamard_permutation = np.asarray(source["hadamard_permutation"], dtype=np.int64)
    logical_x = logical_z[hadamard_permutation]
    pairing = (logical_z @ logical_z.T) % 2
    permutation_x = np.asarray(source["grid_x_physical_permutation"], dtype=np.int64)
    permutation_y = np.asarray(source["grid_y_physical_permutation"], dtype=np.int64)
    labels_z = np.asarray(source["disjoint_batch_of_z_logical"], dtype=np.int64)

    assert np.array_equal(check_x, check_z)
    assert gf2_rank(check_x) == gf2_rank(check_z) == 28
    assert np.count_nonzero((check_x @ check_z.T) % 2) == 0
    assert np.count_nonzero((check_x @ logical_z.T) % 2) == 0
    assert np.count_nonzero((check_z @ logical_x.T) % 2) == 0
    assert gf2_rank(np.vstack([check_z, logical_z])) == 36
    assert np.array_equal((logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert permutation_matrix(pairing)
    assert np.array_equal(np.argmax(pairing, axis=1), hadamard_permutation)
    assert np.array_equal(hadamard_permutation[hadamard_permutation], np.arange(8))
    assert np.all(logical_z.sum(axis=1) == 8)
    assert np.all(logical_x.sum(axis=1) == 8)

    batches_z = [
        np.flatnonzero(labels_z == label).astype(int).tolist()
        for label in sorted(set(map(int, labels_z)))
    ]
    assert sorted(logical for batch in batches_z for logical in batch) == list(range(8))
    assert all(len(batch) >= 2 for batch in batches_z)
    assert all(np.all(logical_z[batch].sum(axis=0) <= 1) for batch in batches_z)
    batches_x = [sorted(hadamard_permutation[batch].astype(int).tolist()) for batch in batches_z]
    assert sorted(logical for batch in batches_x for logical in batch) == list(range(8))
    assert all(np.all(logical_x[batch].sum(axis=0) <= 1) for batch in batches_x)
    labels_x = np.asarray(
        [
            next(batch_index for batch_index, batch in enumerate(batches_x) if logical in batch)
            for logical in range(8)
        ],
        dtype=np.int64,
    )

    assert same_row_space(check_z, permute_rows(check_z, permutation_x))
    assert same_row_space(check_z, permute_rows(check_z, permutation_y))
    action_x_z = (permute_rows(logical_z, permutation_x) @ logical_x.T) % 2
    action_y_z = (permute_rows(logical_z, permutation_y) @ logical_x.T) % 2
    action_x_x = (permute_rows(logical_x, permutation_x) @ logical_z.T) % 2
    action_y_x = (permute_rows(logical_x, permutation_y) @ logical_z.T) % 2
    assert all(
        permutation_matrix(action)
        for action in (action_x_z, action_y_z, action_x_x, action_y_x)
    )
    map_x_z = np.argmax(action_x_z, axis=1).astype(np.int64)
    map_y_z = np.argmax(action_y_z, axis=1).astype(np.int64)
    map_x_x = np.argmax(action_x_x, axis=1).astype(np.int64)
    map_y_x = np.argmax(action_y_x, axis=1).astype(np.int64)
    assert permutation_order(map_x_z) == permutation_order(map_x_x) == 4
    assert permutation_order(map_y_z) == permutation_order(map_y_x) == 2
    assert np.array_equal(map_x_z[map_y_z], map_y_z[map_x_z])
    assert np.array_equal(map_x_x[map_y_x], map_y_x[map_x_x])
    orbit = set()
    for xx in range(4):
        for yy in range(2):
            value = 0
            for _ in range(xx):
                value = int(map_x_z[value])
            for _ in range(yy):
                value = int(map_y_z[value])
            orbit.add(value)
    assert orbit == set(range(8))

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
        disjoint_batch_of_z_logical=labels_z,
        disjoint_batch_of_x_logical=labels_x,
    )
    certificate = {
        "code": "Liang-Chen self-dual [[64,8,8]] BB code",
        "same_basis_claim": True,
        "n": 64,
        "k": 8,
        "d": 8,
        "check_rank_each_css_type": 28,
        "check_weight_set": sorted(
            set(np.count_nonzero(check_x, axis=1).astype(int).tolist())
        ),
        "logical_z_weights": logical_z.sum(axis=1).astype(int).tolist(),
        "logical_x_weights": logical_x.sum(axis=1).astype(int).tolist(),
        "total_logical_z_weight": int(logical_z.sum()),
        "representative_weight_is_individually_optimal": True,
        "optimality_reason": "Every nontrivial logical has weight at least d=8.",
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
        "source_files": {
            "source_npz": str(args.source_npz),
            "improved_batch_npz": str(args.improved_batch_npz),
        },
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
    }
    atomic_json(certificate_path, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
