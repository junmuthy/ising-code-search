#!/usr/bin/env python3
"""Regenerate the package's plain-text matrix exports and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def support(row: np.ndarray) -> list[int]:
    return np.flatnonzero(row).astype(int).tolist()


def batches(labels: np.ndarray) -> list[list[int]]:
    return [
        np.flatnonzero(labels == label).astype(int).tolist()
        for label in sorted(set(map(int, labels)))
    ]


def main() -> None:
    output_directory = DIRECTORY / "matrices_csv"
    output_directory.mkdir(exist_ok=True)
    with np.load(DIRECTORY / "bb64_two_batch_basis.npz") as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    for name, array in arrays.items():
        np.savetxt(
            output_directory / f"{name}.csv",
            array.reshape(1, -1) if array.ndim == 1 else array,
            delimiter=",",
            fmt="%d",
        )

    logical_z = arrays["logical_z"]
    logical_x = arrays["logical_x"]
    map_x = np.argmax(arrays["grid_x_z_logical_action"], axis=1)
    map_y = np.argmax(arrays["grid_y_z_logical_action"], axis=1)
    schedule = json.loads((DIRECTORY / "baseline_schedule.json").read_text())
    metadata = {
        "schema_version": 1,
        "code": "[[64,8,8]] self-dual bivariate-bicycle code",
        "source": "Liang and Chen, arXiv:2510.05211v2",
        "construction": {
            "relations": ["x^4 y^4=1", "y^8=1"],
            "polynomials": ["f=1+x+y+y^-1", "g=f^dagger"],
            "checks": "H_X=H_Z=[F|F^T]",
            "displayed_check_shape": [32, 64],
            "rank_each_css_type": 28,
            "displayed_check_weight": 8,
        },
        "indexing": {
            "L": "q_(8*x+y), 0<=x<4, 0<=y<8",
            "R": "q_(32+8*x+y), 0<=x<4, 0<=y<8",
            "permutation_convention": "output[permutation[i]] = input[i]",
        },
        "logical_z_supports": [support(row) for row in logical_z],
        "logical_x_supports": [support(row) for row in logical_x],
        "logical_z_weights": logical_z.sum(axis=1).astype(int).tolist(),
        "logical_x_weights": logical_x.sum(axis=1).astype(int).tolist(),
        "disjoint_z_batches": batches(arrays["disjoint_batch_of_z_logical"]),
        "disjoint_x_batches": batches(arrays["disjoint_batch_of_x_logical"]),
        "hadamard_permutation": arrays["hadamard_permutation"].astype(int).tolist(),
        "grid_x_logical_map": map_x.astype(int).tolist(),
        "grid_y_logical_map": map_y.astype(int).tolist(),
        "grid_x_physical_permutation": arrays[
            "grid_x_physical_permutation"
        ].astype(int).tolist(),
        "grid_y_physical_permutation": arrays[
            "grid_y_physical_permutation"
        ].astype(int).tolist(),
        "schedule": {
            "file": "baseline_schedule.json",
            "schedule_id": schedule["schedule_id"],
            "cnot_depth": schedule["cnot_depth"],
            "cnot_count": schedule["cnot_count"],
        },
        "claim_boundary": {
            "distance": "independently reproducible with verify_distance.py",
            "two_batch_search": "exact for the eight saved affine candidates at weight 8",
            "fault_distance": "supplementary retained certificates; not needed to reconstruct the code",
        },
    }
    (DIRECTORY / "code_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(arrays)} CSV files and code_metadata.json")


if __name__ == "__main__":
    main()
