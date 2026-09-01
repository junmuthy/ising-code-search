#!/usr/bin/env python3
"""Build a self-contained data package for the published [[64,8,8]] BB code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8)
    size = len(matrix)
    augmented = np.hstack([matrix.copy(), np.eye(size, dtype=np.uint8)])
    for column in range(size):
        pivot = next(row for row in range(column, size) if augmented[row, column])
        augmented[[column, pivot]] = augmented[[pivot, column]]
        for row in range(size):
            if row != column and augmented[row, column]:
                augmented[row] ^= augmented[column]
    return augmented[:, size:]


def independent_row_indices(matrix: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    current = np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    rank = 0
    for index, row in enumerate(np.asarray(matrix, dtype=np.uint8)):
        candidate = np.vstack([current, row])
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            selected.append(index)
            current = candidate
            rank = candidate_rank
    return np.asarray(selected, dtype=np.int64)


def permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=np.uint8)
    matrix[np.arange(len(permutation)), permutation] = 1
    return matrix


def symplectic_basis_change(pairing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return S,K with S pairing S^T = K = four 2x2 exchange blocks."""
    size = len(pairing)
    vectors = [np.eye(size, dtype=np.uint8)[index] for index in range(size)]
    output: list[np.ndarray] = []
    while vectors:
        left = vectors.pop(0)
        partner_index = next(
            index
            for index, candidate in enumerate(vectors)
            if int((left @ pairing @ candidate) % 2) == 1
        )
        right = vectors.pop(partner_index)
        output.extend([left, right])
        orthogonalized = []
        for vector in vectors:
            coefficient_left = int((vector @ pairing @ left) % 2)
            coefficient_right = int((vector @ pairing @ right) % 2)
            orthogonalized.append(
                (vector ^ (coefficient_right * left) ^ (coefficient_left * right)).astype(
                    np.uint8
                )
            )
        vectors = orthogonalized
    change = np.asarray(output, dtype=np.uint8)
    transformed = (change @ pairing @ change.T) % 2
    return change, transformed


def supports(matrix: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def save_csv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, np.asarray(matrix), fmt="%d", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    csv_dir = args.output_dir / "matrices_csv"
    csv_dir.mkdir()

    saved = np.load(args.source)
    hx = np.asarray(saved["matrix_x"], dtype=np.uint8)
    hz = np.asarray(saved["matrix_z"], dtype=np.uint8)
    raw_z = np.asarray(saved["orbit"], dtype=np.uint8)
    tx = np.asarray(saved["grid_x_permutation"], dtype=np.int64)
    ty = np.asarray(saved["grid_y_permutation"], dtype=np.int64)
    tx_matrix = permutation_matrix(tx)
    ty_matrix = permutation_matrix(ty)

    pairing = (raw_z @ raw_z.T) % 2
    pairing_inverse = gf2_inverse(pairing)
    raw_dual_x = (pairing_inverse @ raw_z) % 2
    assert np.array_equal((raw_z @ raw_dual_x.T) % 2, np.eye(8, dtype=np.uint8))

    change, logical_h_action = symplectic_basis_change(pairing)
    change_inverse = gf2_inverse(change)
    dual_change = change_inverse.T
    hperm_z = (change @ raw_z) % 2
    hperm_x = (dual_change @ raw_dual_x) % 2
    assert np.array_equal((hperm_z @ hperm_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert np.array_equal((logical_h_action @ hperm_x) % 2, hperm_z)
    assert np.array_equal((logical_h_action @ hperm_z) % 2, hperm_x)

    raw_tx_action = ((raw_z @ tx_matrix) @ raw_dual_x.T) % 2
    raw_ty_action = ((raw_z @ ty_matrix) @ raw_dual_x.T) % 2
    hperm_tx_z_action = (change @ raw_tx_action @ change_inverse) % 2
    hperm_ty_z_action = (change @ raw_ty_action @ change_inverse) % 2
    hperm_tx_x_action = ((hperm_x @ tx_matrix) @ hperm_z.T) % 2
    hperm_ty_x_action = ((hperm_x @ ty_matrix) @ hperm_z.T) % 2

    rows_x = independent_row_indices(hx)
    rows_z = independent_row_indices(hz)
    zeros = np.zeros_like(hx)
    stabilizer_symplectic = np.vstack(
        [np.hstack([hx, zeros]), np.hstack([zeros, hz])]
    )
    stabilizer_symplectic_independent = np.vstack(
        [
            np.hstack([hx[rows_x], np.zeros_like(hx[rows_x])]),
            np.hstack([np.zeros_like(hz[rows_z]), hz[rows_z]]),
        ]
    )

    elements = np.asarray([(xx, yy) for xx in range(4) for yy in range(8)], dtype=np.int64)
    batches = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    for batch in batches:
        assert not np.any(raw_z[batch[0]] & raw_z[batch[1]])

    np.savez_compressed(
        args.output_dir / "bb64_complete.npz",
        matrix_x=hx,
        matrix_z=hz,
        matrix_x_independent=hx[rows_x],
        matrix_z_independent=hz[rows_z],
        independent_x_row_indices=rows_x,
        independent_z_row_indices=rows_z,
        stabilizer_symplectic=stabilizer_symplectic,
        stabilizer_symplectic_independent=stabilizer_symplectic_independent,
        raw_grid_z=raw_z,
        raw_same_support_x=raw_z,
        raw_dual_x=raw_dual_x,
        raw_zx_pairing=pairing,
        raw_zx_pairing_inverse=pairing_inverse,
        raw_disjoint_batches=batches,
        hperm_change_from_raw_z=change,
        hperm_change_from_raw_z_inverse=change_inverse,
        hperm_z=hperm_z,
        hperm_x=hperm_x,
        hperm_zx_pairing=(hperm_z @ hperm_x.T) % 2,
        logical_h_action=logical_h_action,
        logical_h_permutation=np.argmax(logical_h_action, axis=1),
        grid_x_physical_permutation=tx,
        grid_y_physical_permutation=ty,
        grid_x_physical_matrix=tx_matrix,
        grid_y_physical_matrix=ty_matrix,
        raw_grid_x_logical_action=raw_tx_action,
        raw_grid_y_logical_action=raw_ty_action,
        hperm_grid_x_z_action=hperm_tx_z_action,
        hperm_grid_y_z_action=hperm_ty_z_action,
        hperm_grid_x_x_action=hperm_tx_x_action,
        hperm_grid_y_x_action=hperm_ty_x_action,
        bottom_group_elements=elements,
        physical_hadamard_permutation=np.arange(64, dtype=np.int64),
    )

    csv_matrices = {
        "H_X.csv": hx,
        "H_Z.csv": hz,
        "H_X_independent.csv": hx[rows_x],
        "H_Z_independent.csv": hz[rows_z],
        "raw_grid_Z.csv": raw_z,
        "raw_dual_X.csv": raw_dual_x,
        "raw_ZX_pairing.csv": pairing,
        "hadamard_permutation_Z.csv": hperm_z,
        "hadamard_permutation_X.csv": hperm_x,
        "logical_H_action.csv": logical_h_action,
        "change_raw_to_hadamard_basis.csv": change,
        "grid_x_physical_permutation_matrix.csv": tx_matrix,
        "grid_y_physical_permutation_matrix.csv": ty_matrix,
        "grid_x_raw_logical_action.csv": raw_tx_action,
        "grid_y_raw_logical_action.csv": raw_ty_action,
    }
    for name, matrix in csv_matrices.items():
        save_csv(csv_dir / name, matrix)

    metadata = {
        "code": "[[64,8,8]] self-dual bivariate bicycle code",
        "source": "Liang and Chen, arXiv:2510.05211v2",
        "source_url": "https://arxiv.org/abs/2510.05211",
        "polynomials": {"f": "1+x+y+y^-1", "g": "f^dagger"},
        "ring_relations": ["x^4 y^4 = 1", "y^8 = 1"],
        "n": 64,
        "k": 8,
        "distance": 8,
        "distance_evidence": "published exact result; included weight-8 logical is an upper witness",
        "check_rows_per_type": 32,
        "independent_checks_per_type": 28,
        "check_weight": 8,
        "physical_qubit_indexing": {
            "left_half": "0..31",
            "right_half": "32..63",
            "bottom_index": "8*x+y for x in 0..3 and y in 0..7",
        },
        "raw_grid_order": [
            "1", "T2", "T4", "T4*T2", "T4^2", "T4^2*T2", "T4^3", "T4^3*T2"
        ],
        "raw_disjoint_batches": batches.tolist(),
        "logical_hadamard_permutation": np.argmax(logical_h_action, axis=1).tolist(),
        "logical_hadamard_description": (
            "transversal H maps Z_i to X_pi(i) and X_i to Z_pi(i), "
            "with pi=(0 1)(2 3)(4 5)(6 7)"
        ),
        "basis_caveat": (
            "raw_grid_z has the disjoint batches; hperm_z/hperm_x has canonical "
            "Hadamard-plus-permutation action. They are related by the saved GF(2) basis change, "
            "but the hperm basis is not the raw-disjoint support presentation."
        ),
        "weights": {
            "raw_grid_z": np.count_nonzero(raw_z, axis=1).astype(int).tolist(),
            "raw_dual_x": np.count_nonzero(raw_dual_x, axis=1).astype(int).tolist(),
            "hperm_z": np.count_nonzero(hperm_z, axis=1).astype(int).tolist(),
            "hperm_x": np.count_nonzero(hperm_x, axis=1).astype(int).tolist(),
        },
        "supports": {
            "raw_grid_z": supports(raw_z),
            "raw_dual_x": supports(raw_dual_x),
            "hperm_z": supports(hperm_z),
            "hperm_x": supports(hperm_x),
        },
    }
    (args.output_dir / "code_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
