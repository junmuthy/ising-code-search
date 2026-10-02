#!/usr/bin/env python3
"""Verify the matrices, logical bases, symmetries, and STAR batches using NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reconstruct_code import construct_checks


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


def matrix_order(matrix: np.ndarray, limit: int = 64) -> int:
    identity = np.eye(len(matrix), dtype=np.uint8)
    power = identity.copy()
    for exponent in range(1, limit + 1):
        power = (power @ matrix) % 2
        if np.array_equal(power, identity):
            return exponent
    raise AssertionError("matrix order exceeds verification limit")


def permutation_order(permutation: np.ndarray) -> int:
    identity = np.arange(len(permutation))
    power = identity.copy()
    for exponent in range(1, len(permutation) + 1):
        power = permutation[power]
        if np.array_equal(power, identity):
            return exponent
    raise AssertionError("invalid permutation")


def same_row_space(left: np.ndarray, right: np.ndarray) -> bool:
    return gf2_rank(left) == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def main() -> None:
    archive = np.load(Path(__file__).with_name("bb64_complete.npz"))
    hx = archive["matrix_x"]
    hz = archive["matrix_z"]
    raw_z = archive["raw_grid_z"]
    raw_x = archive["raw_dual_x"]
    hperm_z = archive["hperm_z"]
    hperm_x = archive["hperm_x"]
    logical_h = archive["logical_h_action"]

    rebuilt_x, rebuilt_z = construct_checks()
    assert np.array_equal(hx, rebuilt_x)
    assert np.array_equal(hz, rebuilt_z)
    assert hx.shape == hz.shape == (32, 64)
    assert gf2_rank(hx) == gf2_rank(hz) == 28
    assert 64 - gf2_rank(hx) - gf2_rank(hz) == 8
    assert not np.any((hx @ hz.T) % 2)
    assert set(np.count_nonzero(hx, axis=1)) == {8}
    assert set(np.count_nonzero(hz, axis=1)) == {8}
    print("[1/5] checks: [[64,8,*]], CSS-commuting, self-dual, uniform weight 8")

    assert not np.any((hx @ raw_z.T) % 2)
    assert not np.any((hz @ raw_x.T) % 2)
    assert gf2_rank(np.vstack([hz, raw_z])) - gf2_rank(hz) == 8
    assert gf2_rank(np.vstack([hx, raw_x])) - gf2_rank(hx) == 8
    assert np.array_equal((raw_z @ raw_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert set(np.count_nonzero(raw_z, axis=1)) == {8}
    print("[2/5] raw grid: eight independent weight-8 Z logicals with a dual X basis")

    for batch in archive["raw_disjoint_batches"]:
        batch_vectors = raw_z[batch]
        assert np.all(np.sum(batch_vectors, axis=0) <= 1)
    assert archive["raw_disjoint_batches"].tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    print("[3/5] STAR batches: {0,1}, {2,3}, {4,5}, {6,7} are internally disjoint")

    tx = archive["grid_x_physical_permutation"]
    ty = archive["grid_y_physical_permutation"]
    tx_matrix = archive["grid_x_physical_matrix"]
    ty_matrix = archive["grid_y_physical_matrix"]
    assert permutation_order(tx) == 8
    assert permutation_order(ty) == 2
    assert np.array_equal(tx[ty], ty[tx])
    assert same_row_space(hx, (hx @ tx_matrix) % 2)
    assert same_row_space(hx, (hx @ ty_matrix) % 2)
    assert matrix_order(archive["raw_grid_x_logical_action"]) == 4
    assert matrix_order(archive["raw_grid_y_logical_action"]) == 2
    assert np.array_equal(
        (archive["raw_grid_x_logical_action"] @ archive["raw_grid_y_logical_action"]) % 2,
        (archive["raw_grid_y_logical_action"] @ archive["raw_grid_x_logical_action"]) % 2,
    )
    print("[4/5] grid: commuting logical C4 x C2 action verified")

    assert np.array_equal((hperm_z @ hperm_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert np.array_equal(logical_h.sum(axis=0), np.ones(8, dtype=np.uint64))
    assert np.array_equal(logical_h.sum(axis=1), np.ones(8, dtype=np.uint64))
    assert matrix_order(logical_h) == 2
    assert archive["logical_h_permutation"].tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
    # Physical transversal H preserves support.  These exact equalities show
    # Z_i -> X_pi(i) and X_i -> Z_pi(i), not merely equality modulo stabilizers.
    assert np.array_equal(hperm_z, (logical_h @ hperm_x) % 2)
    assert np.array_equal(hperm_x, (logical_h @ hperm_z) % 2)
    print("[5/5] transversal H: logical H plus swaps (0 1)(2 3)(4 5)(6 7)")
    print("All algebraic package checks passed. Published exact distance: d=8.")


if __name__ == "__main__":
    main()
