#!/usr/bin/env python3
"""Verify the code, simultaneous logical basis, symmetries, and exact covers."""

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


def same_row_space(left: np.ndarray, right: np.ndarray) -> bool:
    return gf2_rank(left) == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def permute_supports(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def is_permutation_matrix(matrix: np.ndarray) -> bool:
    return bool(np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1))


def batches(labels: np.ndarray) -> list[list[int]]:
    return [
        np.flatnonzero(labels == label).astype(int).tolist()
        for label in sorted(set(map(int, labels)))
    ]


def main() -> None:
    archive = np.load(Path(__file__).with_name("bb64_weight8_basis.npz"))
    hx = np.asarray(archive["matrix_x"], dtype=np.uint8)
    hz = np.asarray(archive["matrix_z"], dtype=np.uint8)
    logical_z = np.asarray(archive["logical_z"], dtype=np.uint8)
    logical_x = np.asarray(archive["logical_x"], dtype=np.uint8)
    permutation_h = np.asarray(archive["hadamard_permutation"], dtype=np.int64)

    rebuilt_x, rebuilt_z = construct_checks()
    assert np.array_equal(hx, rebuilt_x)
    assert np.array_equal(hz, rebuilt_z)
    assert hx.shape == hz.shape == (32, 64)
    assert np.array_equal(hx, hz)
    assert gf2_rank(hx) == gf2_rank(hz) == 28
    assert 64 - gf2_rank(hx) - gf2_rank(hz) == 8
    assert not np.any((hx @ hz.T) % 2)
    assert set(np.count_nonzero(hx, axis=1)) == {8}
    print("[1/5] checks: self-dual [[64,8,*]], CSS-commuting, uniform weight 8")

    assert not np.any((hx @ logical_z.T) % 2)
    assert not np.any((hz @ logical_x.T) % 2)
    assert gf2_rank(np.vstack([hz, logical_z])) - gf2_rank(hz) == 8
    assert gf2_rank(np.vstack([hx, logical_x])) - gf2_rank(hx) == 8
    assert np.array_equal((logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8))
    assert np.all(logical_z.sum(axis=1) == 8)
    assert np.all(logical_x.sum(axis=1) == 8)
    print("[2/5] basis: eight canonical X/Z pairs, every representative weight 8")

    z_batches = batches(archive["disjoint_batch_of_z_logical"])
    x_batches = batches(archive["disjoint_batch_of_x_logical"])
    assert z_batches == [[0, 1, 7], [4, 6], [2, 3, 5]]
    assert x_batches == [[2, 4, 5], [1, 3], [0, 6, 7]]
    assert sorted(item for batch in z_batches for item in batch) == list(range(8))
    assert sorted(item for batch in x_batches for item in batch) == list(range(8))
    assert all(np.all(logical_z[batch].sum(axis=0) <= 1) for batch in z_batches)
    assert all(np.all(logical_x[batch].sum(axis=0) <= 1) for batch in x_batches)
    print("[3/5] exact covers: internally disjoint 3+2+3 batches in both bases")

    physical_x = np.asarray(archive["grid_x_physical_permutation"], dtype=np.int64)
    physical_y = np.asarray(archive["grid_y_physical_permutation"], dtype=np.int64)
    assert same_row_space(hx, permute_supports(hx, physical_x))
    assert same_row_space(hx, permute_supports(hx, physical_y))
    action_x_z = (permute_supports(logical_z, physical_x) @ logical_x.T) % 2
    action_y_z = (permute_supports(logical_z, physical_y) @ logical_x.T) % 2
    action_x_x = (permute_supports(logical_x, physical_x) @ logical_z.T) % 2
    action_y_x = (permute_supports(logical_x, physical_y) @ logical_z.T) % 2
    for name, calculated in (
        ("grid_x_z_logical_action", action_x_z),
        ("grid_y_z_logical_action", action_y_z),
        ("grid_x_x_logical_action", action_x_x),
        ("grid_y_x_logical_action", action_y_x),
    ):
        assert is_permutation_matrix(calculated)
        assert np.array_equal(calculated, archive[name])
    map_x = np.argmax(action_x_z, axis=1)
    map_y = np.argmax(action_y_z, axis=1)
    assert map_x.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]
    assert map_y.tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
    assert np.array_equal(map_x[map_y], map_y[map_x])
    print("[4/5] grid: regular commuting C4 x C2 logical translations")

    assert permutation_h.tolist() == [5, 4, 7, 6, 1, 0, 3, 2]
    assert np.array_equal(permutation_h[permutation_h], np.arange(8))
    # Exact support equalities for H: Z_i -> X_p(i), X_i -> Z_p(i).
    assert np.array_equal(logical_z, logical_x[permutation_h])
    assert np.array_equal(logical_x, logical_z[permutation_h])
    assert np.array_equal(permutation_h, map_y[map_x[map_x]])
    print("[5/5] H: transversal H plus p=(0 5)(1 4)(2 7)(3 6)=T_x^2 T_y")
    print("All algebraic package checks passed. Published exact distance: d=8.")


if __name__ == "__main__":
    main()
