#!/usr/bin/env python3
"""Verify the BB64 checks, two-batch basis, symmetries, and syndrome schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from reconstruct_code import construct_checks


DIRECTORY = Path(__file__).resolve().parent


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


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def same_row_space(left: np.ndarray, right: np.ndarray) -> bool:
    rank = gf2_rank(left)
    return rank == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def permutation_order(permutation: np.ndarray) -> int:
    identity = np.arange(len(permutation), dtype=np.int64)
    power = identity.copy()
    for exponent in range(1, len(permutation) + 1):
        power = permutation[power]
        if np.array_equal(power, identity):
            return exponent
    raise AssertionError("invalid permutation")


def batches(labels: np.ndarray) -> list[list[int]]:
    return [
        np.flatnonzero(labels == label).astype(int).tolist()
        for label in sorted(set(map(int, labels)))
    ]


def check_checksums() -> None:
    checksum_path = DIRECTORY / "SHA256SUMS"
    if not checksum_path.exists():
        raise AssertionError("SHA256SUMS is missing")
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = DIRECTORY / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def verify_schedule(schedule: dict[str, object], hx: np.ndarray, hz: np.ndarray) -> None:
    assert schedule["schedule_id"] == "67e327dc7cb35335"
    assert schedule["cnot_depth"] == 8
    assert schedule["cnot_count"] == 512
    layers = schedule["layers"]
    assert isinstance(layers, list) and len(layers) == 8
    observed: dict[str, set[tuple[int, int]]] = {"X": set(), "Z": set()}
    for layer in layers:
        assert isinstance(layer, dict)
        gates = layer["gates"]
        assert isinstance(gates, list) and len(gates) == 64
        assert sorted(int(gate["data"]) for gate in gates) == list(range(64))
        for kind in ("X", "Z"):
            selected = [gate for gate in gates if gate["type"] == kind]
            assert sorted(int(gate["check"]) for gate in selected) == list(range(32))
            observed[kind].update(
                (int(gate["check"]), int(gate["data"])) for gate in selected
            )
    for kind, matrix in (("X", hx), ("Z", hz)):
        expected = {
            (check, int(data))
            for check, row in enumerate(matrix)
            for data in np.flatnonzero(row)
        }
        assert observed[kind] == expected


def main() -> None:
    check_checksums()
    print("[1/7] archive checksums match")

    with np.load(DIRECTORY / "bb64_two_batch_basis.npz") as archive:
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
        print("[2/7] checks reconstruct exactly: self-dual [[64,8,*]], rank 28+28")

        assert not np.any((hx @ logical_z.T) % 2)
        assert not np.any((hz @ logical_x.T) % 2)
        assert gf2_rank(np.vstack([hz, logical_z])) - gf2_rank(hz) == 8
        assert gf2_rank(np.vstack([hx, logical_x])) - gf2_rank(hx) == 8
        assert np.array_equal(
            (logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8)
        )
        assert np.all(logical_z.sum(axis=1) == 8)
        assert np.all(logical_x.sum(axis=1) == 8)
        print("[3/7] logical basis: eight canonical X/Z pairs, all weight eight")

        z_batches = batches(archive["disjoint_batch_of_z_logical"])
        x_batches = batches(archive["disjoint_batch_of_x_logical"])
        assert z_batches == [[0, 3, 4, 7], [1, 2, 5, 6]]
        assert x_batches == [[1, 2, 5, 6], [0, 3, 4, 7]]
        assert all(np.all(logical_z[batch].sum(axis=0) <= 1) for batch in z_batches)
        assert all(np.all(logical_x[batch].sum(axis=0) <= 1) for batch in x_batches)
        assert all(int(logical_z[batch].sum()) == 32 for batch in z_batches)
        assert all(int(logical_x[batch].sum()) == 32 for batch in x_batches)
        print("[4/7] exact covers: two internally disjoint 4+4 batches in X and Z")

        expected_h = np.asarray([5, 4, 7, 6, 1, 0, 3, 2], dtype=np.int64)
        assert np.array_equal(permutation_h, expected_h)
        assert np.array_equal(logical_x, logical_z[permutation_h])
        assert np.array_equal(logical_z, logical_x[permutation_h])
        assert np.array_equal(
            archive["zx_pairing"], (logical_z @ logical_z.T) % 2
        )
        assert np.array_equal(archive["zx_pairing"], np.eye(8)[permutation_h])
        print("[5/7] transversal H: p=(0 5)(1 4)(2 7)(3 6), exact on supports")

        physical_x = np.asarray(archive["grid_x_physical_permutation"], dtype=np.int64)
        physical_y = np.asarray(archive["grid_y_physical_permutation"], dtype=np.int64)
        assert sorted(physical_x.tolist()) == list(range(64))
        assert sorted(physical_y.tolist()) == list(range(64))
        assert permutation_order(physical_x) == 8
        assert permutation_order(physical_y) == 2
        assert np.array_equal(physical_x[physical_y], physical_y[physical_x])
        assert same_row_space(hx, permute_rows(hx, physical_x))
        assert same_row_space(hx, permute_rows(hx, physical_y))
        for axis, physical in (("x", physical_x), ("y", physical_y)):
            calculated_z = (permute_rows(logical_z, physical) @ logical_x.T) % 2
            calculated_x = (permute_rows(logical_x, physical) @ logical_z.T) % 2
            assert np.array_equal(calculated_z, archive[f"grid_{axis}_z_logical_action"])
            assert np.array_equal(calculated_x, archive[f"grid_{axis}_x_logical_action"])
        map_x = np.argmax(archive["grid_x_z_logical_action"], axis=1)
        map_y = np.argmax(archive["grid_y_z_logical_action"], axis=1)
        assert map_x.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]
        assert map_y.tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
        orbit: set[int] = set()
        for xx in range(4):
            value = 0
            for _ in range(xx):
                value = int(map_x[value])
            orbit.add(value)
            orbit.add(int(map_y[value]))
        assert orbit == set(range(8))
        print("[6/7] physical symmetries induce a regular commuting C4 x C2 action")

    schedule = json.loads((DIRECTORY / "baseline_schedule.json").read_text())
    verify_schedule(schedule, hx, hz)
    print("[7/7] depth-eight schedule covers every check edge exactly once")
    print("All package checks passed. Run verify_distance.py for the d=8 MILP.")


if __name__ == "__main__":
    main()
