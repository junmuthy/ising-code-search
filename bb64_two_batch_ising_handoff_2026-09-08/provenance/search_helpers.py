"""Small GF(2) helpers shared by the standalone search and document generator."""

from __future__ import annotations

from typing import Sequence

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


def independent_rows(matrix: np.ndarray) -> np.ndarray:
    selected: list[np.ndarray] = []
    rank = 0
    for row in np.asarray(matrix, dtype=np.uint8):
        candidate = np.asarray([*selected, row], dtype=np.uint8)
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            selected.append(row)
            rank = candidate_rank
    return np.asarray(selected, dtype=np.uint8)


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def permutation_order(permutation: np.ndarray) -> int:
    identity = np.arange(len(permutation), dtype=np.int64)
    power = identity.copy()
    for exponent in range(1, len(permutation) + 1):
        power = permutation[power]
        if np.array_equal(power, identity):
            return exponent
    raise ValueError("invalid permutation")


def permutation_matrix_test(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=np.uint8)
    return bool(np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1))


def batch_disjointness(
    logicals: np.ndarray, partition: Sequence[Sequence[int]]
) -> bool:
    return all(
        np.all(logicals[np.asarray(batch, dtype=np.int64)].sum(axis=0) <= 1)
        for batch in partition
    )
