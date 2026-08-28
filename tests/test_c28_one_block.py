"""Regression tests for the one-block weakly self-dual ``C28`` search."""

from __future__ import annotations

import numpy as np

from c28_one_block.search import (
    ORDER,
    circulant_matrix,
    circulant_rank_mask,
    is_self_orthogonal_mask,
    logical_fibres,
    rotate_mask,
)


def test_nilpotent_example_is_self_orthogonal() -> None:
    mask = 1 | (1 << 14)
    matrix = circulant_matrix(mask)
    assert is_self_orthogonal_mask(mask)
    assert not np.any((matrix @ matrix.T) % 2)


def test_packed_rank_matches_binary_matrix_rank() -> None:
    mask = 1 | (1 << 1) | (1 << 7) | (1 << 12)
    matrix = circulant_matrix(mask)
    reduced = matrix.copy()
    rank = 0
    for column in range(ORDER):
        pivots = np.flatnonzero(reduced[rank:, column])
        if len(pivots) == 0:
            continue
        pivot = rank + int(pivots[0])
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in range(ORDER):
            if row != rank and reduced[row, column]:
                reduced[row] ^= reduced[rank]
        rank += 1
    assert circulant_rank_mask(mask) == rank


def test_four_weight_seven_fibres_are_disjoint_c4_orbit() -> None:
    fibres = logical_fibres()
    assert np.all(np.count_nonzero(fibres, axis=1) == 7)
    assert np.max(np.sum(fibres, axis=0)) == 1
    first_mask = sum(1 << int(index) for index in np.flatnonzero(fibres[0]))
    for logical in range(4):
        translated = rotate_mask(first_mask, 7 * logical)
        expected = sum(
            1 << int(index) for index in np.flatnonzero(fibres[logical])
        )
        assert translated == expected
    assert np.array_equal((fibres @ fibres.T) % 2, np.eye(4, dtype=np.uint8))
