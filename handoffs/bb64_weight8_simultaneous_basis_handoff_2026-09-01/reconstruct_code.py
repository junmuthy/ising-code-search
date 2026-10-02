#!/usr/bin/env python3
"""Reconstruct the Liang--Chen self-dual [[64,8,8]] BB check matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ELEMENTS = tuple((xx, yy) for xx in range(4) for yy in range(8))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}


def add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    """Multiply monomials subject to x^4 y^4=1 and y^8=1."""
    raw_x = left[0] + right[0]
    return raw_x % 4, (left[1] + right[1] + 4 * (raw_x // 4)) % 8


def translation(element: tuple[int, int]) -> np.ndarray:
    return np.asarray([INDEX[add(element, source)] for source in ELEMENTS], dtype=np.int64)


def permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    output = np.zeros((len(permutation), len(permutation)), dtype=np.uint8)
    output[np.arange(len(permutation)), permutation] = 1
    return output


def construct_checks() -> tuple[np.ndarray, np.ndarray]:
    shift_x = permutation_matrix(translation((1, 0)))
    shift_y = permutation_matrix(translation((0, 1)))
    identity = np.eye(32, dtype=np.uint8)
    # f=1+x+y+y^-1 and g=f^dagger, so H_X=H_Z=[F|F^T].
    polynomial_matrix = (identity + shift_x + shift_y + shift_y.T) % 2
    check = np.hstack([polynomial_matrix, polynomial_matrix.T])
    return check, check.copy()


if __name__ == "__main__":
    matrix_x, matrix_z = construct_checks()
    archive = np.load(Path(__file__).with_name("bb64_weight8_basis.npz"))
    assert np.array_equal(matrix_x, archive["matrix_x"])
    assert np.array_equal(matrix_z, archive["matrix_z"])
    print("Reconstruction matches bb64_weight8_basis.npz exactly.")
