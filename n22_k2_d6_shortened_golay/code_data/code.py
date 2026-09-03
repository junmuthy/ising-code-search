"""Cyclic weight-eight presentation of the ``[[22,2,6]]`` shortened Golay code.

The presentation is the MCR/generalized-bicycle instance

    ell = 11, f = 1 + x, p = 1,
    q = x + x^3 + x^4 + x^5 + x^9.

Writing a = p f and b = q f, the CSS matrices are

    H_X = [circ(a) | circ(b)]
    H_Z = [circ(b)^T | circ(a)^T].

Rows and cyclic coordinates are numbered from zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


ELL = 11
NUM_DATA_QUBITS = 2 * ELL
NUM_CHECKS_PER_TYPE = ELL
NUM_LOGICALS = 2
STATIC_DISTANCE = 6

F_SUPPORT = (0, 1)
P_SUPPORT = (0,)
Q_SUPPORT = (1, 3, 4, 5, 9)

# Smallest simultaneously disjoint Z representatives in this cyclic presentation.
LOGICAL_Z_SUPPORTS = (
    (0, 1, 3, 4, 5, 9, 11),
    (2, 7, 8, 12, 15, 16, 17),
)

# Symplectic X partners for LOGICAL_Z_SUPPORTS.  The first happens to have the
# same physical support as Z_0; X_1 is a different weight-seven representative.
LOGICAL_X_SUPPORTS = (
    (0, 1, 3, 4, 5, 9, 11),
    (1, 4, 5, 7, 8, 11, 12),
)


def cyclic_product(left: Iterable[int], right: Iterable[int]) -> tuple[int, ...]:
    """Multiply binary polynomials modulo ``x**ELL - 1``."""

    support: set[int] = set()
    for aa in left:
        for bb in right:
            exponent = (int(aa) + int(bb)) % ELL
            if exponent in support:
                support.remove(exponent)
            else:
                support.add(exponent)
    return tuple(sorted(support))


A_SUPPORT = cyclic_product(P_SUPPORT, F_SUPPORT)
B_SUPPORT = cyclic_product(Q_SUPPORT, F_SUPPORT)


def support_matrix(supports: Sequence[Sequence[int]], width: int = NUM_DATA_QUBITS) -> np.ndarray:
    matrix = np.zeros((len(supports), width), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, list(support)] = 1
    return matrix


def build_checks() -> tuple[np.ndarray, np.ndarray]:
    checks_x = np.zeros((ELL, 2 * ELL), dtype=np.uint8)
    checks_z = np.zeros_like(checks_x)
    for check in range(ELL):
        for exponent in A_SUPPORT:
            checks_x[check, (check + exponent) % ELL] = 1
            checks_z[check, ELL + (check - exponent) % ELL] = 1
        for exponent in B_SUPPORT:
            checks_x[check, ELL + (check + exponent) % ELL] = 1
            checks_z[check, (check - exponent) % ELL] = 1
    return checks_x, checks_z


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy()
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    work = np.asarray(matrix, dtype=np.uint8).copy()
    rows, columns = work.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        work[[pivot_row, selected]] = work[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row, column]:
                work[row] ^= work[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = np.zeros(columns, dtype=np.uint8)
        vector[free] = 1
        for row, pivot in reversed(list(enumerate(pivot_columns))):
            vector[pivot] = int(np.dot(work[row], vector) % 2)
        basis.append(vector)
    return np.asarray(basis, dtype=np.uint8)


def enumerate_span(rows: np.ndarray) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    rank = 0
    for row in np.asarray(rows, dtype=np.uint8):
        candidate = np.asarray([*basis, row], dtype=np.uint8)
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            basis.append(row.copy())
            rank = candidate_rank
    output: list[np.ndarray] = []
    for mask in range(1 << len(basis)):
        word = np.zeros(rows.shape[1], dtype=np.uint8)
        for index, row in enumerate(basis):
            if mask >> index & 1:
                word ^= row
        output.append(word)
    return output


def fold_permutation() -> tuple[int, ...]:
    """Universal BB ZX fold: ``L_i <-> R_{-i}``."""

    return tuple(
        ELL + (-qubit) % ELL
        if qubit < ELL
        else (-(qubit - ELL)) % ELL
        for qubit in range(2 * ELL)
    )


def fold_check_action() -> tuple[int, ...]:
    """X-check index mapped to its Z-check index by the fold."""

    return tuple((-check) % ELL for check in range(ELL))


def permute_vector(vector: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(vector)
    output[list(permutation)] = vector
    return output


def validate_code(*, exhaustive: bool = True) -> dict[str, object]:
    checks_x, checks_z = build_checks()
    logical_x = support_matrix(LOGICAL_X_SUPPORTS)
    logical_z = support_matrix(LOGICAL_Z_SUPPORTS)
    rank_x = gf2_rank(checks_x)
    rank_z = gf2_rank(checks_z)
    if np.any((checks_x @ checks_z.T) % 2):
        raise AssertionError("CSS commutation failure")
    if rank_x != 10 or rank_z != 10:
        raise AssertionError("wrong stabilizer rank")
    if not np.array_equal((logical_x @ logical_z.T) % 2, np.eye(2, dtype=np.uint8)):
        raise AssertionError("logical basis is not symplectic")
    if np.any((checks_z @ logical_x.T) % 2) or np.any((checks_x @ logical_z.T) % 2):
        raise AssertionError("logical does not commute with a stabilizer")
    if set(LOGICAL_Z_SUPPORTS[0]) & set(LOGICAL_Z_SUPPORTS[1]):
        raise AssertionError("Z logical supports are not disjoint")
    permutation = fold_permutation()
    check_action = fold_check_action()
    folded = np.zeros_like(checks_x)
    folded[:, list(permutation)] = checks_x
    if not np.array_equal(folded, checks_z[list(check_action)]):
        raise AssertionError("claimed ZX fold does not map H_X to H_Z")

    distance_x: int | None = None
    distance_z: int | None = None
    disjoint_minimum: tuple[int, int] | None = None
    if exhaustive:
        span_x = {row.tobytes() for row in enumerate_span(checks_x)}
        span_z = {row.tobytes() for row in enumerate_span(checks_z)}
        kernel_x = enumerate_span(gf2_nullspace(checks_z))
        kernel_z = enumerate_span(gf2_nullspace(checks_x))
        x_logicals = [word for word in kernel_x if word.tobytes() not in span_x]
        z_logicals = [word for word in kernel_z if word.tobytes() not in span_z]
        distance_x = min(int(np.sum(word)) for word in x_logicals)
        distance_z = min(int(np.sum(word)) for word in z_logicals)
        for total in range(2 * STATIC_DISTANCE, 2 * NUM_DATA_QUBITS + 1):
            found = False
            for left in z_logicals:
                left_weight = int(np.sum(left))
                right_weight = total - left_weight
                if right_weight < STATIC_DISTANCE:
                    continue
                for right in z_logicals:
                    if int(np.sum(right)) != right_weight or np.any(left & right):
                        continue
                    if (left ^ right).tobytes() in span_z:
                        continue
                    disjoint_minimum = (left_weight, right_weight)
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if distance_x != STATIC_DISTANCE or distance_z != STATIC_DISTANCE:
            raise AssertionError("wrong exact CSS distance")
        if disjoint_minimum != (7, 7):
            raise AssertionError(f"unexpected disjoint minimum: {disjoint_minimum}")

    row_weights_x = tuple(int(value) for value in checks_x.sum(axis=1))
    row_weights_z = tuple(int(value) for value in checks_z.sum(axis=1))
    degree_x = tuple(int(value) for value in checks_x.sum(axis=0))
    degree_z = tuple(int(value) for value in checks_z.sum(axis=0))
    return {
        "parameters": "[[22,2,6]]",
        "ell": ELL,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "row_weights_x": row_weights_x,
        "row_weights_z": row_weights_z,
        "data_degrees_x": degree_x,
        "data_degrees_z": degree_z,
        "combined_data_degrees": tuple(xx + zz for xx, zz in zip(degree_x, degree_z, strict=True)),
        "distance_x": distance_x,
        "distance_z": distance_z,
        "disjoint_z_logical_weights": tuple(len(row) for row in LOGICAL_Z_SUPPORTS),
        "minimum_disjoint_z_logical_weights": disjoint_minimum,
        "zx_fold": permutation,
        "zx_fold_check_action": check_action,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_code(), indent=2))

