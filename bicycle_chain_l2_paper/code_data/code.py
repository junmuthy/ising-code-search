"""Exact ``ell=2, m=7`` bicycle-chain specialization from Ismail et al.

The paper works in ``F_2[x,y]/(x**ell-1, y**m-1)`` and fixes

    a = 1 + y**3 + x*y**2 + x*y**4,
    b = a^dagger.

With ``A`` and ``B`` the corresponding block circulants,

    H_X = [A | B],        H_Z = [B.T | A.T].

Because ``b=a^dagger``, ``B=A.T`` and the displayed checks are exactly
self-dual.  The paper labels the ``m=7`` family ``[[14*ell,2*ell,6]]``.
Exact enumeration at ``ell=2`` instead gives ``[[28,4,5]]``; this module
deliberately freezes and tests that literal specialization.

Data labels use ``q(half,i,j) = half*ell*m + i*m + j`` with
``half=0`` for L and ``half=1`` for R.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ELL = 2
M = 7
CELL_COUNT = ELL * M
NUM_DATA_QUBITS = 2 * CELL_COUNT
NUM_CHECKS_PER_TYPE = CELL_COUNT
NUM_LOGICALS = 2 * ELL
STATIC_DISTANCE = 5
CHECK_WEIGHT = 8

# Monomial exponents (x exponent, y exponent), reduced modulo (ell,m).
A_SUPPORT = ((0, 0), (0, 3), (1, 2), (1, 4))
B_SUPPORT = tuple(sorted(((-ii) % ELL, (-jj) % M) for ii, jj in A_SUPPORT))


def cell_index(ii: int, jj: int) -> int:
    """Return the flattened torus-cell index."""

    return (int(ii) % ELL) * M + (int(jj) % M)


def data_index(half: int, ii: int, jj: int) -> int:
    """Return a physical data-qubit index in the L/R convention."""

    if half not in (0, 1):
        raise ValueError("half must be 0 (L) or 1 (R)")
    return half * CELL_COUNT + cell_index(ii, jj)


# Complete disjoint logical basis: one odd-weight circumference fibre for
# each x coordinate on each physical half.  The X and Z representatives are
# identical, so transversal physical H implements logical H exactly.
LOGICAL_Z_SUPPORTS = tuple(
    tuple(data_index(half, ii, jj) for jj in range(M))
    for half in range(2)
    for ii in range(ELL)
)
LOGICAL_X_SUPPORTS = LOGICAL_Z_SUPPORTS


def support_matrix(
    supports: Sequence[Sequence[int]], width: int = NUM_DATA_QUBITS
) -> np.ndarray:
    matrix = np.zeros((len(supports), width), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, list(support)] = 1
    return matrix


def build_checks() -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical BB checks using the paper's displacement rule."""

    checks_x = np.zeros((CELL_COUNT, NUM_DATA_QUBITS), dtype=np.uint8)
    checks_z = np.zeros_like(checks_x)
    for ii in range(ELL):
        for jj in range(M):
            check = cell_index(ii, jj)
            for di, dj in A_SUPPORT:
                checks_x[check, data_index(0, ii + di, jj + dj)] ^= 1
                # B.T has support A.
                checks_z[check, data_index(0, ii + di, jj + dj)] ^= 1
            for di, dj in B_SUPPORT:
                checks_x[check, data_index(1, ii + di, jj + dj)] ^= 1
                # A.T has support B.
                checks_z[check, data_index(1, ii + di, jj + dj)] ^= 1
    return checks_x, checks_z


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    work = np.asarray(matrix, dtype=np.uint8).copy()
    pivots: list[int] = []
    rank = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        pivots.append(column)
        rank += 1
        if rank == work.shape[0]:
            break
    return work, tuple(pivots)


def gf2_rank(matrix: np.ndarray) -> int:
    return len(gf2_rref(matrix)[1])


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    work, pivots = gf2_rref(matrix)
    columns = work.shape[1]
    free_columns = [column for column in range(columns) if column not in pivots]
    basis = []
    for free in free_columns:
        vector = np.zeros(columns, dtype=np.uint8)
        vector[free] = 1
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = int(np.dot(work[row], vector) % 2)
        basis.append(vector)
    return np.asarray(basis, dtype=np.uint8)


def independent_rows(matrix: np.ndarray) -> np.ndarray:
    """Return an independent subset spanning the input row space."""

    basis: list[np.ndarray] = []
    rank = 0
    for row in np.asarray(matrix, dtype=np.uint8):
        candidate = np.asarray([*basis, row], dtype=np.uint8)
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            basis.append(row.copy())
            rank = candidate_rank
    return np.asarray(basis, dtype=np.uint8)


def enumerate_span(rows: np.ndarray) -> list[np.ndarray]:
    basis = independent_rows(rows)
    output: list[np.ndarray] = []
    for mask in range(1 << len(basis)):
        word = np.zeros(rows.shape[1], dtype=np.uint8)
        for index, row in enumerate(basis):
            if mask >> index & 1:
                word ^= row
        output.append(word)
    return output


def redundant_relations(matrix: np.ndarray | None = None) -> tuple[tuple[int, ...], ...]:
    """Return a basis of displayed-check dependencies."""

    checks = build_checks()[0] if matrix is None else np.asarray(matrix, dtype=np.uint8)
    basis = gf2_nullspace(checks.T)
    return tuple(tuple(np.flatnonzero(row).astype(int).tolist()) for row in basis)


def fold_permutation() -> tuple[int, ...]:
    """Physical ZX fold; self-duality makes it the identity."""

    return tuple(range(NUM_DATA_QUBITS))


def fold_check_action() -> tuple[int, ...]:
    """X-check index mapped to its identical Z-check index."""

    return tuple(range(NUM_CHECKS_PER_TYPE))


def translation_x_permutation() -> tuple[int, ...]:
    """Physical x translation inducing two independent logical C2 swaps."""

    return tuple(
        data_index(half, ii + 1, jj)
        for half in range(2)
        for ii in range(ELL)
        for jj in range(M)
    )


def permute_vector(vector: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(vector)
    output[list(permutation)] = vector
    return output


def _exact_css_distance(checks: np.ndarray) -> tuple[int, tuple[int, ...]]:
    stabilizers = {row.tobytes() for row in enumerate_span(checks)}
    logicals = [
        row
        for row in enumerate_span(gf2_nullspace(checks))
        if row.tobytes() not in stabilizers
    ]
    witness = min(logicals, key=lambda row: int(np.sum(row)))
    return int(np.sum(witness)), tuple(np.flatnonzero(witness).astype(int).tolist())


def validate_code(*, exhaustive: bool = True) -> dict[str, object]:
    checks_x, checks_z = build_checks()
    logical_x = support_matrix(LOGICAL_X_SUPPORTS)
    logical_z = support_matrix(LOGICAL_Z_SUPPORTS)
    rank_x = gf2_rank(checks_x)
    rank_z = gf2_rank(checks_z)
    if not np.array_equal(checks_x, checks_z):
        raise AssertionError("paper specialization is not exactly self-dual")
    if np.any((checks_x @ checks_z.T) % 2):
        raise AssertionError("CSS commutation failure")
    if rank_x != 12 or rank_z != 12:
        raise AssertionError("wrong stabilizer rank")
    if NUM_DATA_QUBITS - rank_x - rank_z != NUM_LOGICALS:
        raise AssertionError("wrong encoded dimension")
    if not np.array_equal(
        (logical_x @ logical_z.T) % 2, np.eye(NUM_LOGICALS, dtype=np.uint8)
    ):
        raise AssertionError("logical basis is not symplectic")
    if np.any((checks_z @ logical_x.T) % 2) or np.any((checks_x @ logical_z.T) % 2):
        raise AssertionError("logical does not commute with a stabilizer")
    for left in range(NUM_LOGICALS):
        for right in range(left):
            if set(LOGICAL_Z_SUPPORTS[left]) & set(LOGICAL_Z_SUPPORTS[right]):
                raise AssertionError("Z logical supports are not disjoint")

    translation = translation_x_permutation()
    expected_action = (1, 0, 3, 2)
    translation_action = []
    for logical in logical_z:
        moved = permute_vector(logical, translation)
        matches = [index for index, row in enumerate(logical_z) if np.array_equal(moved, row)]
        if len(matches) != 1:
            raise AssertionError("x translation does not preserve the logical basis")
        translation_action.append(matches[0])
    if tuple(translation_action) != expected_action:
        raise AssertionError("unexpected logical C2 action")

    distance_x: int | None = None
    distance_z: int | None = None
    witness_x: tuple[int, ...] | None = None
    witness_z: tuple[int, ...] | None = None
    if exhaustive:
        distance_x, witness_x = _exact_css_distance(checks_z)
        distance_z, witness_z = _exact_css_distance(checks_x)
        if distance_x != STATIC_DISTANCE or distance_z != STATIC_DISTANCE:
            raise AssertionError("wrong exact CSS distance")

    row_weights_x = tuple(int(value) for value in checks_x.sum(axis=1))
    row_weights_z = tuple(int(value) for value in checks_z.sum(axis=1))
    degree_x = tuple(int(value) for value in checks_x.sum(axis=0))
    degree_z = tuple(int(value) for value in checks_z.sum(axis=0))
    if set(row_weights_x + row_weights_z) != {CHECK_WEIGHT}:
        raise AssertionError("unexpected check weight")
    return {
        "parameters": "[[28,4,5]]",
        "paper_family_label": "[[14*ell,2*ell,6]]",
        "ell": ELL,
        "m": M,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "row_weights_x": row_weights_x,
        "row_weights_z": row_weights_z,
        "data_degrees_x": degree_x,
        "data_degrees_z": degree_z,
        "combined_data_degrees": tuple(xx + zz for xx, zz in zip(degree_x, degree_z, strict=True)),
        "distance_x": distance_x,
        "distance_z": distance_z,
        "minimum_logical_witness_x": witness_x,
        "minimum_logical_witness_z": witness_z,
        "disjoint_z_logical_weights": tuple(len(row) for row in LOGICAL_Z_SUPPORTS),
        "zx_fold": fold_permutation(),
        "zx_fold_check_action": fold_check_action(),
        "translation_x": translation,
        "translation_x_logical_action": tuple(translation_action),
        "redundant_relations": redundant_relations(checks_x),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_code(), indent=2))
