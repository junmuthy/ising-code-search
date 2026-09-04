"""Exact ``ell=4, m=7`` bicycle-chain code from Ismail et al.

The code is the self-dual BB construction over
``F_2[x,y]/(x**4-1,y**7-1)`` with

    a = 1 + y**3 + x*y**2 + x*y**4,
    b = a**dagger.

Data labels use ``q(half,i,j) = half*ell*m + i*m + j``, with ``half=0``
for L and ``half=1`` for R.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bicycle_chain_l2_paper.tmr_postselection.model import gf2_rank, row_relations


ELL = 4
M = 7
CELL_COUNT = ELL * M
NUM_DATA_QUBITS = 2 * CELL_COUNT
NUM_CHECKS_PER_TYPE = CELL_COUNT
NUM_LOGICALS = 2 * ELL
STATIC_DISTANCE = 6
CHECK_WEIGHT = 8

A_SUPPORT = ((0, 0), (0, 3), (1, 2), (1, 4))
B_SUPPORT = tuple(sorted(((-ii) % ELL, (-jj) % M) for ii, jj in A_SUPPORT))


def cell_index(ii: int, jj: int) -> int:
    return (int(ii) % ELL) * M + (int(jj) % M)


def data_index(half: int, ii: int, jj: int) -> int:
    if half not in (0, 1):
        raise ValueError("half must be 0 (L) or 1 (R)")
    return half * CELL_COUNT + cell_index(ii, jj)


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
    checks = np.zeros((CELL_COUNT, NUM_DATA_QUBITS), dtype=np.uint8)
    for ii in range(ELL):
        for jj in range(M):
            check = cell_index(ii, jj)
            for di, dj in A_SUPPORT:
                checks[check, data_index(0, ii + di, jj + dj)] ^= 1
            for di, dj in B_SUPPORT:
                checks[check, data_index(1, ii + di, jj + dj)] ^= 1
    return checks, checks.copy()


def redundant_relations(
    matrix: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    checks = build_checks()[0] if matrix is None else np.asarray(matrix, dtype=np.uint8)
    return row_relations(checks)


def fold_permutation() -> tuple[int, ...]:
    return tuple(range(NUM_DATA_QUBITS))


def fold_check_action() -> tuple[int, ...]:
    return tuple(range(NUM_CHECKS_PER_TYPE))


def translation_x_permutation() -> tuple[int, ...]:
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


def _distance_certificate() -> dict[str, object]:
    """Certify one translated logical class on each half with HiGHS."""

    from gala_search.abelian_fibre_codes import _solve_minimum_weight_parity_problem

    checks = build_checks()[0]
    logicals = support_matrix(LOGICAL_Z_SUPPORTS)
    classes = []
    for logical_index in (0, ELL):
        effective = np.vstack((checks, logicals[logical_index]))
        syndrome = np.zeros(len(effective), dtype=int)
        syndrome[-1] = 1
        weight, support, status, seconds = _solve_minimum_weight_parity_problem(
            effective, syndrome, solver="HIGHS"
        )
        classes.append(
            {
                "logical_index": logical_index,
                "minimum_weight": int(weight),
                "support": list(map(int, support)),
                "solver_status": status,
                "seconds": round(float(seconds), 6),
            }
        )
    return {
        "method": "exact minimum-weight parity MILP on one translation class per half",
        "classes": classes,
        "distance": min(item["minimum_weight"] for item in classes),
    }


def validate_code(*, exhaustive: bool = False) -> dict[str, object]:
    checks_x, checks_z = build_checks()
    logical_x = support_matrix(LOGICAL_X_SUPPORTS)
    logical_z = support_matrix(LOGICAL_Z_SUPPORTS)
    rank_x = gf2_rank(checks_x)
    rank_z = gf2_rank(checks_z)
    if not np.array_equal(checks_x, checks_z):
        raise AssertionError("paper code is not exactly self-dual")
    if np.any((checks_x @ checks_z.T) % 2):
        raise AssertionError("CSS commutation failure")
    if (rank_x, rank_z) != (24, 24):
        raise AssertionError("wrong stabilizer ranks")
    if NUM_DATA_QUBITS - rank_x - rank_z != NUM_LOGICALS:
        raise AssertionError("wrong encoded dimension")
    if not np.array_equal(
        (logical_x @ logical_z.T) % 2, np.eye(NUM_LOGICALS, dtype=np.uint8)
    ):
        raise AssertionError("logical basis is not symplectic")
    if np.any((checks_x @ logical_z.T) % 2):
        raise AssertionError("logical does not commute with a stabilizer")
    union: set[int] = set()
    for support in LOGICAL_Z_SUPPORTS:
        if union.intersection(support):
            raise AssertionError("logical supports overlap")
        union.update(support)
    if len(union) != NUM_DATA_QUBITS:
        raise AssertionError("logical fibres do not partition all data qubits")

    permutation = translation_x_permutation()
    action = []
    for logical in logical_z:
        moved = permute_vector(logical, permutation)
        matches = [
            index for index, row in enumerate(logical_z) if np.array_equal(moved, row)
        ]
        if len(matches) != 1:
            raise AssertionError("translation does not preserve the logical basis")
        action.append(matches[0])
    expected_action = (1, 2, 3, 0, 5, 6, 7, 4)
    if tuple(action) != expected_action:
        raise AssertionError("wrong logical C4 action")

    distance = _distance_certificate() if exhaustive else None
    if distance is not None and distance["distance"] != STATIC_DISTANCE:
        raise AssertionError("wrong exact distance")
    return {
        "parameters": "[[56,8,6]]",
        "ell": ELL,
        "m": M,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight": CHECK_WEIGHT,
        "data_degree_per_type": 4,
        "disjoint_logical_weights": tuple(map(len, LOGICAL_Z_SUPPORTS)),
        "translation_x_logical_action": tuple(action),
        "redundant_relations": redundant_relations(checks_x),
        "distance_certificate": distance,
    }
