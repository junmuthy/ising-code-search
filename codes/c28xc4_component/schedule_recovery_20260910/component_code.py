"""The exact-self-dual ``[[112,16,7]]`` C28 x C4 Ising component code."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

U_ORDER = 28
V_ORDER = 4
N = U_ORDER * V_ORDER
CHECK_RANK = 48


def polynomial(terms: Iterable[tuple[int, int]]) -> np.ndarray[Any, Any]:
    """Build a binary polynomial in F2[C28 x C4]."""
    values = np.zeros((U_ORDER, V_ORDER), dtype=np.uint8)
    for power_u, power_v in terms:
        values[power_u % U_ORDER, power_v % V_ORDER] ^= 1
    return values


SIMPLEX_SUPPORT = (0, 4, 8, 16)
SIMPLEX_DAGGER_SUPPORT = (0, 12, 20, 24)


def check_seed_polynomials() -> tuple[np.ndarray[Any, Any], ...]:
    """Return the three weight-16 orbit seeds that generate the check space.

    With ``s(u)=1+u^4+u^8+u^16``, these are

    ``s(u) (1+v+v^2+v^3)``,
    ``s(u) (1+u^7+u^14+u^21)``, and
    ``s(u)^dagger (1+u^21) (1+v)``.
    """
    first = polynomial(
        (power_u, power_v)
        for power_u in SIMPLEX_SUPPORT
        for power_v in range(V_ORDER)
    )
    second = polynomial(
        ((power_u + logical_shift) % U_ORDER, 0)
        for power_u in SIMPLEX_SUPPORT
        for logical_shift in (0, 7, 14, 21)
    )
    third = polynomial(
        ((power_u + logical_shift) % U_ORDER, power_v)
        for power_u in SIMPLEX_DAGGER_SUPPORT
        for logical_shift in (0, 21)
        for power_v in (0, 1)
    )
    return first, second, third


def translate(
    vector: np.ndarray[Any, Any], shift_u: int, shift_v: int
) -> np.ndarray[Any, Any]:
    source = np.asarray(vector, dtype=np.uint8).reshape(U_ORDER, V_ORDER)
    return np.roll(source, (shift_u, shift_v), axis=(0, 1))


def unique_translation_orbit(
    seed: np.ndarray[Any, Any],
) -> list[tuple[int, int, np.ndarray[Any, Any]]]:
    records: dict[bytes, tuple[int, int, np.ndarray[Any, Any]]] = {}
    for shift_u in range(U_ORDER):
        for shift_v in range(V_ORDER):
            row = translate(seed, shift_u, shift_v).reshape(-1)
            key = np.packbits(row, bitorder="little").tobytes()
            records.setdefault(key, (shift_u, shift_v, row))
    return list(records.values())


def gf2_rref(
    matrix: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    pivot_row = 0
    pivots: list[int] = []
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        work[[pivot_row, selected]] = work[[selected, pivot_row]]
        factors = np.flatnonzero(work[:, column])
        factors = factors[factors != pivot_row]
        work[factors] ^= work[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    return work[:pivot_row], tuple(pivots)


def translated_check_catalog(
) -> tuple[np.ndarray[Any, Any], list[tuple[int, int, int]], tuple[int, ...]]:
    rows = []
    metadata = []
    orbit_sizes = []
    for seed_index, seed in enumerate(check_seed_polynomials(), start=1):
        orbit = unique_translation_orbit(seed)
        orbit_sizes.append(len(orbit))
        for shift_u, shift_v, row in orbit:
            rows.append(row)
            metadata.append((seed_index, shift_u, shift_v))
    return np.asarray(rows, dtype=np.uint8), metadata, tuple(orbit_sizes)


def check_space_basis() -> np.ndarray[Any, Any]:
    rows, _metadata, _orbit_sizes = translated_check_catalog()
    return gf2_rref(rows)[0]


def logical_fibres() -> np.ndarray[Any, Any]:
    """Return the 16 disjoint weight-seven C4 x C4 logical fibres."""
    logicals = np.zeros((16, U_ORDER, V_ORDER), dtype=np.uint8)
    row = 0
    for logical_u in range(4):
        for logical_v in range(4):
            for thickness in range(7):
                logicals[row, logical_u + 4 * thickness, logical_v] = 1
            row += 1
    return logicals.reshape(16, N)


def independent_selection(
    rows: np.ndarray[Any, Any], order: Sequence[int], *, target_rank: int = CHECK_RANK
) -> list[int]:
    """Select original rows greedily while retaining binary independence."""
    row_integers = [
        int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        for row in rows
    ]
    pivots: dict[int, int] = {}
    selected = []
    for index in order:
        value = row_integers[index]
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                selected.append(int(index))
                break
        if len(selected) == target_rank:
            return selected
    return selected


def validate(check: np.ndarray[Any, Any]) -> dict[str, Any]:
    basis = gf2_rref(check)[0]
    logicals = logical_fibres()
    combined_rank = len(gf2_rref(np.vstack([basis, logicals]))[0])
    return {
        "n": N,
        "check_rank": len(basis),
        "k": N - 2 * len(basis),
        "self_orthogonal": bool(not np.any(basis @ basis.T % 2)),
        "exact_zx_self_dual": True,
        "doubly_even_check_space": bool(
            not np.any(basis @ basis.T % 2)
            and np.all(basis.sum(axis=1) % 4 == 0)
        ),
        "logical_count": len(logicals),
        "logical_weights": sorted(set(map(int, logicals.sum(axis=1)))),
        "logical_gram_identity": bool(
            np.array_equal(logicals @ logicals.T % 2, np.eye(16, dtype=np.uint8))
        ),
        "logicals_commute_with_checks": bool(not np.any(basis @ logicals.T % 2)),
        "logicals_independent_mod_checks": combined_rank - len(basis),
    }


def qldpc_code(check: np.ndarray[Any, Any] | None = None):
    """Construct this code as a multistaq qLDPC ``CSSCode``."""
    from qldpc.codes import CSSCode

    matrix = check_space_basis() if check is None else np.asarray(check, dtype=int)
    return CSSCode(matrix, matrix, promise_equal_distance_xz=True)
