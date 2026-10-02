"""Exhaustive algebra and certification utilities for the ``C28`` search."""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter
from typing import Any

import numpy as np
from numba import njit, types
from numba.typed import List

ORDER = 28
LOGICAL_ORDER = 4
THICKNESS = 7
TARGET_RANK = 12
TARGET_K = 4
MASK = (1 << ORDER) - 1
SEARCH_WEIGHTS = (6, 8, 10, 12)


@njit(cache=True)
def rotate_mask(mask: int, shift: int) -> int:
    shift %= ORDER
    if shift == 0:
        return mask & MASK
    return ((mask << shift) | (mask >> (ORDER - shift))) & MASK


@njit(cache=True)
def parity(mask: int) -> int:
    mask ^= mask >> 16
    mask ^= mask >> 8
    mask ^= mask >> 4
    mask ^= mask >> 2
    mask ^= mask >> 1
    return mask & 1


@njit(cache=True)
def invert_mask(mask: int) -> int:
    output = 0
    for exponent in range(ORDER):
        if (mask >> exponent) & 1:
            output |= 1 << ((-exponent) % ORDER)
    return output


@njit(cache=True)
def cyclic_canonical_mask(mask: int) -> int:
    inverse = invert_mask(mask)
    best = mask
    for shift in range(ORDER):
        best = min(best, rotate_mask(mask, shift), rotate_mask(inverse, shift))
    return best


@njit(cache=True)
def is_self_orthogonal_mask(mask: int) -> bool:
    """Return whether the circulant represented by ``mask`` squares to zero.

    Equivalently, every cyclic autocorrelation of the support is even. Shifts
    ``s`` and ``28-s`` agree, so only zero through fourteen are needed.
    """
    for shift in range(15):
        if parity(mask & rotate_mask(mask, shift)):
            return False
    return True


@njit(cache=True)
def circulant_rank_mask(mask: int) -> int:
    basis = np.zeros(ORDER, dtype=np.uint64)
    rank = 0
    for shift in range(ORDER):
        row = rotate_mask(mask, shift)
        while row:
            pivot = -1
            for column in range(ORDER - 1, -1, -1):
                if (row >> column) & 1:
                    pivot = column
                    break
            if basis[pivot] == 0:
                basis[pivot] = row
                rank += 1
                break
            row ^= int(basis[pivot])
    return rank


@njit(cache=True)
def _sector_masks() -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((LOGICAL_ORDER, 64), dtype=np.uint64)
    weights = np.zeros(64, dtype=np.uint8)
    for free in range(64):
        sector_weight = 0
        for local in range(6):
            if (free >> local) & 1:
                sector_weight += 1
        weights[free] = sector_weight + (sector_weight & 1)
        for sector in range(LOGICAL_ORDER):
            value = 0
            for local in range(6):
                if (free >> local) & 1:
                    value |= 1 << (sector + LOGICAL_ORDER * local)
            if sector_weight & 1:
                value |= 1 << (sector + LOGICAL_ORDER * 6)
            masks[sector, free] = value
    return masks, weights


@njit(cache=True)
def enumerate_structural_masks() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exhaust all ``2^24`` even-sector polynomials and retain rank twelve."""
    sector_masks, sector_weights = _sector_masks()
    retained = List.empty_list(types.uint64)
    considered = np.zeros(13, dtype=np.int64)
    self_orthogonal = np.zeros(13, dtype=np.int64)
    rank_twelve = np.zeros(13, dtype=np.int64)
    canonical = np.zeros(13, dtype=np.int64)
    for encoded in range(1 << 24):
        indices = (
            encoded & 63,
            (encoded >> 6) & 63,
            (encoded >> 12) & 63,
            (encoded >> 18) & 63,
        )
        weight = (
            int(sector_weights[indices[0]])
            + int(sector_weights[indices[1]])
            + int(sector_weights[indices[2]])
            + int(sector_weights[indices[3]])
        )
        if weight not in SEARCH_WEIGHTS:
            continue
        considered[weight] += 1
        mask = (
            int(sector_masks[0, indices[0]])
            | int(sector_masks[1, indices[1]])
            | int(sector_masks[2, indices[2]])
            | int(sector_masks[3, indices[3]])
        )
        if not is_self_orthogonal_mask(mask):
            continue
        self_orthogonal[weight] += 1
        if circulant_rank_mask(mask) != TARGET_RANK:
            continue
        rank_twelve[weight] += 1
        if cyclic_canonical_mask(mask) != mask:
            continue
        canonical[weight] += 1
        retained.append(mask)
    return retained, considered, self_orthogonal, rank_twelve, canonical


def mask_support(mask: int) -> list[int]:
    return [exponent for exponent in range(ORDER) if (mask >> exponent) & 1]


def mask_product_coordinates(mask: int) -> list[dict[str, int]]:
    """Serialize exponents as ``u^i v^j`` for ``C4 x C7``.

    Under the CRT identification, ``u=x^7`` and ``v=x^4``.
    """
    coordinates = []
    for exponent in mask_support(mask):
        logical = (-exponent) % LOGICAL_ORDER
        thickness = ((exponent - 7 * logical) // 4) % THICKNESS
        coordinates.append(
            {"exponent": exponent, "logical": logical, "thickness": thickness}
        )
    return coordinates


def circulant_matrix(mask: int) -> np.ndarray:
    matrix = np.zeros((ORDER, ORDER), dtype=np.uint8)
    for row in range(ORDER):
        shifted = int(rotate_mask(mask, row))
        matrix[row, mask_support(shifted)] = 1
    return matrix


def logical_fibres() -> np.ndarray:
    fibres = np.zeros((LOGICAL_ORDER, ORDER), dtype=np.uint8)
    for logical in range(LOGICAL_ORDER):
        for thickness in range(THICKNESS):
            fibres[logical, (7 * logical + 4 * thickness) % ORDER] = 1
    return fibres


def packed_row_basis(matrix: np.ndarray) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = sum(int(bit) << index for index, bit in enumerate(row))
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
    return basis


def packed_in_span(value: int, basis: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def find_logical_below_six(matrix: np.ndarray) -> dict[str, Any] | None:
    column_syndromes = [
        sum(int(matrix[row, column]) << row for row in range(ORDER))
        for column in range(ORDER)
    ]
    stabilizer_basis = packed_row_basis(matrix)
    for weight in range(1, 6):
        for support in itertools.combinations(range(ORDER), weight):
            syndrome = 0
            vector = 0
            for qubit in support:
                syndrome ^= column_syndromes[qubit]
                vector |= 1 << qubit
            if syndrome == 0 and not packed_in_span(vector, stabilizer_basis):
                return {"weight": weight, "support": list(support)}
    return None


def tanner_connected(matrix: np.ndarray) -> bool:
    num_checks, num_qubits = matrix.shape
    adjacency = [set() for _ in range(num_checks + num_qubits)]
    rows, columns = np.nonzero(matrix)
    for check, qubit in zip(rows, columns, strict=True):
        adjacency[int(check)].add(num_checks + int(qubit))
        adjacency[num_checks + int(qubit)].add(int(check))
    reached = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    return len(reached) == num_checks + num_qubits


def analyze_mask(mask: int, *, certify_distance: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    matrix = circulant_matrix(mask)
    fibres = logical_fibres()
    rank = int(circulant_rank_mask(mask))
    pairing = (fibres @ fibres.T) % 2
    logical_rank = int(np.linalg.matrix_rank(pairing.astype(float)))
    low_logical = find_logical_below_six(matrix) if certify_distance else None
    return {
        "mask": int(mask),
        "support": mask_support(mask),
        "terms": mask_product_coordinates(mask),
        "check_weight": int(mask.bit_count()),
        "self_orthogonal": bool(not np.any((matrix @ matrix.T) % 2)),
        "rank": rank,
        "k": ORDER - 2 * rank,
        "tanner_connected": tanner_connected(matrix),
        "logical_fibres": [
            np.flatnonzero(fibre).astype(int).tolist() for fibre in fibres
        ],
        "logical_weights": np.count_nonzero(fibres, axis=1).astype(int).tolist(),
        "logical_fibres_in_kernel": bool(not np.any((matrix @ fibres.T) % 2)),
        "logical_pairing": pairing.astype(int).tolist(),
        "logical_pairing_rank": logical_rank,
        "logical_translation_exponent": 7,
        "logical_translation_is_c4": True,
        "logical_supports_pairwise_disjoint": bool(
            np.max(np.sum(fibres, axis=0), initial=0) <= 1
        ),
        "low_logical_below_six": low_logical,
        "certified_distance_at_least_six": certify_distance and low_logical is None,
        "distance_upper_bound": 7,
        "supports_checked": (
            sum(math.comb(ORDER, weight) for weight in range(1, 6))
            if certify_distance
            else 0
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


def enumeration_summary(
    considered: np.ndarray,
    self_orthogonal: np.ndarray,
    rank_twelve: np.ndarray,
    canonical: np.ndarray,
) -> dict[str, Any]:
    def counts(values: np.ndarray) -> dict[str, int]:
        return {str(weight): int(values[weight]) for weight in SEARCH_WEIGHTS}

    return {
        "even_sector_polynomials": 1 << 24,
        "search_weights": list(SEARCH_WEIGHTS),
        "considered_by_weight": counts(considered),
        "self_orthogonal_by_weight": counts(self_orthogonal),
        "rank_twelve_by_weight": counts(rank_twelve),
        "canonical_by_weight": counts(canonical),
    }
