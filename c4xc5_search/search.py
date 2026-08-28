"""Exhaustive self-dual ``C4 x C5`` BB search with one protected row."""

from __future__ import annotations

import functools
import itertools
import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from numba import njit, types
from numba.typed import Dict

from gala_search.single_row import _pack, _tanner_connected, gf2_rank

X_ORDER = 4
Y_ORDER = 5
GROUP_ORDER = X_ORDER * Y_ORDER
NUM_HALVES = 2
NUM_QUBITS = NUM_HALVES * GROUP_ORDER
MAXIMUM_POLYNOMIAL_WEIGHT = 6
MINIMUM_K = 4
LOGICAL_WEIGHTS = (7, 9)
ELEMENTS = tuple(itertools.product(range(X_ORDER), range(Y_ORDER)))


def add(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    return ((left[0] + right[0]) % X_ORDER, (left[1] + right[1]) % Y_ORDER)


def index(element: Sequence[int]) -> int:
    return (element[0] % X_ORDER) * Y_ORDER + (element[1] % Y_ORDER)


def group_permutation(element: Sequence[int]) -> np.ndarray:
    return np.asarray([index(add(source, element)) for source in ELEMENTS], dtype=int)


@functools.lru_cache(maxsize=1)
def basis_lifts() -> tuple[np.ndarray, ...]:
    output = []
    for element in ELEMENTS:
        permutation = group_permutation(element)
        matrix = np.zeros((GROUP_ORDER, GROUP_ORDER), dtype=np.uint8)
        matrix[permutation, np.arange(GROUP_ORDER)] = 1
        output.append(matrix)
    return tuple(output)


def polynomial_matrix(support: Sequence[int]) -> np.ndarray:
    return np.bitwise_xor.reduce(
        np.asarray([basis_lifts()[member] for member in support]), axis=0
    )


def build_checks(support: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    polynomial = polynomial_matrix(support)
    check = np.hstack([polynomial, polynomial.T]).astype(np.uint8)
    return check, check.copy()


def logical_translation_permutation() -> np.ndarray:
    internal = group_permutation((1, 0))
    return np.concatenate([internal, GROUP_ORDER + internal]).astype(int)


def permute_vector(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vector)
    output[permutation] = vector
    return output


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    permutation = logical_translation_permutation()
    rows = []
    current = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(X_ORDER):
        rows.append(current)
        current = permute_vector(current, permutation)
    return np.asarray(rows, dtype=np.uint8)


@functools.lru_cache(maxsize=1)
def c4_fibres() -> tuple[tuple[int, ...], ...]:
    permutation = logical_translation_permutation()
    seen: set[int] = set()
    fibres = []
    for start in range(NUM_QUBITS):
        if start in seen:
            continue
        orbit = []
        current = start
        for _ in range(X_ORDER):
            orbit.append(current)
            seen.add(current)
            current = int(permutation[current])
        fibres.append(tuple(orbit))
    return tuple(fibres)


@njit(cache=True)
def _find_seed_codes(syndromes: np.ndarray) -> tuple[int, int, int]:
    """Return target weight and base-five phase codes for a zero syndrome."""
    for target_weight in LOGICAL_WEIGHTS:
        left = Dict.empty(key_type=types.uint64, value_type=types.int64)
        for encoded in range(5**5):
            value = encoded
            count = 0
            syndrome = np.uint64(0)
            for fibre in range(5):
                digit = value % 5
                value //= 5
                if digit:
                    count += 1
                    syndrome ^= syndromes[fibre, digit - 1]
            key = (np.uint64(count) << np.uint64(20)) | syndrome
            if key not in left:
                left[key] = encoded
        for encoded in range(5**5):
            value = encoded
            count = 0
            syndrome = np.uint64(0)
            for local in range(5):
                digit = value % 5
                value //= 5
                if digit:
                    count += 1
                    syndrome ^= syndromes[5 + local, digit - 1]
            needed = target_weight - count
            if needed < 0 or needed > 5:
                continue
            key = (np.uint64(needed) << np.uint64(20)) | syndrome
            if key in left:
                return target_weight, int(left[key]), encoded
    return 0, -1, -1


@njit(cache=True)
def find_seed_codes_at_weight(
    syndromes: np.ndarray, target_weight: int
) -> tuple[int, int]:
    """Return one five-plus-five phase match at an exact target weight."""
    left = Dict.empty(key_type=types.uint64, value_type=types.int64)
    for encoded in range(5**5):
        value = encoded
        count = 0
        syndrome = np.uint64(0)
        for fibre in range(5):
            digit = value % 5
            value //= 5
            if digit:
                count += 1
                syndrome ^= syndromes[fibre, digit - 1]
        key = (np.uint64(count) << np.uint64(20)) | syndrome
        if key not in left:
            left[key] = encoded
    for encoded in range(5**5):
        value = encoded
        count = 0
        syndrome = np.uint64(0)
        for local in range(5):
            digit = value % 5
            value //= 5
            if digit:
                count += 1
                syndrome ^= syndromes[5 + local, digit - 1]
        needed = target_weight - count
        if needed < 0 or needed > 5:
            continue
        key = (np.uint64(needed) << np.uint64(20)) | syndrome
        if key in left:
            return int(left[key]), encoded
    return -1, -1


def find_graph_seed(check: np.ndarray) -> tuple[int, ...] | None:
    fibres = c4_fibres()
    syndromes = np.zeros((len(fibres), X_ORDER), dtype=np.uint64)
    for fibre, orbit in enumerate(fibres):
        for phase, qubit in enumerate(orbit):
            syndromes[fibre, phase] = _pack(check[:, qubit])
    weight, left_code, right_code = _find_seed_codes(syndromes)
    if weight == 0:
        return None
    support = []
    for offset, encoded in ((0, left_code), (5, right_code)):
        value = encoded
        for local in range(5):
            digit = value % 5
            value //= 5
            if digit:
                support.append(fibres[offset + local][digit - 1])
    if len(support) != weight:
        raise AssertionError("decoded seed weight disagrees with compiled solver")
    return tuple(sorted(support))


def analyze_logical_orbit(
    check: np.ndarray, support: tuple[int, ...]
) -> dict[str, Any]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = translation_orbit(seed)
    pairing = (orbit @ orbit.T) % 2
    rank = gf2_rank(check)
    return {
        "seed_support": list(support),
        "logical_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in orbit
        ],
        "logical_weights": np.count_nonzero(orbit, axis=1).astype(int).tolist(),
        "pairwise_disjoint": bool(np.max(np.sum(orbit, axis=0), initial=0) <= 1),
        "orbit_in_kernel": bool(not np.any((check @ orbit.T) % 2)),
        "rank_mod_stabilizers": gf2_rank(np.vstack([check, orbit])) - rank,
        "zx_pairing": pairing.astype(int).tolist(),
        "zx_pairing_rank": gf2_rank(pairing),
        "zx_pairing_is_identity": bool(
            np.array_equal(pairing, np.eye(X_ORDER, dtype=np.uint8))
        ),
        "translation": [1, 0],
    }


def packed_row_basis(matrix: np.ndarray) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = sum(int(bit) << column for column, bit in enumerate(row))
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
        sum(int(matrix[row, column]) << row for row in range(GROUP_ORDER))
        for column in range(NUM_QUBITS)
    ]
    stabilizer_basis = packed_row_basis(matrix)
    for weight in range(1, 6):
        for support in itertools.combinations(range(NUM_QUBITS), weight):
            syndrome = 0
            vector = 0
            for qubit in support:
                syndrome ^= column_syndromes[qubit]
                vector |= 1 << qubit
            if syndrome == 0 and not packed_in_span(vector, stabilizer_basis):
                return {"weight": weight, "support": list(support)}
    return None


def certify_distance_six(matrix: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    logical = find_logical_below_six(matrix)
    return {
        "certified_distance_at_least_six": logical is None,
        "z_logical_below_six": logical,
        "x_logical_below_six": logical,
        "supports_checked_per_sector": sum(
            math.comb(NUM_QUBITS, weight) for weight in range(1, 6)
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


def search(
    *,
    maximum_certifications: int = 0,
    maximum_saved_near_misses: int = 50,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    structural_candidates: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    started = time.perf_counter()
    for weight in range(1, MAXIMUM_POLYNOMIAL_WEIGHT + 1):
        for support in itertools.combinations(range(GROUP_ORDER), weight):
            counters["polynomials"] += 1
            matrix_x, matrix_z = build_checks(support)
            if np.any((matrix_x @ matrix_z.T) % 2):
                raise AssertionError("self-dual BB construction failed CSS")
            rank = gf2_rank(matrix_x)
            dimension = NUM_QUBITS - 2 * rank
            counters[f"k_{dimension}"] += 1
            if dimension < MINIMUM_K:
                counters["dimension_below_four"] += 1
                continue
            counters["dimension_at_least_four"] += 1
            connected = _tanner_connected(matrix_x, matrix_z)
            counters["connected" if connected else "disconnected"] += 1
            seed = find_graph_seed(matrix_x)
            if seed is None:
                counters["no_disjoint_logical_seed"] += 1
                continue
            counters["disjoint_logical_seed"] += 1
            logical = analyze_logical_orbit(matrix_x, seed)
            if not (
                logical["pairwise_disjoint"]
                and logical["orbit_in_kernel"]
                and logical["rank_mod_stabilizers"] == 4
                and logical["zx_pairing_is_identity"]
            ):
                raise AssertionError("compiled seed failed exact validation")
            structural_candidates.append(
                {
                    "polynomial_indices": list(support),
                    "polynomial_support": [list(ELEMENTS[index]) for index in support],
                    "polynomial_weight": weight,
                    "check_weight": 2 * weight,
                    "rank_x": rank,
                    "rank_z": rank,
                    "k": dimension,
                    "tanner_connected": connected,
                    "logical_grid": logical,
                    "distance": None,
                }
            )
    limit = maximum_certifications or len(structural_candidates)
    for index, record in enumerate(structural_candidates):
        if index >= limit:
            break
        matrix, _matrix_z = build_checks(record["polynomial_indices"])
        distance = certify_distance_six(matrix)
        record["distance"] = distance
        counters["distance_certifications"] += 1
        if distance["certified_distance_at_least_six"]:
            counters["distance_at_least_six"] += 1
            survivors.append(record)
        else:
            low_weight = distance["z_logical_below_six"]["weight"]
            counters[f"distance_{low_weight}"] += 1
            if len(near_misses) < maximum_saved_near_misses:
                near_misses.append(record)
    return {
        "target": "[[40,k,>=6]] with k >= 4 and one protected C4 row",
        "family": "self-dual-two-block-C4xC5-BB",
        "maximum_polynomial_weight": MAXIMUM_POLYNOMIAL_WEIGHT,
        "maximum_check_weight": 2 * MAXIMUM_POLYNOMIAL_WEIGHT,
        "counters": dict(sorted(counters.items())),
        "structural_candidates": structural_candidates,
        "distance_at_least_six_survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
