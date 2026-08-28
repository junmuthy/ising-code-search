"""Exhaustive self-dual ``C4 x C4`` two-block search at ``n=32``."""

from __future__ import annotations

import functools
import itertools
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from qldpc import codes

from gala_search.abelian_single_row import (
    C4_ORDER,
    LOGICAL_WEIGHT,
    NUM_QUBITS,
    TARGET_CHECK_RANK,
    TARGET_DIMENSION,
    certify_distance_six,
)
from gala_search.single_row import (
    _pack,
    _tanner_connected,
    gf2_rank,
)

SCHEMA_VERSION = 1
HALF_SIZE = 16
ELEMENTS = tuple(itertools.product(range(4), repeat=2))
LOGICAL_TRANSLATION = (1, 0)


def add(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    return ((left[0] + right[0]) % 4, (left[1] + right[1]) % 4)


INVOLUTIONS = tuple(element for element in ELEMENTS if add(element, element) == (0, 0))


def inverse(element: Sequence[int]) -> tuple[int, int]:
    return ((-element[0]) % 4, (-element[1]) % 4)


def index(element: Sequence[int]) -> int:
    return ELEMENTS.index((element[0] % 4, element[1] % 4))


def group_permutation(element: Sequence[int]) -> np.ndarray:
    return np.asarray([index(add(source, element)) for source in ELEMENTS], dtype=int)


@functools.lru_cache(maxsize=1)
def basis_lifts() -> tuple[np.ndarray, ...]:
    output = []
    for element in ELEMENTS:
        permutation = group_permutation(element)
        matrix = np.zeros((HALF_SIZE, HALF_SIZE), dtype=np.uint8)
        matrix[permutation, np.arange(HALF_SIZE)] = 1
        output.append(matrix)
    return tuple(output)


def polynomial_matrix(support: Sequence[int]) -> np.ndarray:
    if not support:
        return np.zeros((HALF_SIZE, HALF_SIZE), dtype=np.uint8)
    return np.bitwise_xor.reduce(
        np.asarray([basis_lifts()[member] for member in support]), axis=0
    )


def build_checks(support: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    polynomial = polynomial_matrix(support)
    check = np.hstack([polynomial, polynomial.T])
    return check, check.copy()


def permute_columns(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, np.asarray(permutation, dtype=int)] = matrix
    return output


def standard_fold(translation: Sequence[int]) -> np.ndarray:
    """Swap BB halves and map ``g`` to ``-g+t`` in each half."""
    output = np.empty(NUM_QUBITS, dtype=int)
    for half in range(2):
        for member, element in enumerate(ELEMENTS):
            target_member = index(add(inverse(element), translation))
            output[half * HALF_SIZE + member] = (1 - half) * HALF_SIZE + target_member
    return output


def diagonal_fold(translation: Sequence[int]) -> np.ndarray:
    """Apply the same order-two group translation in both BB halves."""
    internal = group_permutation(translation)
    return np.concatenate([internal, HALF_SIZE + internal]).astype(int)


def verify_fold(check: np.ndarray, fold: np.ndarray) -> dict[str, Any]:
    folded = permute_columns(check, fold)
    matching_row_permutations: list[dict[str, Any]] = []
    for reflect in (False, True):
        for translation in ELEMENTS:
            row_permutation = np.empty(HALF_SIZE, dtype=int)
            for member, element in enumerate(ELEMENTS):
                source = inverse(element) if reflect else element
                row_permutation[member] = index(add(source, translation))
            reordered = np.zeros_like(folded)
            reordered[row_permutation] = folded
            if np.array_equal(reordered, check):
                matching_row_permutations.append(
                    {"reflect": reflect, "translation": list(translation)}
                )
    return {
        "involution": bool(np.array_equal(fold[fold], np.arange(NUM_QUBITS))),
        "exact_without_row_reindexing": bool(np.array_equal(folded, check)),
        "exact_with_group_row_reindexing": bool(matching_row_permutations),
        "matching_row_permutations": matching_row_permutations,
    }


def logical_translation_permutation() -> np.ndarray:
    internal = group_permutation(LOGICAL_TRANSLATION)
    return np.concatenate([internal, HALF_SIZE + internal]).astype(int)


def permutation_orbit(seed: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    rows = []
    current = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(C4_ORDER):
        rows.append(current)
        current = permute_columns(current[None, :], permutation)[0]
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
        for _ in range(C4_ORDER):
            orbit.append(current)
            seen.add(current)
            current = int(permutation[current])
        fibres.append(tuple(orbit))
    return tuple(fibres)


def iter_graph_seeds(weight: int = LOGICAL_WEIGHT) -> Iterable[tuple[int, ...]]:
    fibres = c4_fibres()
    for selected in itertools.combinations(range(len(fibres)), weight):
        for tail in itertools.product(range(C4_ORDER), repeat=weight - 1):
            phases = (0, *tail)
            yield tuple(
                sorted(
                    fibres[fibre][phase]
                    for fibre, phase in zip(selected, phases)
                )
            )


@functools.lru_cache(maxsize=None)
def paired_graph_seeds(translation_index: int) -> tuple[tuple[int, ...], ...]:
    fold = diagonal_fold(ELEMENTS[translation_index])
    logical_shift = logical_translation_permutation()
    permutation_pairing = []
    general_pairing = []
    for support in iter_graph_seeds():
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbit = permutation_orbit(seed, logical_shift)
        x_orbit = permute_columns(z_orbit, fold)
        pairing = (z_orbit @ x_orbit.T) % 2
        if gf2_rank(pairing) != TARGET_DIMENSION:
            continue
        target = (
            permutation_pairing
            if np.all(np.count_nonzero(pairing, axis=0) == 1)
            and np.all(np.count_nonzero(pairing, axis=1) == 1)
            else general_pairing
        )
        target.append(support)
    return tuple((*permutation_pairing, *general_pairing))


def find_disjoint_logical_seed(
    check: np.ndarray, translation_index: int
) -> dict[str, Any] | None:
    fold = diagonal_fold(ELEMENTS[translation_index])
    logical_shift = logical_translation_permutation()
    column_syndromes = [_pack(check[:, qubit]) for qubit in range(NUM_QUBITS)]
    for support in paired_graph_seeds(translation_index):
        syndrome = 0
        for qubit in support:
            syndrome ^= column_syndromes[qubit]
        if syndrome:
            continue
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbit = permutation_orbit(seed, logical_shift)
        x_orbit = permute_columns(z_orbit, fold)
        if np.any((check @ z_orbit.T) % 2) or np.any((check @ x_orbit.T) % 2):
            continue
        pairing = (z_orbit @ x_orbit.T) % 2
        rank_gain = gf2_rank(np.vstack([check, z_orbit])) - gf2_rank(check)
        if rank_gain != TARGET_DIMENSION:
            continue
        return {
            "seed_support": list(support),
            "logical_supports": [
                np.flatnonzero(row).astype(int).tolist() for row in z_orbit
            ],
            "logical_weights": np.count_nonzero(z_orbit, axis=1).astype(int).tolist(),
            "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
            "rank_mod_stabilizers": rank_gain,
            "zx_pairing": pairing.astype(int).tolist(),
            "zx_pairing_rank": gf2_rank(pairing),
            "zx_pairing_is_permutation": bool(
                np.all(np.count_nonzero(pairing, axis=0) == 1)
                and np.all(np.count_nonzero(pairing, axis=1) == 1)
            ),
        }
    return None


def search_c4xc4(
    *,
    maximum_check_weight: int = 12,
    certify_distance: bool = True,
    maximum_saved_near_misses: int = 20,
) -> dict[str, Any]:
    """Exhaust all self-dual polynomials below the check-weight ceiling."""
    maximum_polynomial_weight = maximum_check_weight // 2
    counters: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    started = time.perf_counter()
    for weight in range(1, maximum_polynomial_weight + 1):
        for support in itertools.combinations(range(HALF_SIZE), weight):
            counters["polynomials"] += 1
            matrix_x, matrix_z = build_checks(support)
            if np.any((matrix_x @ matrix_z.T) % 2):
                counters["non_css"] += 1
                continue
            counters["css"] += 1
            rank = gf2_rank(matrix_x)
            if rank != TARGET_CHECK_RANK:
                counters["wrong_rank"] += 1
                continue
            counters["rank_14"] += 1
            if not _tanner_connected(matrix_x, matrix_z):
                counters["disconnected"] += 1
                continue
            counters["connected"] += 1
            selected: dict[str, Any] | None = None
            for translation in INVOLUTIONS:
                translation_index = index(translation)
                fold = diagonal_fold(translation)
                fold_analysis = verify_fold(matrix_x, fold)
                if not (
                    fold_analysis["involution"]
                    and fold_analysis["exact_with_group_row_reindexing"]
                ):
                    continue
                counters["paper_fold"] += 1
                seed = find_disjoint_logical_seed(matrix_x, translation_index)
                if seed is None:
                    continue
                counters["disjoint_logical_seed"] += 1
                selected = {
                    "group": "C4 x C4",
                    "family": "self-dual-two-block",
                    "polynomial_support": [list(ELEMENTS[index]) for index in support],
                    "polynomial_weight": weight,
                    "maximum_check_weight": 2 * weight,
                    "rank_x": rank,
                    "rank_z": rank,
                    "k": NUM_QUBITS - 2 * rank,
                    "fold_translation": list(translation),
                    "fold": {
                        **fold_analysis,
                        "permutation": fold.astype(int).tolist(),
                    },
                    "logical_grid": seed,
                }
                break
            if selected is None:
                counters["no_disjoint_logical_seed"] += 1
                continue
            distance = certify_distance_six(matrix_x, matrix_z) if certify_distance else None
            selected["distance"] = distance
            if distance and distance["certified_distance_at_least_six"]:
                counters["accepted"] += 1
                accepted.append(selected)
            else:
                counters["distance_below_six"] += 1
                if len(near_misses) < maximum_saved_near_misses:
                    near_misses.append(selected)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "self-dual-two-block",
        "group": "C4 x C4",
        "target": "[[32,4,6]]",
        "maximum_check_weight": maximum_check_weight,
        "maximum_polynomial_weight": maximum_polynomial_weight,
        "counters": dict(sorted(counters.items())),
        "accepted": accepted,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }


def build_code_from_record(record: dict[str, Any]) -> codes.CSSCode:
    support = tuple(index(element) for element in record["polynomial_support"])
    matrix_x, matrix_z = build_checks(support)
    return codes.CSSCode(matrix_x, matrix_z)
