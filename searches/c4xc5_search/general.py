"""Bounded general two-polynomial ``C4 x C5`` BB search."""

from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from searches.c4xc5_search.search import (
    ELEMENTS,
    GROUP_ORDER,
    NUM_QUBITS,
    X_ORDER,
    add,
    c4_fibres,
    find_seed_codes_at_weight,
    group_permutation,
    index,
    logical_translation_permutation,
    packed_in_span,
    packed_row_basis,
    polynomial_matrix,
    translation_orbit,
)
from gala_search.single_row import _pack, _tanner_connected, gf2_rank

PAIRING_WEIGHTS = (7, 9, 6, 8)
TOTAL_CHECK_WEIGHTS = (6, 8, 10, 12)


def inverse(element: Sequence[int]) -> tuple[int, int]:
    return ((-element[0]) % 4, (-element[1]) % 5)


def build_checks(
    support_a: Sequence[int], support_b: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    aa = polynomial_matrix(support_a)
    bb = polynomial_matrix(support_b)
    return (
        np.hstack([aa, bb]).astype(np.uint8),
        np.hstack([bb.T, aa.T]).astype(np.uint8),
    )


def standard_fold(translation: Sequence[int]) -> np.ndarray:
    output = np.empty(NUM_QUBITS, dtype=int)
    for half in range(2):
        for member, element in enumerate(ELEMENTS):
            target = index(add(inverse(element), translation))
            output[half * GROUP_ORDER + member] = (
                (1 - half) * GROUP_ORDER + target
            )
    return output


def permute_vectors(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def rowspace_zx_dual(matrix_x: np.ndarray, matrix_z: np.ndarray) -> bool:
    fold = standard_fold((0, 0))
    transformed = np.zeros_like(matrix_x)
    transformed[:, fold] = matrix_x
    return gf2_rank(np.vstack([matrix_z, transformed])) == gf2_rank(matrix_z)


def _decode_seed(
    ordered_fibres: list[tuple[int, ...]], left_code: int, right_code: int
) -> tuple[int, ...]:
    support = []
    for offset, encoded in ((0, left_code), (5, right_code)):
        value = encoded
        for local in range(5):
            digit = value % 5
            value //= 5
            if digit:
                support.append(ordered_fibres[offset + local][digit - 1])
    return tuple(sorted(support))


def find_paired_seed(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    *,
    rng: random.Random,
    restarts: int,
) -> dict[str, Any] | None:
    base_fibres = list(c4_fibres())
    column_syndromes = [_pack(matrix_x[:, qubit]) for qubit in range(NUM_QUBITS)]
    for _restart in range(restarts):
        order = list(range(len(base_fibres)))
        rng.shuffle(order)
        ordered_fibres = []
        for fibre_index in order:
            fibre = list(base_fibres[fibre_index])
            shift = rng.randrange(X_ORDER)
            fibre = fibre[shift:] + fibre[:shift]
            ordered_fibres.append(tuple(fibre))
        syndromes = np.zeros((len(ordered_fibres), X_ORDER), dtype=np.uint64)
        for fibre_index, fibre in enumerate(ordered_fibres):
            for phase, qubit in enumerate(fibre):
                syndromes[fibre_index, phase] = column_syndromes[qubit]
        for target_weight in PAIRING_WEIGHTS:
            left_code, right_code = find_seed_codes_at_weight(
                syndromes, target_weight
            )
            if left_code < 0:
                continue
            support = _decode_seed(ordered_fibres, left_code, right_code)
            seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
            seed[list(support)] = 1
            z_orbit = translation_orbit(seed)
            if np.any((matrix_x @ z_orbit.T) % 2):
                raise AssertionError("compiled general seed failed kernel validation")
            for translation in ELEMENTS:
                fold = standard_fold(translation)
                x_orbit = permute_vectors(z_orbit, fold)
                if np.any((matrix_z @ x_orbit.T) % 2):
                    raise AssertionError("ZX partner failed kernel validation")
                pairing = (z_orbit @ x_orbit.T) % 2
                if gf2_rank(pairing) != 4:
                    continue
                rank_z = gf2_rank(matrix_z)
                rank_gain = gf2_rank(np.vstack([matrix_z, z_orbit])) - rank_z
                if rank_gain != 4:
                    continue
                return {
                    "seed_support": list(support),
                    "logical_supports": [
                        np.flatnonzero(row).astype(int).tolist() for row in z_orbit
                    ],
                    "logical_weights": np.count_nonzero(z_orbit, axis=1)
                    .astype(int)
                    .tolist(),
                    "pairwise_disjoint": bool(
                        np.max(np.sum(z_orbit, axis=0), initial=0) <= 1
                    ),
                    "rank_mod_stabilizers": rank_gain,
                    "fold_translation": list(translation),
                    "fold_permutation": fold.astype(int).tolist(),
                    "zx_pairing": pairing.astype(int).tolist(),
                    "zx_pairing_rank": 4,
                }
    return None


def find_logical_below_six(
    check: np.ndarray, stabilizer: np.ndarray
) -> dict[str, Any] | None:
    column_syndromes = [
        sum(int(check[row, column]) << row for row in range(GROUP_ORDER))
        for column in range(NUM_QUBITS)
    ]
    stabilizer_basis = packed_row_basis(stabilizer)
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


def certify_distance_six(
    matrix_x: np.ndarray, matrix_z: np.ndarray
) -> dict[str, Any]:
    started = time.perf_counter()
    z_logical = find_logical_below_six(matrix_x, matrix_z)
    x_logical = find_logical_below_six(matrix_z, matrix_x)
    return {
        "certified_distance_at_least_six": z_logical is None and x_logical is None,
        "z_logical_below_six": z_logical,
        "x_logical_below_six": x_logical,
        "supports_checked_per_sector": sum(
            math.comb(NUM_QUBITS, weight) for weight in range(1, 6)
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


def _random_support_pair(
    rng: random.Random, total_weight: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    weight_a = rng.randrange(1, total_weight)
    weight_b = total_weight - weight_a
    if weight_a > GROUP_ORDER or weight_b > GROUP_ORDER:
        return _random_support_pair(rng, total_weight)
    return (
        tuple(sorted(rng.sample(range(GROUP_ORDER), weight_a))),
        tuple(sorted(rng.sample(range(GROUP_ORDER), weight_b))),
    )


def search_general(
    *,
    trials: int,
    seed_restarts: int,
    random_seed: int,
    maximum_saved_near_misses: int = 50,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    records = []
    survivors = []
    near_misses = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    started = time.perf_counter()
    for trial in range(trials):
        total_weight = TOTAL_CHECK_WEIGHTS[trial % len(TOTAL_CHECK_WEIGHTS)]
        support_a, support_b = _random_support_pair(rng, total_weight)
        key = min((support_a, support_b), (support_b, support_a))
        if key in seen:
            counters["duplicates"] += 1
            continue
        seen.add(key)
        counters["pairs"] += 1
        counters[f"check_weight_{total_weight}"] += 1
        matrix_x, matrix_z = build_checks(support_a, support_b)
        if np.any((matrix_x @ matrix_z.T) % 2):
            raise AssertionError("abelian two-polynomial construction failed CSS")
        rank_x = gf2_rank(matrix_x)
        rank_z = gf2_rank(matrix_z)
        dimension = NUM_QUBITS - rank_x - rank_z
        counters[f"k_{dimension}"] += 1
        if dimension < 4:
            counters["dimension_below_four"] += 1
            continue
        if not _tanner_connected(matrix_x, matrix_z):
            counters["disconnected"] += 1
            continue
        counters["connected"] += 1
        logical = find_paired_seed(
            matrix_x, matrix_z, rng=rng, restarts=seed_restarts
        )
        if logical is None:
            counters["no_paired_logical_seed"] += 1
            continue
        counters["paired_logical_seed"] += 1
        distance = certify_distance_six(matrix_x, matrix_z)
        record = {
            "trial": trial,
            "support_a_indices": list(support_a),
            "support_b_indices": list(support_b),
            "support_a": [list(ELEMENTS[index]) for index in support_a],
            "support_b": [list(ELEMENTS[index]) for index in support_b],
            "check_weight": total_weight,
            "rank_x": rank_x,
            "rank_z": rank_z,
            "k": dimension,
            "rowspace_zx_dual": rowspace_zx_dual(matrix_x, matrix_z),
            "logical_grid": logical,
            "distance": distance,
        }
        records.append(record)
        if distance["certified_distance_at_least_six"]:
            counters["distance_at_least_six"] += 1
            survivors.append(record)
        else:
            low = min(
                item["weight"]
                for item in (
                    distance["z_logical_below_six"],
                    distance["x_logical_below_six"],
                )
                if item is not None
            )
            counters[f"distance_{low}"] += 1
            if len(near_misses) < maximum_saved_near_misses:
                near_misses.append(record)
    return {
        "target": "[[40,k,>=6]] general two-polynomial C4xC5 BB",
        "method": "bounded-random-sparse-pairs-exact-seed-and-distance-screen",
        "trials": trials,
        "seed_restarts": seed_restarts,
        "random_seed": random_seed,
        "check_weights": list(TOTAL_CHECK_WEIGHTS),
        "counters": dict(sorted(counters.items())),
        "records": records,
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
