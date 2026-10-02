"""Seed-constrained sparse two-polynomial search over ``C4 x C5``.

Unlike the elementary syzygy in :mod:`c4xc5_search.seeded`, this search fixes
a disjoint C4 seed ``z`` and solves the full linear condition
``a z_L + b z_R = 0`` for sparse, otherwise independent polynomials ``a,b``.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import Counter
from typing import Any

import numpy as np

from searches.c4xc5_search.general import (
    ELEMENTS,
    GROUP_ORDER,
    TOTAL_CHECK_WEIGHTS,
    build_checks,
    certify_distance_six,
    rowspace_zx_dual,
)
from searches.c4xc5_search.search import NUM_QUBITS, X_ORDER, _pack, basis_lifts
from searches.c4xc5_search.seeded import analyze_known_seed, random_graph_seed
from gala_search.single_row import _tanner_connected, gf2_rank


def seed_constraint_syndromes(seed_support: tuple[int, ...]) -> tuple[int, ...]:
    """Return the 40 packed columns of ``a z_L + b z_R = 0``."""
    left = np.zeros(GROUP_ORDER, dtype=np.uint8)
    right = np.zeros(GROUP_ORDER, dtype=np.uint8)
    for qubit in seed_support:
        if qubit < GROUP_ORDER:
            left[qubit] = 1
        else:
            right[qubit - GROUP_ORDER] = 1
    return tuple(
        [_pack(term @ left % 2) for term in basis_lifts()]
        + [_pack(term @ right % 2) for term in basis_lifts()]
    )


def _xor_syndrome(indices: tuple[int, ...], syndromes: tuple[int, ...]) -> int:
    value = 0
    for item in indices:
        value ^= syndromes[item]
    return value


def compatible_sparse_pairs(
    seed_support: tuple[int, ...],
    *,
    rng: random.Random,
    restarts: int,
    maximum_pairs: int,
    matches_per_key: int = 4,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Find sparse ``(a,b)`` solutions with randomized meet-in-the-middle.

    Each restart partitions the 40 coefficient variables into two sets of 20.
    We enumerate up to six selected variables per side.  Random repartitioning
    makes every solution of total weight at most 12 visible with high
    probability, while retaining several collisions per syndrome to avoid
    repeatedly returning the elementary stabilizer-valued syzygy.
    """
    syndromes = seed_constraint_syndromes(seed_support)
    output: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for _restart in range(restarts):
        order = list(range(2 * GROUP_ORDER))
        rng.shuffle(order)
        left_variables = order[:GROUP_ORDER]
        right_variables = order[GROUP_ORDER:]
        left_maps: list[dict[int, list[tuple[int, ...]]]] = [
            {} for _ in range(7)
        ]
        for count in range(7):
            for combination in itertools.combinations(left_variables, count):
                syndrome = _xor_syndrome(combination, syndromes)
                matches = left_maps[count].setdefault(syndrome, [])
                if len(matches) < matches_per_key:
                    matches.append(combination)
        for target_weight in TOTAL_CHECK_WEIGHTS:
            for right_count in range(7):
                left_count = target_weight - right_count
                if not 0 <= left_count <= 6:
                    continue
                for right_combination in itertools.combinations(
                    right_variables, right_count
                ):
                    syndrome = _xor_syndrome(right_combination, syndromes)
                    for left_combination in left_maps[left_count].get(
                        syndrome, ()
                    ):
                        coefficients = (*left_combination, *right_combination)
                        support_a = tuple(
                            sorted(item for item in coefficients if item < GROUP_ORDER)
                        )
                        support_b = tuple(
                            sorted(
                                item - GROUP_ORDER
                                for item in coefficients
                                if item >= GROUP_ORDER
                            )
                        )
                        if not support_a or not support_b:
                            continue
                        key = min((support_a, support_b), (support_b, support_a))
                        if key in seen:
                            continue
                        seen.add(key)
                        output.append((support_a, support_b))
                        if len(output) >= maximum_pairs:
                            return output
    return output


def search_solved(
    *,
    seeds: int,
    solver_restarts: int,
    pairs_per_seed: int,
    random_seed: int,
    maximum_saved_near_misses: int = 100,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    survivors = []
    near_misses = []
    started = time.perf_counter()
    for seed_index in range(seeds):
        seed_weight = (6, 7, 8, 9)[seed_index % 4]
        seed_support = random_graph_seed(rng, seed_weight)
        counters[f"seed_weight_{seed_weight}"] += 1
        pairs = compatible_sparse_pairs(
            seed_support,
            rng=rng,
            restarts=solver_restarts,
            maximum_pairs=pairs_per_seed,
        )
        counters["compatible_pairs_returned"] += len(pairs)
        if not pairs:
            counters["seeds_without_sparse_solution"] += 1
        for support_a, support_b in pairs:
            key = min((support_a, support_b), (support_b, support_a))
            if key in seen_pairs:
                counters["duplicate_polynomial_pairs"] += 1
                continue
            seen_pairs.add(key)
            counters["polynomial_pairs"] += 1
            check_weight = len(support_a) + len(support_b)
            counters[f"check_weight_{check_weight}"] += 1
            matrix_x, matrix_z = build_checks(support_a, support_b)
            if np.any((matrix_x @ matrix_z.T) % 2):
                raise AssertionError("solved abelian BB construction failed CSS")
            rank_x = gf2_rank(matrix_x)
            rank_z = gf2_rank(matrix_z)
            dimension = NUM_QUBITS - rank_x - rank_z
            counters[f"k_{dimension}"] += 1
            if dimension < X_ORDER:
                counters["dimension_below_four"] += 1
                continue
            if not _tanner_connected(matrix_x, matrix_z):
                counters["disconnected"] += 1
                continue
            counters["connected"] += 1
            logical = analyze_known_seed(matrix_x, matrix_z, seed_support)
            if logical is None:
                counters["seed_not_nondegenerate"] += 1
                continue
            counters["paired_logical_seed"] += 1
            distance = certify_distance_six(matrix_x, matrix_z)
            record = {
                "seed_index": seed_index,
                "seed_support": list(seed_support),
                "support_a_indices": list(support_a),
                "support_b_indices": list(support_b),
                "support_a": [list(ELEMENTS[item]) for item in support_a],
                "support_b": [list(ELEMENTS[item]) for item in support_b],
                "check_weight": check_weight,
                "rank_x": rank_x,
                "rank_z": rank_z,
                "k": dimension,
                "rowspace_zx_dual": rowspace_zx_dual(matrix_x, matrix_z),
                "logical_grid": logical,
                "distance": distance,
            }
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
        "target": "[[40,k,>=6]] seed-constrained C4xC5 BB",
        "method": "choose-disjoint-c4-seed-and-solve-full-sparse-kernel-system",
        "seeds": seeds,
        "solver_restarts": solver_restarts,
        "pairs_per_seed": pairs_per_seed,
        "random_seed": random_seed,
        "check_weights": list(TOTAL_CHECK_WEIGHTS),
        "counters": dict(sorted(counters.items())),
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
