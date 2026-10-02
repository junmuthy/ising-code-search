"""Seed-first sparse ``C4 x C5`` BB search.

For a candidate Z logical ``z=(z_L,z_R)``, the BB kernel equation is
``a z_L + b z_R = 0``.  Commutativity makes
``(a,b)=(c z_R,c z_L)`` an exact sparse solution.  We therefore sample the
desired disjoint C4 orbit first and construct compatible checks around it.
"""

from __future__ import annotations

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
    permute_vectors,
    rowspace_zx_dual,
    standard_fold,
)
from searches.c4xc5_search.search import (
    NUM_QUBITS,
    X_ORDER,
    add,
    c4_fibres,
    index,
    translation_orbit,
)
from gala_search.single_row import _tanner_connected, gf2_rank

SEED_WEIGHTS = (6, 7, 8, 9)
FACTOR_WEIGHTS = (1, 2, 3)


def multiply_supports(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Multiply binary group-ring supports in ``C4 x C5``."""
    parity: set[int] = set()
    for left_index in left:
        for right_index in right:
            product = index(add(ELEMENTS[left_index], ELEMENTS[right_index]))
            if product in parity:
                parity.remove(product)
            else:
                parity.add(product)
    return tuple(sorted(parity))


def random_graph_seed(rng: random.Random, weight: int) -> tuple[int, ...]:
    """Return a seed meeting each physical C4 orbit at most once."""
    fibres = c4_fibres()
    while True:
        selected = rng.sample(range(len(fibres)), weight)
        support = tuple(
            sorted(fibres[fibre][rng.randrange(X_ORDER)] for fibre in selected)
        )
        if any(qubit < GROUP_ORDER for qubit in support) and any(
            qubit >= GROUP_ORDER for qubit in support
        ):
            return support


def random_factor(rng: random.Random, weight: int) -> tuple[int, ...]:
    """Return a factor normalized to contain the identity monomial."""
    if weight == 1:
        return (0,)
    return tuple(sorted((0, *rng.sample(range(1, GROUP_ORDER), weight - 1))))


def polynomial_pair_from_seed(
    seed_support: tuple[int, ...], factor_support: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Construct ``(a,b)=(c z_R,c z_L)`` from a physical seed."""
    left = tuple(qubit for qubit in seed_support if qubit < GROUP_ORDER)
    right = tuple(
        qubit - GROUP_ORDER
        for qubit in seed_support
        if qubit >= GROUP_ORDER
    )
    return (
        multiply_supports(factor_support, right),
        multiply_supports(factor_support, left),
    )


def analyze_known_seed(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    seed_support: tuple[int, ...],
) -> dict[str, Any] | None:
    """Validate a disjoint C4 seed and find a full-rank ZX fold."""
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(seed_support)] = 1
    z_orbit = translation_orbit(seed)
    if np.any((matrix_x @ z_orbit.T) % 2):
        return None
    rank_z = gf2_rank(matrix_z)
    rank_gain = gf2_rank(np.vstack([matrix_z, z_orbit])) - rank_z
    if rank_gain != X_ORDER:
        return None
    for translation in ELEMENTS:
        fold = standard_fold(translation)
        x_orbit = permute_vectors(z_orbit, fold)
        if np.any((matrix_z @ x_orbit.T) % 2):
            raise AssertionError("automatic BB ZX partner left the X kernel")
        pairing = (z_orbit @ x_orbit.T) % 2
        if gf2_rank(pairing) != X_ORDER:
            continue
        return {
            "seed_support": list(seed_support),
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
            "zx_pairing_rank": X_ORDER,
        }
    return None


def search_seeded(
    *, trials: int, random_seed: int, maximum_saved_near_misses: int = 100
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    survivors = []
    near_misses = []
    started = time.perf_counter()
    for trial in range(trials):
        seed_weight = SEED_WEIGHTS[trial % len(SEED_WEIGHTS)]
        factor_weight = FACTOR_WEIGHTS[
            (trial // len(SEED_WEIGHTS)) % len(FACTOR_WEIGHTS)
        ]
        seed_support = random_graph_seed(rng, seed_weight)
        factor_support = random_factor(rng, factor_weight)
        support_a, support_b = polynomial_pair_from_seed(
            seed_support, factor_support
        )
        check_weight = len(support_a) + len(support_b)
        counters[f"seed_weight_{seed_weight}"] += 1
        counters[f"factor_weight_{factor_weight}"] += 1
        counters[f"derived_check_weight_{check_weight}"] += 1
        if check_weight not in TOTAL_CHECK_WEIGHTS:
            counters["check_weight_out_of_scope"] += 1
            continue
        if not support_a or not support_b:
            counters["empty_polynomial"] += 1
            continue
        key = min((support_a, support_b), (support_b, support_a))
        if key in seen:
            counters["duplicates"] += 1
            continue
        seen.add(key)
        counters["polynomial_pairs"] += 1
        matrix_x, matrix_z = build_checks(support_a, support_b)
        if np.any((matrix_x @ matrix_z.T) % 2):
            raise AssertionError("seeded abelian BB construction failed CSS")
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
            "trial": trial,
            "support_a_indices": list(support_a),
            "support_b_indices": list(support_b),
            "support_a": [list(ELEMENTS[item]) for item in support_a],
            "support_b": [list(ELEMENTS[item]) for item in support_b],
            "factor_support_indices": list(factor_support),
            "factor_support": [list(ELEMENTS[item]) for item in factor_support],
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
        "target": "[[40,k,>=6]] seed-first C4xC5 BB",
        "method": "sample-disjoint-seed-then-construct-commutative-syzygy",
        "trials": trials,
        "random_seed": random_seed,
        "seed_weights": list(SEED_WEIGHTS),
        "factor_weights": list(FACTOR_WEIGHTS),
        "maximum_check_weight": max(TOTAL_CHECK_WEIGHTS),
        "counters": dict(sorted(counters.items())),
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
