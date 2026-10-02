"""Exhaustive floating-``k`` search over self-dual ``C4 x C4`` BB codes."""

from __future__ import annotations

import itertools
import time
from collections import Counter
from typing import Any

import numpy as np

from gala_search.abelian_bb32 import (
    ELEMENTS,
    HALF_SIZE,
    build_checks,
    c4_fibres,
    logical_translation_permutation,
    permutation_orbit,
)
from gala_search.abelian_single_row import certify_distance_six
from gala_search.single_row import _pack, _tanner_connected, gf2_rank

NUM_QUBITS = 32
LOGICAL_WEIGHT = 7
MINIMUM_K = 4
MAXIMUM_K = NUM_QUBITS
MAXIMUM_POLYNOMIAL_WEIGHT = 6


def _xor_support(values: list[int], support: tuple[int, ...]) -> int:
    output = 0
    for index in support:
        output ^= values[index]
    return output


def find_graph_seed(check: np.ndarray) -> tuple[int, ...] | None:
    """Solve the seven-of-eight fibre zero-syndrome problem exactly.

    For each omitted fibre, the remaining seven are split three plus four.
    Their 320 phase assignments are matched by packed syndrome, replacing the
    original 32,768-seed scan.
    """
    fibres = c4_fibres()
    column_syndromes = [_pack(check[:, qubit]) for qubit in range(NUM_QUBITS)]
    for omitted in range(len(fibres)):
        selected = [index for index in range(len(fibres)) if index != omitted]
        left_fibres = selected[:3]
        right_fibres = selected[3:]
        left_by_syndrome: dict[int, tuple[int, ...]] = {}
        for phases in itertools.product(range(4), repeat=len(left_fibres)):
            support = tuple(
                fibres[fibre][phase]
                for fibre, phase in zip(left_fibres, phases, strict=True)
            )
            left_by_syndrome.setdefault(
                _xor_support(column_syndromes, support), support
            )
        for phases in itertools.product(range(4), repeat=len(right_fibres)):
            support = tuple(
                fibres[fibre][phase]
                for fibre, phase in zip(right_fibres, phases, strict=True)
            )
            syndrome = _xor_support(column_syndromes, support)
            if syndrome in left_by_syndrome:
                return tuple(sorted((*left_by_syndrome[syndrome], *support)))
    return None


def analyze_logical_orbit(
    check: np.ndarray, support: tuple[int, ...]
) -> dict[str, Any]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = permutation_orbit(seed, logical_translation_permutation())
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
            np.array_equal(pairing, np.eye(4, dtype=np.uint8))
        ),
        "translation": [1, 0],
    }


def search(
    *,
    maximum_certifications: int = 0,
    maximum_saved_near_misses: int = 50,
) -> dict[str, Any]:
    """Exhaust every polynomial of weight at most six with ``k >= 4``."""
    counters: Counter[str] = Counter()
    structural_candidates: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    started = time.perf_counter()
    for weight in range(1, MAXIMUM_POLYNOMIAL_WEIGHT + 1):
        for support in itertools.combinations(range(HALF_SIZE), weight):
            counters["polynomials"] += 1
            matrix_x, matrix_z = build_checks(support)
            if np.any((matrix_x @ matrix_z.T) % 2):
                raise AssertionError("self-dual BB construction failed CSS")
            rank = gf2_rank(matrix_x)
            dimension = NUM_QUBITS - 2 * rank
            counters[f"k_{dimension}"] += 1
            if not MINIMUM_K <= dimension <= MAXIMUM_K:
                counters["dimension_outside_target"] += 1
                continue
            counters["dimension_target"] += 1
            connected = _tanner_connected(matrix_x, matrix_z)
            if not connected:
                counters["disconnected"] += 1
            else:
                counters["connected"] += 1
            seed = find_graph_seed(matrix_x)
            if seed is None:
                counters["no_disjoint_logical_seed"] += 1
                continue
            counters["disjoint_logical_seed"] += 1
            if not connected:
                counters["disjoint_logical_seed_disconnected"] += 1
            logical = analyze_logical_orbit(matrix_x, seed)
            if not (
                logical["pairwise_disjoint"]
                and logical["orbit_in_kernel"]
                and logical["rank_mod_stabilizers"] == 4
                and logical["zx_pairing_is_identity"]
            ):
                raise AssertionError("meet-in-the-middle seed failed validation")
            record = {
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
            structural_candidates.append(record)
    limit = maximum_certifications or len(structural_candidates)
    for index, record in enumerate(structural_candidates):
        if index >= limit:
            break
        matrix_x, matrix_z = build_checks(record["polynomial_indices"])
        distance = certify_distance_six(matrix_x, matrix_z)
        record["distance"] = distance
        counters["distance_certifications"] += 1
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
        "target": "[[32,k,>=6]] with k >= 4 and one protected C4 row",
        "family": "floating-k-self-dual-C4xC4-BB",
        "maximum_polynomial_weight": MAXIMUM_POLYNOMIAL_WEIGHT,
        "maximum_check_weight": 2 * MAXIMUM_POLYNOMIAL_WEIGHT,
        "counters": dict(sorted(counters.items())),
        "structural_candidates": structural_candidates,
        "distance_at_least_six_survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
