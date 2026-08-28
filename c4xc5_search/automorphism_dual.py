"""Exhaustive half-preserving ZX-dual search over ``C4 x C5``.

For an involutive group automorphism ``sigma``, impose
``b = sigma(a)^dagger``.  The resulting BB checks are exchanged by
Hadamard followed by the half-preserving physical permutation ``sigma``.
This avoids the identically zero self-pairing caused by the universal
half-swapping BB fold.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from c4xc5_search.general import (
    ELEMENTS,
    GROUP_ORDER,
    PAIRING_WEIGHTS,
    build_checks,
    certify_distance_six,
    permute_vectors,
)
from c4xc5_search.search import (
    MAXIMUM_POLYNOMIAL_WEIGHT,
    NUM_QUBITS,
    X_ORDER,
    add,
    c4_fibres,
    find_seed_codes_at_weight,
    index,
    logical_translation_permutation,
    translation_orbit,
)
from gala_search.single_row import _pack, _tanner_connected, gf2_rank

INVOLUTIONS = ((3, 1), (1, 4), (3, 4))


def transformed_partner_support(
    support_a: tuple[int, ...], sigma: tuple[int, int]
) -> tuple[int, ...]:
    """Return the support of ``b=sigma(a)^dagger``."""
    return tuple(
        sorted(
            index(((-sigma[0] * x) % 4, (-sigma[1] * y) % 5))
            for x, y in (ELEMENTS[item] for item in support_a)
        )
    )


def automorphism_fold(
    sigma: tuple[int, int], translation: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Return the half-preserving affine data permutation ``t sigma``."""
    output = np.empty(NUM_QUBITS, dtype=int)
    for half in range(2):
        for member, (x, y) in enumerate(ELEMENTS):
            target = add(
                ((sigma[0] * x) % 4, (sigma[1] * y) % 5), translation
            )
            output[half * GROUP_ORDER + member] = half * GROUP_ORDER + index(
                target
            )
    return output


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


def find_automorphism_seed(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    sigma: tuple[int, int],
    *,
    rng: random.Random,
    restarts: int,
) -> dict[str, Any] | None:
    """Find a disjoint, nondegenerate C4 sector closed by the ZX fold."""
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
                raise AssertionError("compiled seed failed its kernel constraint")
            for translation in ELEMENTS:
                fold = automorphism_fold(sigma, translation)
                x_orbit = permute_vectors(z_orbit, fold)
                if np.any((matrix_z @ x_orbit.T) % 2):
                    raise AssertionError("automorphism ZX partner left the X kernel")
                pairing = (z_orbit @ x_orbit.T) % 2
                if gf2_rank(pairing) != X_ORDER:
                    continue
                rank_z = gf2_rank(matrix_z)
                rank_gain = gf2_rank(np.vstack([matrix_z, z_orbit])) - rank_z
                if rank_gain != X_ORDER:
                    raise AssertionError("full logical pairing had deficient rank")
                return {
                    "seed_support": list(support),
                    "logical_supports": [
                        np.flatnonzero(row).astype(int).tolist()
                        for row in z_orbit
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


def rowspace_automorphism_dual(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    sigma: tuple[int, int],
) -> bool:
    fold = automorphism_fold(sigma)
    transformed_x = permute_vectors(matrix_x, fold)
    transformed_z = permute_vectors(matrix_z, fold)
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    return bool(
        gf2_rank(np.vstack([matrix_z, transformed_x])) == rank_z
        and gf2_rank(np.vstack([matrix_x, transformed_z])) == rank_x
    )


def search_automorphism_dual(
    *,
    seed_restarts: int,
    random_seed: int,
    maximum_certifications: int = 0,
    maximum_saved_near_misses: int = 100,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    candidate_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    structural_candidates = []
    survivors = []
    near_misses = []
    started = time.perf_counter()
    for sigma in INVOLUTIONS:
        sigma_label = f"{sigma[0]}_{sigma[1]}"
        for weight in range(1, MAXIMUM_POLYNOMIAL_WEIGHT + 1):
            for support_a in itertools.combinations(range(GROUP_ORDER), weight):
                counters["polynomials"] += 1
                counters[f"sigma_{sigma_label}_polynomials"] += 1
                if (
                    progress_callback is not None
                    and counters["polynomials"] % 1000 == 0
                ):
                    progress_callback(
                        {
                            "status": "structural_search",
                            "sigma": list(sigma),
                            "polynomial_weight": weight,
                            "processed": counters["polynomials"],
                            "structural_candidates": len(structural_candidates),
                            "counters": dict(sorted(counters.items())),
                            "seconds": round(time.perf_counter() - started, 6),
                        }
                    )
                support_b = transformed_partner_support(support_a, sigma)
                matrix_x, matrix_z = build_checks(support_a, support_b)
                if np.any((matrix_x @ matrix_z.T) % 2):
                    raise AssertionError("automorphism-dual family failed CSS")
                if not rowspace_automorphism_dual(matrix_x, matrix_z, sigma):
                    raise AssertionError("generator relation failed ZX duality")
                rank_x = gf2_rank(matrix_x)
                rank_z = gf2_rank(matrix_z)
                dimension = NUM_QUBITS - rank_x - rank_z
                counters[f"k_{dimension}"] += 1
                if dimension < X_ORDER:
                    counters["dimension_below_four"] += 1
                    continue
                connected = _tanner_connected(matrix_x, matrix_z)
                counters["connected" if connected else "disconnected"] += 1
                if not connected:
                    continue
                logical = find_automorphism_seed(
                    matrix_x,
                    matrix_z,
                    sigma,
                    rng=rng,
                    restarts=seed_restarts,
                )
                if logical is None:
                    counters["no_paired_logical_seed"] += 1
                    continue
                counters["paired_logical_seed"] += 1
                record = {
                    "sigma": list(sigma),
                    "support_a_indices": list(support_a),
                    "support_b_indices": list(support_b),
                    "support_a": [list(ELEMENTS[item]) for item in support_a],
                    "support_b": [list(ELEMENTS[item]) for item in support_b],
                    "polynomial_weight": weight,
                    "check_weight": 2 * weight,
                    "rank_x": rank_x,
                    "rank_z": rank_z,
                    "k": dimension,
                    "tanner_connected": True,
                    "rowspace_zx_dual": True,
                    "logical_grid": logical,
                    "distance": None,
                }
                structural_candidates.append(record)
                if candidate_callback is not None:
                    candidate_callback(record)
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "structural_search",
                        "sigma": list(sigma),
                        "polynomial_weight_completed": weight,
                        "processed": counters["polynomials"],
                        "structural_candidates": len(structural_candidates),
                        "counters": dict(sorted(counters.items())),
                        "seconds": round(time.perf_counter() - started, 6),
                    }
                )
    limit = maximum_certifications or len(structural_candidates)
    for record in structural_candidates[:limit]:
        matrix_x, matrix_z = build_checks(
            record["support_a_indices"], record["support_b_indices"]
        )
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
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "distance_certification",
                    "processed": counters["polynomials"],
                    "distance_certifications": counters["distance_certifications"],
                    "structural_candidates": len(structural_candidates),
                    "survivors": len(survivors),
                    "counters": dict(sorted(counters.items())),
                    "seconds": round(time.perf_counter() - started, 6),
                }
            )
    return {
        "target": "[[40,k,>=6]] half-preserving ZX-dual C4xC5 BB",
        "method": "exhaustive-a-with-b-equal-sigma-a-dagger",
        "involutions": [list(item) for item in INVOLUTIONS],
        "maximum_polynomial_weight": MAXIMUM_POLYNOMIAL_WEIGHT,
        "maximum_check_weight": 2 * MAXIMUM_POLYNOMIAL_WEIGHT,
        "seed_restarts": seed_restarts,
        "random_seed": random_seed,
        "counters": dict(sorted(counters.items())),
        "structural_candidates": structural_candidates,
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
