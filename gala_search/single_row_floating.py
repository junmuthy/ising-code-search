"""Floating-dimension ``n=32`` search over ``GL(2,2) x C4``.

The code need not encode exactly four qubits.  Instead, it must contain a
four-dimensional, nondegenerate logical ``C4`` sector represented by four
pairwise-disjoint weight-six supports and preserved by an involutive GALA ZX
fold.  Additional logical qubits are retained and analyzed separately.
"""

from __future__ import annotations

import functools
import itertools
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .single_row import (
    COEFFICIENTS_PER_ENTRY,
    NUM_COEFFICIENTS,
    NUM_DATA_BLOCKS,
    NUM_FIBRES,
    NUM_LOGICALS,
    NUM_QUBITS,
    TranslationFold,
    _permute_columns,
    _permute_rows,
    _tanner_connected,
    analyze_logical_grid,
    analyze_seed_fold,
    certify_distance_six,
    checks_from_coefficients,
    gf2_nullspace,
    gf2_rank,
    gf2_rref,
    is_zx_fold,
    iter_structured_gl_folds,
    serialize_coefficients,
    structured_gl_fold,
    translation_orbit,
)
from .abelian_single_row import find_css_logical_of_weight

SCHEMA_VERSION = 1
TARGET_ROW_SIZE = 4
LOGICAL_WEIGHT = 6


def involutive_gl_folds(num_blocks: int = NUM_DATA_BLOCKS) -> tuple[TranslationFold, ...]:
    """Return all structured top-coordinate swaps of order two."""
    folds = []
    for forward in itertools.permutations(range(num_blocks)):
        backward = [0] * num_blocks
        for source, target in enumerate(forward):
            backward[target] = source
        fold = structured_gl_fold(forward, backward)
        permutation = fold.permutation
        if not np.array_equal(permutation[permutation], np.arange(len(permutation))):
            raise AssertionError("constructed GL fold is not involutive")
        folds.append(fold)
    return tuple(folds)


@functools.lru_cache(maxsize=None)
def graph_seed_supports(weight: int = LOGICAL_WEIGHT) -> tuple[tuple[int, ...], ...]:
    """Enumerate disjoint ``C4``-orbit seeds modulo a common translation."""
    supports = []
    for fibres in itertools.combinations(range(NUM_FIBRES), weight):
        for phases_tail in itertools.product(range(4), repeat=weight - 1):
            phases = (0, *phases_tail)
            supports.append(
                tuple(
                    sorted(
                        fibre * 4 + phase
                        for fibre, phase in zip(fibres, phases)
                    )
                )
            )
    return tuple(supports)


def select_seed_fold_witnesses(
    *, witnesses_per_fold: int = 4, random_seed: int = 320806
) -> list[dict[str, Any]]:
    """Select deterministic, diverse full-pairing seeds for every involution."""
    if witnesses_per_fold < 1:
        raise ValueError("witnesses_per_fold must be positive")
    supports = graph_seed_supports()
    witnesses = []
    for fold_index, fold in enumerate(involutive_gl_folds()):
        order = list(range(len(supports)))
        random.Random(random_seed + 1009 * fold_index).shuffle(order)
        permutation_pairing = []
        general_pairing = []
        for support_index in order:
            support = supports[support_index]
            analysis = analyze_seed_fold(support, fold)
            if analysis["zx_pairing_rank"] != TARGET_ROW_SIZE:
                continue
            target = (
                permutation_pairing
                if analysis["pairing_is_permutation"]
                else general_pairing
            )
            target.append(
                {
                    "fold_index": fold_index,
                    "fold": fold.to_dict(),
                    **analysis,
                }
            )
            if len(permutation_pairing) >= witnesses_per_fold:
                break
        selected = permutation_pairing[:witnesses_per_fold]
        if len(selected) < witnesses_per_fold:
            selected.extend(general_pairing[: witnesses_per_fold - len(selected)])
        if len(selected) != witnesses_per_fold:
            raise RuntimeError(
                f"fold {fold_index} supplied only {len(selected)} witnesses"
            )
        witnesses.extend(selected)
    return witnesses


def identity_fold(num_fibres: int) -> TranslationFold:
    """Return the identity translation-compatible ZX fold."""
    return TranslationFold(tuple(range(num_fibres)), (0,) * num_fibres)


def select_self_dual_seed_witnesses(
    *, count: int = 16, random_seed: int = 320807
) -> list[dict[str, Any]]:
    """Select odd, disjoint ``C4`` orbits for the identity ZX fold."""
    if count < 1:
        raise ValueError("count must be positive")
    supports = graph_seed_supports(7)
    order = list(range(1, len(supports)))
    random.Random(random_seed).shuffle(order)
    selected_indices = [0, *order[: count - 1]]
    fold = identity_fold(NUM_FIBRES)
    witnesses = []
    for witness_index, support_index in enumerate(selected_indices):
        analysis = analyze_seed_fold(supports[support_index], fold)
        if analysis["zx_pairing_rank"] != TARGET_ROW_SIZE:
            raise AssertionError("odd disjoint identity-fold orbit is degenerate")
        witnesses.append(
            {
                "witness_index": witness_index,
                "fold": fold.to_dict(),
                **analysis,
            }
        )
    return witnesses


def fold_from_record(record: dict[str, Any]) -> TranslationFold:
    return TranslationFold(
        tuple(map(int, record["fibre_images"])),
        tuple(map(int, record["fibre_shifts"])),
    )


def floating_candidate_analysis(
    coefficients: np.ndarray,
    *,
    support: Sequence[int],
    data_fold: TranslationFold,
    check_fold: TranslationFold,
    require_exact_forward: bool = True,
) -> dict[str, Any]:
    """Analyze one code while allowing every total dimension ``k >= 4``."""
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    dimension = NUM_QUBITS - rank_x - rank_z
    grid = analyze_logical_grid(matrix_x, matrix_z, support, data_fold)
    fold_permutation = data_fold.permutation
    folded_x = _permute_rows(
        _permute_columns(matrix_x, fold_permutation), check_fold.permutation
    )
    pairing = np.asarray(grid["pairing"], dtype=np.uint8)
    checks = {
        "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
        "zx_equal_ranks": rank_x == rank_z,
        "dimension_at_least_four": dimension >= TARGET_ROW_SIZE,
        "involutive_physical_fold": bool(
            np.array_equal(
                fold_permutation[fold_permutation], np.arange(NUM_QUBITS)
            )
        ),
        "reverse_zx_fold": is_zx_fold(matrix_x, matrix_z, fold_permutation),
        "logical_c4_sector": bool(
            grid["pairwise_disjoint"]
            and grid["z_orbit_in_kernel"]
            and grid["x_orbit_in_kernel"]
            and grid["orbit_rank_mod_stabilizers"] == NUM_LOGICALS
            and grid["zx_pairing_rank"] == NUM_LOGICALS
        ),
        "tanner_connected": _tanner_connected(matrix_x, matrix_z),
    }
    if require_exact_forward:
        checks["exact_forward_fold"] = bool(np.array_equal(matrix_z, folded_x))
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    return {
        "n": NUM_QUBITS,
        "k": dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "logical_grid": {
            **grid,
            "pairing_is_permutation": bool(
                np.all(np.count_nonzero(pairing, axis=0) == 1)
                and np.all(np.count_nonzero(pairing, axis=1) == 1)
            ),
            "fold_invariant_nondegenerate_sector": bool(
                checks["involutive_physical_fold"]
                and grid["zx_pairing_rank"] == NUM_LOGICALS
            ),
        },
        "row_weights_x": sorted(set(map(int, row_weights_x))),
        "row_weights_z": sorted(set(map(int, row_weights_z))),
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "coefficient_weight": int(np.count_nonzero(coefficients)),
        "coefficients": np.flatnonzero(coefficients).astype(int).tolist(),
        "entries": serialize_coefficients(coefficients),
        "data_fold": data_fold.to_dict(),
        "check_fold": check_fold.to_dict(),
    }


def search_floating_constraint_space(
    constraints: np.ndarray,
    *,
    support: Sequence[int],
    data_fold: TranslationFold,
    check_fold: TranslationFold,
    maximum_check_weight: int = 12,
    maximum_saved_hits_per_k: int = 20,
) -> dict[str, Any]:
    """Exhaust one exact-fold linear space without fixing the code rank."""
    started = time.perf_counter()
    reduced, pivots = gf2_rref(constraints)
    basis = gf2_nullspace(reduced)
    nullity = len(basis)
    if nullity > 22:
        raise ValueError(
            f"constraint nullity {nullity} is too large for exact enumeration"
        )
    counters: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    dimension_counts: Counter[int] = Counter()
    saved_by_k: Counter[int] = Counter()
    hits = []
    seen_codes: set[bytes] = set()
    for integer in range(1, 1 << nullity):
        counters["generators_tested"] += 1
        selected = [bit for bit in range(nullity) if integer >> bit & 1]
        coefficients = np.bitwise_xor.reduce(basis[selected], axis=0)
        if any(
            not np.any(
                coefficients[
                    entry * COEFFICIENTS_PER_ENTRY : (entry + 1)
                    * COEFFICIENTS_PER_ENTRY
                ]
            )
            for entry in range(4)
        ):
            counters["empty_polynomial_entry"] += 1
            continue
        matrix_x, matrix_z = checks_from_coefficients(coefficients)
        rank_x = gf2_rank(matrix_x)
        rank_z = gf2_rank(matrix_z)
        rank_counts[rank_x] += 1
        dimension = NUM_QUBITS - rank_x - rank_z
        if rank_x != rank_z:
            counters["unequal_zx_rank"] += 1
            continue
        dimension_counts[dimension] += 1
        if dimension < TARGET_ROW_SIZE:
            counters["dimension_below_four"] += 1
            continue
        counters["dimension_at_least_four"] += 1
        check_weight = max(
            np.count_nonzero(matrix_x, axis=1).max(initial=0),
            np.count_nonzero(matrix_z, axis=1).max(initial=0),
        )
        if check_weight > maximum_check_weight:
            counters["check_weight_too_large"] += 1
            continue
        counters["check_weight"] += 1
        if np.any((matrix_x @ matrix_z.T) % 2):
            counters["non_css"] += 1
            continue
        counters["css"] += 1
        if not is_zx_fold(matrix_x, matrix_z, data_fold.permutation):
            counters["invalid_reverse_fold"] += 1
            continue
        analysis = floating_candidate_analysis(
            coefficients,
            support=support,
            data_fold=data_fold,
            check_fold=check_fold,
        )
        if not analysis["accepted_structurally"]:
            for name, passed in analysis["checks"].items():
                if not passed:
                    counters[f"failed_{name}"] += 1
            continue
        counters["structural_hits"] += 1
        if analysis["logical_grid"]["pairing_is_permutation"]:
            counters["permutation_pairing_hits"] += 1
        code_key = np.packbits(
            np.concatenate([matrix_x.ravel(), matrix_z.ravel()])
        ).tobytes()
        if code_key in seen_codes:
            counters["duplicate_code"] += 1
            continue
        seen_codes.add(code_key)
        if saved_by_k[dimension] >= maximum_saved_hits_per_k:
            counters["unsaved_hit"] += 1
            continue
        saved_by_k[dimension] += 1
        hits.append(analysis)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "hit" if hits else "searched_without_structural_hit",
        "constraint_rank": len(pivots),
        "constraint_nullity": nullity,
        "method": "exhaustive",
        "planned": (1 << nullity) - 1,
        "counters": dict(sorted(counters.items())),
        "rank_counts": dict(sorted(rank_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "hits": hits,
        "seconds": round(time.perf_counter() - started, 6),
    }


def certify_floating_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct and exactly certify one saved structural candidate."""
    coefficients = np.zeros(NUM_COEFFICIENTS, dtype=np.uint8)
    coefficients[record["coefficients"]] = 1
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    distance = certify_distance_six(matrix_x, matrix_z)
    seed_weight = int(record["logical_grid"]["weight"])
    weight_six_z = None
    weight_six_x = None
    if distance["certified_distance_at_least_six"] and seed_weight > 6:
        weight_six_z = find_css_logical_of_weight(matrix_x, matrix_z, 6)
        weight_six_x = find_css_logical_of_weight(matrix_z, matrix_x, 6)
    if not distance["certified_distance_at_least_six"]:
        exact_distance = min(
            item["weight"]
            for item in (
                distance["z_logical_below_six"],
                distance["x_logical_below_six"],
            )
            if item is not None
        )
    elif weight_six_z is not None or weight_six_x is not None:
        exact_distance = 6
    else:
        exact_distance = seed_weight
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": record,
        "distance": distance,
        "weight_six_z_logical": weight_six_z,
        "weight_six_x_logical": weight_six_x,
        "certified_distance": exact_distance,
        "accepted": distance["certified_distance_at_least_six"],
    }


def find_second_c4_sector(record: dict[str, Any]) -> dict[str, Any] | None:
    """Find another internally disjoint row forming a clean logical complement."""
    coefficients = np.zeros(NUM_COEFFICIENTS, dtype=np.uint8)
    coefficients[record["coefficients"]] = 1
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    fold = fold_from_record(record["data_fold"])
    primary_support = tuple(record["logical_grid"]["seed_support"])
    primary_seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    primary_seed[list(primary_support)] = 1
    primary_z = translation_orbit(primary_seed)
    primary_x = _permute_columns(primary_z, fold.permutation)
    primary_weight = len(primary_support)
    for support in graph_seed_supports(primary_weight):
        if support == primary_support:
            continue
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbit = translation_orbit(seed)
        if np.any((matrix_x @ z_orbit.T) % 2):
            continue
        x_orbit = _permute_columns(z_orbit, fold.permutation)
        if np.any((matrix_z @ x_orbit.T) % 2):
            continue
        combined_z = np.vstack([primary_z, z_orbit])
        combined_x = np.vstack([primary_x, x_orbit])
        rank_gain = gf2_rank(np.vstack([matrix_z, combined_z])) - gf2_rank(matrix_z)
        if rank_gain != 8:
            continue
        pairing = (combined_z @ combined_x.T) % 2
        if gf2_rank(pairing) != 8:
            continue
        cross_pairing_zero = bool(
            not np.any((primary_z @ x_orbit.T) % 2)
            and not np.any((z_orbit @ primary_x.T) % 2)
        )
        support_overlap = np.sum(primary_z, axis=0) + np.sum(z_orbit, axis=0)
        return {
            "seed_support": list(support),
            "logical_supports": [
                np.flatnonzero(row).astype(int).tolist() for row in z_orbit
            ],
            "internally_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
            "overlaps_primary_row": bool(np.any(support_overlap > 1)),
            "combined_rank_mod_stabilizers": rank_gain,
            "combined_zx_pairing_rank": gf2_rank(pairing),
            "cross_pairing_zero": cross_pairing_zero,
            "combined_pairing": pairing.astype(int).tolist(),
        }
    return None


def check_folds() -> tuple[TranslationFold, ...]:
    return tuple(iter_structured_gl_folds(2))
