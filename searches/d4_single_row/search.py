"""Identity-ZX-fold ``D4 x C4`` two-block search at ``n=32``."""

from __future__ import annotations

import functools
import itertools
import random
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from gala_search.single_row import (
    _permute_columns,
    _permute_rows,
    _tanner_connected,
    certify_distance_six,
    gf2_nullspace,
    gf2_rank,
    gf2_rref,
    is_zx_fold,
)

SCHEMA_VERSION = 1
CYCLE_ORDER = 4
TOP_DIMENSION = 4
BLOCK_SIZE = TOP_DIMENSION * CYCLE_ORDER
NUM_HALVES = 2
NUM_QUBITS = NUM_HALVES * BLOCK_SIZE
NUM_CHECKS = BLOCK_SIZE
NUM_FIBRES = NUM_HALVES * TOP_DIMENSION
NUM_LOGICALS = CYCLE_ORDER
TARGET_ROW_SIZE = CYCLE_ORDER
ENTRY_NAMES = ("A", "B")


def permutation_matrix(images: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((len(images), len(images)), dtype=np.uint8)
    matrix[np.asarray(images, dtype=int), np.arange(len(images))] = 1
    return matrix


D4_ROTATION = permutation_matrix(((1, 2, 3, 0)))
D4_REFLECTION = permutation_matrix(((0, 3, 2, 1)))
IDENTITY = np.eye(TOP_DIMENSION, dtype=np.uint8)


def d4_group_matrices() -> tuple[np.ndarray, ...]:
    output = []
    for reflection_power in range(2):
        for rotation_power in range(4):
            matrix = np.linalg.matrix_power(D4_ROTATION, rotation_power) % 2
            if reflection_power:
                matrix = (D4_REFLECTION @ matrix) % 2
            if not any(np.array_equal(matrix, item) for item in output):
                output.append(matrix.astype(np.uint8))
    return tuple(output)


def represented_algebra_basis() -> tuple[np.ndarray, ...]:
    basis = []
    rank = 0
    for matrix in d4_group_matrices():
        proposed = np.asarray([item.ravel() for item in (*basis, matrix)])
        proposed_rank = gf2_rank(proposed)
        if proposed_rank > rank:
            basis.append(matrix)
            rank = proposed_rank
    return tuple(basis)


TOP_MATRIX_BASIS = represented_algebra_basis()
COEFFICIENTS_PER_ENTRY = len(TOP_MATRIX_BASIS) * CYCLE_ORDER
NUM_COEFFICIENTS = len(ENTRY_NAMES) * COEFFICIENTS_PER_ENTRY


@functools.lru_cache(maxsize=None)
def cyclic_shift_matrix(shift: int) -> np.ndarray:
    matrix = np.zeros((CYCLE_ORDER, CYCLE_ORDER), dtype=np.uint8)
    for source in range(CYCLE_ORDER):
        matrix[(source + shift) % CYCLE_ORDER, source] = 1
    return matrix


@functools.lru_cache(maxsize=None)
def coefficient_lift(top: int, shift: int) -> np.ndarray:
    return np.kron(TOP_MATRIX_BASIS[top], cyclic_shift_matrix(shift)).astype(
        np.uint8
    )


@functools.lru_cache(maxsize=1)
def coefficient_checks() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    checks = []
    for entry in range(len(ENTRY_NAMES)):
        for top in range(len(TOP_MATRIX_BASIS)):
            for shift in range(CYCLE_ORDER):
                lift = coefficient_lift(top, shift)
                matrix_x = np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
                matrix_z = np.zeros_like(matrix_x)
                if entry == 0:
                    matrix_x[:, :BLOCK_SIZE] = lift
                    matrix_z[:, BLOCK_SIZE:] = lift.T
                else:
                    matrix_x[:, BLOCK_SIZE:] = lift
                    matrix_z[:, :BLOCK_SIZE] = lift.T
                checks.append((matrix_x, matrix_z))
    return tuple(checks)


def checks_from_coefficients(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(coefficients)
    if not len(selected):
        zero = np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
        return zero, zero.copy()
    matrix_x = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][0] for index in selected]), axis=0
    )
    matrix_z = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][1] for index in selected]), axis=0
    )
    return matrix_x, matrix_z


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    rows = []
    current = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(CYCLE_ORDER):
        rows.append(current)
        shifted = np.zeros_like(current)
        for fibre in range(NUM_FIBRES):
            start = fibre * CYCLE_ORDER
            shifted[start : start + CYCLE_ORDER] = np.roll(
                current[start : start + CYCLE_ORDER], 1
            )
        current = shifted
    return np.asarray(rows, dtype=np.uint8)


def canonical_seed_support(support: Sequence[int]) -> tuple[int, ...]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in translation_orbit(seed)
    )


def select_seed_witnesses(
    *, count: int, random_seed: int, weight: int = 7
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    seen: set[tuple[int, ...]] = set()
    witnesses = []
    while len(witnesses) < count:
        fibres = rng.sample(range(NUM_FIBRES), weight)
        support = canonical_seed_support(
            [fibre * CYCLE_ORDER + rng.randrange(CYCLE_ORDER) for fibre in fibres]
        )
        if support in seen:
            continue
        seen.add(support)
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        orbit = translation_orbit(seed)
        pairing = (orbit @ orbit.T) % 2
        if not np.array_equal(pairing, np.eye(NUM_LOGICALS, dtype=np.uint8)):
            raise AssertionError("odd disjoint seed did not identity-pair")
        witnesses.append(
            {
                "witness_index": len(witnesses),
                "seed_support": list(support),
                "weight": weight,
                "pairwise_disjoint": True,
                "zx_pairing": pairing.astype(int).tolist(),
                "zx_pairing_rank": NUM_LOGICALS,
            }
        )
    return witnesses


def matrix_permutation(matrix: np.ndarray) -> tuple[int, ...]:
    """Return the coordinate permutation represented by a permutation matrix."""
    return tuple(np.argmax(matrix, axis=0).astype(int).tolist())


def d4_involution_permutations() -> tuple[tuple[int, ...], ...]:
    """Return identity, the half-turn, and the four vertex reflections."""
    output = []
    for matrix in d4_group_matrices():
        if np.array_equal((matrix @ matrix) % 2, IDENTITY):
            output.append(matrix_permutation(matrix))
    return tuple(sorted(output))


def lifted_top_permutation(
    top_permutation: Sequence[int], *, blocks: int
) -> np.ndarray:
    """Repeat one top-coordinate permutation over blocks and C4 positions."""
    output = np.empty(blocks * TOP_DIMENSION * CYCLE_ORDER, dtype=int)
    for block in range(blocks):
        for top, target_top in enumerate(top_permutation):
            for position in range(CYCLE_ORDER):
                source = (block * TOP_DIMENSION + top) * CYCLE_ORDER + position
                target = (
                    (block * TOP_DIMENSION + int(target_top)) * CYCLE_ORDER
                    + position
                )
                output[source] = target
    return output


def select_fold_seed_witnesses(
    *, witnesses_per_data_fold: int, random_seed: int, weight: int = 7
) -> list[dict[str, Any]]:
    """Select full-pairing graph seeds for every involutive D4 data fold."""
    rng = random.Random(random_seed)
    witnesses = []
    for data_fold_index, top_permutation in enumerate(d4_involution_permutations()):
        # A fixed-point-free involution pairs every top fibre with a distinct
        # mate. Every graph-seed overlap is therefore doubled over GF(2), so
        # the induced logical pairing vanishes identically.
        if not any(source == target for source, target in enumerate(top_permutation)):
            continue
        data_permutation = lifted_top_permutation(top_permutation, blocks=NUM_HALVES)
        seen: set[tuple[int, ...]] = set()
        while len(seen) < witnesses_per_data_fold:
            fibres = rng.sample(range(NUM_FIBRES), weight)
            support = canonical_seed_support(
                [
                    fibre * CYCLE_ORDER + rng.randrange(CYCLE_ORDER)
                    for fibre in fibres
                ]
            )
            if support in seen:
                continue
            seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
            seed[list(support)] = 1
            z_orbit = translation_orbit(seed)
            x_orbit = _permute_columns(z_orbit, data_permutation)
            pairing = (z_orbit @ x_orbit.T) % 2
            if gf2_rank(pairing) != NUM_LOGICALS:
                continue
            seen.add(support)
            witnesses.append(
                {
                    "data_fold_index": data_fold_index,
                    "data_top_permutation": list(top_permutation),
                    "seed_support": list(support),
                    "weight": weight,
                    "pairwise_disjoint": True,
                    "zx_pairing": pairing.astype(int).tolist(),
                    "zx_pairing_rank": gf2_rank(pairing),
                }
            )
    return witnesses


def identity_fold_constraints(support: Sequence[int]) -> np.ndarray:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    columns = []
    for matrix_x, matrix_z in coefficient_checks():
        columns.append(
            np.concatenate([(matrix_x @ seed) % 2, (matrix_z ^ matrix_x).ravel()])
        )
    return np.asarray(columns, dtype=np.uint8).T


def exact_fold_constraints(
    support: Sequence[int],
    *,
    data_top_permutation: Sequence[int],
    check_top_permutation: Sequence[int],
) -> np.ndarray:
    """Impose the seed kernel and one exact nonidentity displayed ZX fold."""
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    data_permutation = lifted_top_permutation(
        data_top_permutation, blocks=NUM_HALVES
    )
    check_permutation = lifted_top_permutation(check_top_permutation, blocks=1)
    columns = []
    for matrix_x, matrix_z in coefficient_checks():
        folded_x = _permute_rows(
            _permute_columns(matrix_x, data_permutation), check_permutation
        )
        columns.append(
            np.concatenate([(matrix_x @ seed) % 2, (matrix_z ^ folded_x).ravel()])
        )
    return np.asarray(columns, dtype=np.uint8).T


def serialize_coefficients(coefficients: np.ndarray) -> dict[str, Any]:
    entries: dict[str, list[dict[str, int]]] = {name: [] for name in ENTRY_NAMES}
    for coefficient in np.flatnonzero(coefficients):
        entry, local = divmod(int(coefficient), COEFFICIENTS_PER_ENTRY)
        top, shift = divmod(local, CYCLE_ORDER)
        entries[ENTRY_NAMES[entry]].append({"top_basis": top, "x": shift})
    return entries


def analyze_candidate(
    coefficients: np.ndarray, support: Sequence[int]
) -> dict[str, Any]:
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    dimension = NUM_QUBITS - rank_x - rank_z
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = translation_orbit(seed)
    pairing = (orbit @ orbit.T) % 2
    rank_gain = gf2_rank(np.vstack([matrix_z, orbit])) - rank_z
    checks = {
        "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
        "identity_zx_fold": bool(np.array_equal(matrix_x, matrix_z)),
        "dimension_at_least_four": dimension >= TARGET_ROW_SIZE,
        "logical_c4_sector": bool(
            np.all(np.sum(orbit, axis=0) <= 1)
            and not np.any((matrix_x @ orbit.T) % 2)
            and not np.any((matrix_z @ orbit.T) % 2)
            and rank_gain == NUM_LOGICALS
            and gf2_rank(pairing) == NUM_LOGICALS
        ),
        "tanner_connected": _tanner_connected(matrix_x, matrix_z),
    }


def analyze_fold_candidate(
    coefficients: np.ndarray,
    support: Sequence[int],
    *,
    data_top_permutation: Sequence[int],
    check_top_permutation: Sequence[int],
) -> dict[str, Any]:
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    dimension = NUM_QUBITS - rank_x - rank_z
    data_permutation = lifted_top_permutation(
        data_top_permutation, blocks=NUM_HALVES
    )
    check_permutation = lifted_top_permutation(check_top_permutation, blocks=1)
    folded_x = _permute_rows(
        _permute_columns(matrix_x, data_permutation), check_permutation
    )
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    z_orbit = translation_orbit(seed)
    x_orbit = _permute_columns(z_orbit, data_permutation)
    pairing = (z_orbit @ x_orbit.T) % 2
    rank_gain = gf2_rank(np.vstack([matrix_z, z_orbit])) - rank_z
    checks = {
        "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
        "exact_forward_zx_fold": bool(np.array_equal(matrix_z, folded_x)),
        "reverse_zx_fold": is_zx_fold(matrix_x, matrix_z, data_permutation),
        "dimension_at_least_four": dimension >= TARGET_ROW_SIZE,
        "logical_c4_sector": bool(
            np.all(np.sum(z_orbit, axis=0) <= 1)
            and not np.any((matrix_x @ z_orbit.T) % 2)
            and not np.any((matrix_z @ x_orbit.T) % 2)
            and rank_gain == NUM_LOGICALS
            and gf2_rank(pairing) == NUM_LOGICALS
        ),
        "tanner_connected": _tanner_connected(matrix_x, matrix_z),
    }
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "n": NUM_QUBITS,
        "k": dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "logical_grid": {
            "seed_support": list(map(int, support)),
            "weight": len(support),
            "logical_supports": [
                np.flatnonzero(row).astype(int).tolist() for row in z_orbit
            ],
            "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
            "orbit_rank_mod_stabilizers": rank_gain,
            "zx_pairing": pairing.astype(int).tolist(),
            "zx_pairing_rank": gf2_rank(pairing),
        },
        "data_top_permutation": list(map(int, data_top_permutation)),
        "check_top_permutation": list(map(int, check_top_permutation)),
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "row_weights_x": sorted(set(map(int, row_weights_x))),
        "row_weights_z": sorted(set(map(int, row_weights_z))),
        "coefficient_weight": int(np.count_nonzero(coefficients)),
        "coefficients": np.flatnonzero(coefficients).astype(int).tolist(),
        "entries": serialize_coefficients(coefficients),
    }
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "n": NUM_QUBITS,
        "k": dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "logical_grid": {
            "seed_support": list(map(int, support)),
            "weight": len(support),
            "logical_supports": [
                np.flatnonzero(row).astype(int).tolist() for row in orbit
            ],
            "pairwise_disjoint": bool(np.all(np.sum(orbit, axis=0) <= 1)),
            "orbit_rank_mod_stabilizers": rank_gain,
            "zx_pairing": pairing.astype(int).tolist(),
            "zx_pairing_rank": gf2_rank(pairing),
        },
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "row_weights_x": sorted(set(map(int, row_weights_x))),
        "row_weights_z": sorted(set(map(int, row_weights_z))),
        "coefficient_weight": int(np.count_nonzero(coefficients)),
        "coefficients": np.flatnonzero(coefficients).astype(int).tolist(),
        "entries": serialize_coefficients(coefficients),
    }


def search_constraint_space(
    support: Sequence[int],
    *,
    maximum_check_weight: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    constraints = identity_fold_constraints(support)
    reduced, pivots = gf2_rref(constraints)
    basis = gf2_nullspace(reduced)
    nullity = len(basis)
    if nullity > 22:
        raise ValueError(f"constraint nullity {nullity} exceeds exact-search limit")
    counters: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    dimension_counts: Counter[int] = Counter()
    survivors = []
    near_misses = []
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
            for entry in range(len(ENTRY_NAMES))
        ):
            counters["empty_polynomial_entry"] += 1
            continue
        matrix_x, matrix_z = checks_from_coefficients(coefficients)
        rank_x = gf2_rank(matrix_x)
        rank_z = gf2_rank(matrix_z)
        rank_counts[rank_x] += 1
        if rank_x != rank_z:
            counters["unequal_zx_rank"] += 1
            continue
        dimension = NUM_QUBITS - rank_x - rank_z
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
        counters["check_weight_accepted"] += 1
        if np.any((matrix_x @ matrix_z.T) % 2):
            counters["non_css"] += 1
            continue
        counters["css"] += 1
        analysis = analyze_candidate(coefficients, support)
        if not analysis["accepted_structurally"]:
            for name, passed in analysis["checks"].items():
                if not passed:
                    counters[f"failed_{name}"] += 1
            continue
        code_key = np.packbits(
            np.concatenate([matrix_x.ravel(), matrix_z.ravel()])
        ).tobytes()
        if code_key in seen_codes:
            counters["duplicate_code"] += 1
            continue
        seen_codes.add(code_key)
        counters["structural_candidates"] += 1
        distance = certify_distance_six(matrix_x, matrix_z)
        if distance["certified_distance_at_least_six"]:
            counters["distance_at_least_six"] += 1
            survivors.append({**analysis, "distance": distance})
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
                near_misses.append({**analysis, "distance": distance})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if survivors else "searched_without_distance_hit",
        "constraint_rank": len(pivots),
        "constraint_nullity": nullity,
        "method": "exhaustive",
        "planned": (1 << nullity) - 1,
        "counters": dict(sorted(counters.items())),
        "rank_counts": dict(sorted(rank_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }


def search_exact_fold_space(
    support: Sequence[int],
    *,
    data_top_permutation: Sequence[int],
    check_top_permutation: Sequence[int],
    maximum_check_weight: int,
    maximum_saved_near_misses: int,
    maximum_nullity: int = 22,
) -> dict[str, Any]:
    """Exhaust one D4 data/check-fold constraint space."""
    started = time.perf_counter()
    constraints = exact_fold_constraints(
        support,
        data_top_permutation=data_top_permutation,
        check_top_permutation=check_top_permutation,
    )
    reduced, pivots = gf2_rref(constraints)
    basis = gf2_nullspace(reduced)
    nullity = len(basis)
    common = {
        "schema_version": SCHEMA_VERSION,
        "constraint_rank": len(pivots),
        "constraint_nullity": nullity,
        "planned": (1 << nullity) - 1,
        "data_top_permutation": list(map(int, data_top_permutation)),
        "check_top_permutation": list(map(int, check_top_permutation)),
    }
    if nullity > maximum_nullity:
        return {
            **common,
            "status": "nullity_above_exact_limit",
            "counters": {},
            "rank_counts": {},
            "dimension_counts": {},
            "survivors": [],
            "near_misses": [],
            "seconds": round(time.perf_counter() - started, 6),
        }
    counters: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    dimension_counts: Counter[int] = Counter()
    survivors = []
    near_misses = []
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
            for entry in range(len(ENTRY_NAMES))
        ):
            counters["empty_polynomial_entry"] += 1
            continue
        matrix_x, matrix_z = checks_from_coefficients(coefficients)
        rank_x = gf2_rank(matrix_x)
        rank_z = gf2_rank(matrix_z)
        rank_counts[rank_x] += 1
        if rank_x != rank_z:
            counters["unequal_zx_rank"] += 1
            continue
        dimension = NUM_QUBITS - rank_x - rank_z
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
        counters["check_weight_accepted"] += 1
        if np.any((matrix_x @ matrix_z.T) % 2):
            counters["non_css"] += 1
            continue
        counters["css"] += 1
        analysis = analyze_fold_candidate(
            coefficients,
            support,
            data_top_permutation=data_top_permutation,
            check_top_permutation=check_top_permutation,
        )
        if not analysis["accepted_structurally"]:
            for name, passed in analysis["checks"].items():
                if not passed:
                    counters[f"failed_{name}"] += 1
            continue
        code_key = np.packbits(
            np.concatenate([matrix_x.ravel(), matrix_z.ravel()])
        ).tobytes()
        if code_key in seen_codes:
            counters["duplicate_code"] += 1
            continue
        seen_codes.add(code_key)
        counters["structural_candidates"] += 1
        distance = certify_distance_six(matrix_x, matrix_z)
        if distance["certified_distance_at_least_six"]:
            counters["distance_at_least_six"] += 1
            survivors.append({**analysis, "distance": distance})
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
                near_misses.append({**analysis, "distance": distance})
    return {
        **common,
        "status": "accepted" if survivors else "searched_without_distance_hit",
        "method": "exhaustive",
        "counters": dict(sorted(counters.items())),
        "rank_counts": dict(sorted(rank_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "survivors": survivors,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }
