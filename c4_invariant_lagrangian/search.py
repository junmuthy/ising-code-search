"""Random C4-invariant Lagrangian search for ``[[28,4,>=6]]`` codes."""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Any

import numpy as np

from gala_search.single_row import _tanner_connected, gf2_nullspace, gf2_rank, gf2_rref

SCHEMA_VERSION = 1
LOGICAL_ORDER = 4
THICKNESS = 7
NUM_QUBITS = LOGICAL_ORDER * THICKNESS
TARGET_STABILIZER_RANK = (NUM_QUBITS - LOGICAL_ORDER) // 2
DEFAULT_MAXIMUM_CHECK_WEIGHT = 12


def coordinate(logical: int, thickness: int) -> int:
    return (logical % LOGICAL_ORDER) * THICKNESS + thickness % THICKNESS


def logical_columns() -> np.ndarray:
    output = np.zeros((LOGICAL_ORDER, NUM_QUBITS), dtype=np.uint8)
    for logical in range(LOGICAL_ORDER):
        output[logical, logical * THICKNESS : (logical + 1) * THICKNESS] = 1
    return output


def translate_vector(vector: np.ndarray, shift: int = 1) -> np.ndarray:
    reshaped = np.asarray(vector, dtype=np.uint8).reshape(LOGICAL_ORDER, THICKNESS)
    return np.roll(reshaped, shift=shift, axis=0).reshape(NUM_QUBITS)


def translation_orbit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [translate_vector(vector, shift) for shift in range(LOGICAL_ORDER)],
        dtype=np.uint8,
    )


def nilpotent_translation_power(power: int) -> np.ndarray:
    """Matrix for ``(T + I)^power`` under the physical C4 translation."""
    if not 0 <= power <= LOGICAL_ORDER:
        raise ValueError("power must be between zero and four")
    operator = np.eye(NUM_QUBITS, dtype=np.uint8)
    nilpotent = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    for column in range(NUM_QUBITS):
        unit = np.zeros(NUM_QUBITS, dtype=np.uint8)
        unit[column] = 1
        nilpotent[:, column] = translate_vector(unit) ^ unit
    result = operator
    for _factor in range(power):
        result = (nilpotent @ result) % 2
    return result.astype(np.uint8)


def packed(vector: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(vector))


def canonical_rowspace(matrix: np.ndarray) -> np.ndarray:
    reduced, _pivots = gf2_rref(matrix)
    return reduced


def stabilizer_sums(stabilizer: np.ndarray) -> tuple[int, ...]:
    basis = [packed(row) for row in canonical_rowspace(stabilizer)]
    sums = [0]
    for row in basis:
        sums.extend(value ^ row for value in tuple(sums))
    return tuple(sums)


def vector_from_mask(mask: int) -> np.ndarray:
    return np.fromiter(
        ((mask >> qubit) & 1 for qubit in range(NUM_QUBITS)),
        dtype=np.uint8,
        count=NUM_QUBITS,
    )


def minimum_weight_basis(stabilizer: np.ndarray) -> np.ndarray:
    """Return a greedy minimum-weight basis of the stabilizer rowspace."""
    candidates = sorted(
        (mask.bit_count(), mask)
        for mask in stabilizer_sums(stabilizer)
        if mask
    )
    packed_basis: dict[int, int] = {}
    selected = []
    for _weight, candidate in candidates:
        reduced = candidate
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in packed_basis:
                packed_basis[pivot] = reduced
                selected.append(vector_from_mask(candidate))
                break
            reduced ^= packed_basis[pivot]
        if len(selected) == TARGET_STABILIZER_RANK:
            break
    if len(selected) != TARGET_STABILIZER_RANK:
        raise AssertionError("failed to extract a full stabilizer basis")
    return np.asarray(selected, dtype=np.uint8)


def exact_logical_distance(stabilizer: np.ndarray) -> dict[str, Any]:
    """Minimize weight over all 15 nonzero canonical logical cosets."""
    stabilizers = stabilizer_sums(stabilizer)
    logical_masks = [packed(row) for row in logical_columns()]
    best_weight = NUM_QUBITS + 1
    best_logical = 0
    best_stabilizer = 0
    best_operator = 0
    weight_counts: Counter[int] = Counter()
    for logical_bits in range(1, 1 << LOGICAL_ORDER):
        logical = 0
        for bit, mask in enumerate(logical_masks):
            if logical_bits >> bit & 1:
                logical ^= mask
        for stabilizer_mask in stabilizers:
            operator = logical ^ stabilizer_mask
            weight = operator.bit_count()
            if weight <= 7:
                weight_counts[weight] += 1
            if weight < best_weight:
                best_weight = weight
                best_logical = logical_bits
                best_stabilizer = stabilizer_mask
                best_operator = operator
    return {
        "distance": best_weight,
        "logical_combination": best_logical,
        "stabilizer_mask": best_stabilizer,
        "operator_support": [
            qubit for qubit in range(NUM_QUBITS) if best_operator >> qubit & 1
        ],
        "cosets_checked": (1 << LOGICAL_ORDER) - 1,
        "stabilizers_per_coset": len(stabilizers),
        "weight_counts_through_seven": {
            str(weight): weight_counts[weight] for weight in range(1, 8)
        },
    }


def random_nullspace_vector(
    basis: np.ndarray, rng: random.Random
) -> np.ndarray:
    bits = rng.getrandbits(len(basis))
    if bits == 0:
        return np.zeros(basis.shape[1], dtype=np.uint8)
    selected = [index for index in range(len(basis)) if bits >> index & 1]
    return np.bitwise_xor.reduce(basis[selected], axis=0)


def extend_by_free_orbit(
    stabilizer: np.ndarray,
    *,
    rng: random.Random,
    maximum_check_weight: int,
    attempts: int,
) -> np.ndarray | None:
    """Add one rank-four, self-orthogonal C4 orbit to ``stabilizer``."""
    constraints = np.vstack([logical_columns(), stabilizer])
    nullspace = gf2_nullspace(constraints)
    current_rank = gf2_rank(stabilizer)
    for _attempt in range(attempts):
        vector = random_nullspace_vector(nullspace, rng)
        weight = int(np.count_nonzero(vector))
        if not 2 <= weight <= maximum_check_weight:
            continue
        orbit = translation_orbit(vector)
        if gf2_rank(orbit) != LOGICAL_ORDER:
            continue
        if np.any((orbit @ orbit.T) % 2):
            continue
        extended = np.vstack([stabilizer, orbit])
        if gf2_rank(extended) != current_rank + LOGICAL_ORDER:
            continue
        return canonical_rowspace(extended)
    return None


def extend_by_orbit_rank(
    stabilizer: np.ndarray,
    *,
    orbit_rank: int,
    rng: random.Random,
    maximum_check_weight: int,
    attempts: int,
) -> np.ndarray | None:
    """Add an isotropic cyclic C4 submodule of the requested rank."""
    if not 1 <= orbit_rank <= LOGICAL_ORDER:
        raise ValueError("orbit_rank must be in 1..4")
    constraints = [logical_columns(), stabilizer]
    if orbit_rank < LOGICAL_ORDER:
        constraints.append(nilpotent_translation_power(orbit_rank))
    nullspace = gf2_nullspace(np.vstack(constraints))
    current_rank = gf2_rank(stabilizer)
    for _attempt in range(attempts):
        vector = random_nullspace_vector(nullspace, rng)
        weight = int(np.count_nonzero(vector))
        if not 2 <= weight <= maximum_check_weight:
            continue
        orbit = translation_orbit(vector)
        if gf2_rank(orbit) != orbit_rank:
            continue
        if np.any((orbit @ orbit.T) % 2):
            continue
        extended = np.vstack([stabilizer, orbit])
        if gf2_rank(extended) != current_rank + orbit_rank:
            continue
        return canonical_rowspace(extended)
    return None


def module_partitions(
    total: int = TARGET_STABILIZER_RANK,
    maximum_part: int = LOGICAL_ORDER,
) -> tuple[tuple[int, ...], ...]:
    """Integer partitions representing possible cyclic C4 module ranks."""
    output: list[tuple[int, ...]] = []

    def visit(remaining: int, largest: int, prefix: tuple[int, ...]) -> None:
        if remaining == 0:
            output.append(prefix)
            return
        for part in range(min(largest, maximum_part, remaining), 0, -1):
            visit(remaining - part, part, prefix + (part,))

    visit(total, maximum_part, ())
    return tuple(output)


def construct_module_type_candidate(
    module_type: tuple[int, ...],
    *,
    rng: random.Random,
    maximum_check_weight: int = DEFAULT_MAXIMUM_CHECK_WEIGHT,
    attempts_per_orbit: int = 400,
) -> np.ndarray | None:
    """Construct a Lagrangian with a prescribed cyclic-orbit rank profile."""
    if sum(module_type) != TARGET_STABILIZER_RANK:
        raise ValueError("module-type ranks must sum to twelve")
    stabilizer = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    # Smaller constrained orbits are substantially harder to add late, so add
    # them first even though module types are displayed in descending order.
    for orbit_rank in reversed(module_type):
        extended = extend_by_orbit_rank(
            stabilizer,
            orbit_rank=orbit_rank,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if extended is None:
            return None
        stabilizer = extended
    if gf2_rank(stabilizer) != TARGET_STABILIZER_RANK:
        raise AssertionError("constructed stabilizer has the wrong rank")
    return stabilizer


def construct_candidate(
    *,
    rng: random.Random,
    maximum_check_weight: int = DEFAULT_MAXIMUM_CHECK_WEIGHT,
    attempts_per_orbit: int = 200,
) -> np.ndarray | None:
    stabilizer = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    while gf2_rank(stabilizer) < TARGET_STABILIZER_RANK:
        extended = extend_by_free_orbit(
            stabilizer,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if extended is None:
            return None
        stabilizer = extended
    if gf2_rank(stabilizer) != TARGET_STABILIZER_RANK:
        raise AssertionError("constructed stabilizer has the wrong rank")
    return stabilizer


def decompose_free_orbits(stabilizer: np.ndarray) -> tuple[np.ndarray, ...]:
    """Decompose a free rank-three C4 module into three rank-four orbits."""
    chosen_rows = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    generators = []
    candidates = sorted(
        stabilizer_sums(stabilizer), key=lambda mask: (mask.bit_count(), mask)
    )
    for mask in candidates:
        if not mask:
            continue
        vector = vector_from_mask(mask)
        orbit = translation_orbit(vector)
        if gf2_rank(orbit) != LOGICAL_ORDER:
            continue
        current_rank = gf2_rank(chosen_rows)
        if gf2_rank(np.vstack([chosen_rows, orbit])) != current_rank + LOGICAL_ORDER:
            continue
        generators.append(vector)
        chosen_rows = canonical_rowspace(np.vstack([chosen_rows, orbit]))
        if gf2_rank(chosen_rows) == TARGET_STABILIZER_RANK:
            return tuple(generators)
    raise ValueError("stabilizer is not a free rank-three F2[C4] module")


def analyze_candidate(stabilizer: np.ndarray) -> dict[str, Any]:
    logicals = logical_columns()
    rank = gf2_rank(stabilizer)
    distance = exact_logical_distance(stabilizer)
    displayed_basis = minimum_weight_basis(stabilizer)
    row_weights = np.count_nonzero(displayed_basis, axis=1)
    translated = np.asarray(
        [translate_vector(row) for row in stabilizer], dtype=np.uint8
    )
    checks = {
        "rank_twelve": rank == TARGET_STABILIZER_RANK,
        "self_orthogonal": bool(not np.any((stabilizer @ stabilizer.T) % 2)),
        "logical_columns_in_kernel": bool(
            not np.any((stabilizer @ logicals.T) % 2)
        ),
        "c4_invariant_rowspace": gf2_rank(np.vstack([stabilizer, translated]))
        == rank,
        "logical_pairing_identity": bool(
            np.array_equal((logicals @ logicals.T) % 2, np.eye(4, dtype=np.uint8))
        ),
        "tanner_connected": _tanner_connected(stabilizer, stabilizer),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "n": NUM_QUBITS,
        "k": NUM_QUBITS - 2 * rank,
        "rank_x": rank,
        "rank_z": rank,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "distance": distance,
        "certified_distance": distance["distance"],
        "accepted": all(checks.values()) and distance["distance"] >= 6,
        "maximum_check_weight": int(row_weights.max(initial=0)),
        "row_weights": row_weights.astype(int).tolist(),
        "stabilizer_rows": [
            np.flatnonzero(row).astype(int).tolist() for row in displayed_basis
        ],
        "stabilizer_masks": [packed(row) for row in canonical_rowspace(stabilizer)],
        "logical_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in logicals
        ],
        "logical_translation": "(x,y) -> (x+1 mod 4,y)",
        "zx_pairing": ((logicals @ logicals.T) % 2).astype(int).tolist(),
    }


def refinement_score(analysis: dict[str, Any]) -> tuple[int, int, int]:
    distance = int(analysis["certified_distance"])
    counts = analysis["distance"]["weight_counts_through_seven"]
    at_distance = int(counts.get(str(distance), 0))
    below_six = sum(int(counts.get(str(weight), 0)) for weight in range(1, 6))
    return distance, -at_distance, -below_six


def locally_refine(
    stabilizer: np.ndarray,
    *,
    rng: random.Random,
    iterations: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    """Replace one translated generator orbit at a time to remove low logicals."""
    current = canonical_rowspace(stabilizer)
    current_analysis = analyze_candidate(current)
    current_score = refinement_score(current_analysis)
    best = current_analysis
    counters: Counter[str] = Counter()
    for _iteration in range(iterations):
        counters["iterations"] += 1
        generators = decompose_free_orbits(current)
        replaced = rng.randrange(len(generators))
        base = np.vstack(
            [
                translation_orbit(generator)
                for index, generator in enumerate(generators)
                if index != replaced
            ]
        )
        neighbor = extend_by_free_orbit(
            canonical_rowspace(base),
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if neighbor is None:
            counters["replacement_failed"] += 1
            continue
        analysis = analyze_candidate(neighbor)
        if not analysis["checks"]["tanner_connected"]:
            counters["disconnected"] += 1
            continue
        distance = int(analysis["certified_distance"])
        counters[f"neighbor_distance_{distance}"] += 1
        score = refinement_score(analysis)
        if score > refinement_score(best):
            best = analysis
            counters["best_improvement"] += 1
        if analysis["accepted"]:
            counters["accepted"] += 1
            return {
                "accepted": True,
                "best": analysis,
                "counters": dict(sorted(counters.items())),
            }
        if score > current_score or (score == current_score and rng.random() < 0.05):
            current = neighbor
            current_analysis = analysis
            current_score = score
            counters["move_accepted"] += 1
    return {
        "accepted": False,
        "best": best,
        "counters": dict(sorted(counters.items())),
    }


def search_batch(
    *,
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    survivors = []
    near_misses = []
    seen: set[bytes] = set()
    started = time.perf_counter()
    for _trial in range(trials):
        counters["trials"] += 1
        stabilizer = construct_candidate(
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
        )
        if stabilizer is None:
            counters["construction_failed"] += 1
            continue
        counters["constructed"] += 1
        key = canonical_rowspace(stabilizer).tobytes()
        if key in seen:
            counters["duplicate"] += 1
            continue
        seen.add(key)
        analysis = analyze_candidate(stabilizer)
        if not analysis["checks"]["tanner_connected"]:
            counters["disconnected"] += 1
            continue
        counters["connected"] += 1
        distance = int(analysis["certified_distance"])
        counters[f"distance_{distance}"] += 1
        if analysis["accepted"]:
            counters["accepted"] += 1
            survivors.append(analysis)
        elif len(near_misses) < maximum_saved_near_misses:
            near_misses.append(analysis)
    near_misses.sort(
        key=lambda item: (
            -item["certified_distance"],
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "random-free-rank-three-F2-C4-Lagrangians",
        "trials": trials,
        "random_seed": random_seed,
        "maximum_check_weight": maximum_check_weight,
        "attempts_per_orbit": attempts_per_orbit,
        "counters": dict(sorted(counters.items())),
        "survivors": survivors,
        "near_misses": near_misses[:maximum_saved_near_misses],
        "seconds": round(time.perf_counter() - started, 6),
    }


def search_module_type_batch(
    *,
    module_type: tuple[int, ...],
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    """Search one non-free/free cyclic C4 module type."""
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    survivors = []
    near_misses = []
    seen: set[bytes] = set()
    started = time.perf_counter()
    for _trial in range(trials):
        counters["trials"] += 1
        stabilizer = construct_module_type_candidate(
            module_type,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
        )
        if stabilizer is None:
            counters["construction_failed"] += 1
            continue
        counters["constructed"] += 1
        key = canonical_rowspace(stabilizer).tobytes()
        if key in seen:
            counters["duplicate"] += 1
            continue
        seen.add(key)
        analysis = analyze_candidate(stabilizer)
        analysis["module_type"] = list(module_type)
        if not analysis["checks"]["tanner_connected"]:
            counters["disconnected"] += 1
            continue
        counters["connected"] += 1
        distance = int(analysis["certified_distance"])
        counters[f"distance_{distance}"] += 1
        if analysis["accepted"]:
            counters["accepted"] += 1
            survivors.append(analysis)
        elif len(near_misses) < maximum_saved_near_misses:
            near_misses.append(analysis)
    near_misses.sort(
        key=lambda item: (
            -item["certified_distance"],
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "random-prescribed-cyclic-C4-module-type",
        "module_type": list(module_type),
        "trials": trials,
        "random_seed": random_seed,
        "maximum_check_weight": maximum_check_weight,
        "attempts_per_orbit": attempts_per_orbit,
        "counters": dict(sorted(counters.items())),
        "survivors": survivors,
        "near_misses": near_misses[:maximum_saved_near_misses],
        "seconds": round(time.perf_counter() - started, 6),
    }
