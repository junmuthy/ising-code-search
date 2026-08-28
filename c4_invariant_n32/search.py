"""Algebra and exact certification for C4-invariant ``[[32,4,*]]`` codes."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterable

import numpy as np

from gala_search.single_row import _tanner_connected, gf2_nullspace, gf2_rank, gf2_rref

SCHEMA_VERSION = 1
LOGICAL_ORDER = 4
ACTIVE_THICKNESS = 7
PHYSICAL_THICKNESS = 8
NUM_QUBITS = LOGICAL_ORDER * PHYSICAL_THICKNESS
TARGET_STABILIZER_RANK = (NUM_QUBITS - LOGICAL_ORDER) // 2
DEFAULT_MAXIMUM_CHECK_WEIGHT = 12


def coordinate(logical: int, thickness: int) -> int:
    return (logical % LOGICAL_ORDER) * PHYSICAL_THICKNESS + thickness


def logical_columns() -> np.ndarray:
    output = np.zeros((LOGICAL_ORDER, NUM_QUBITS), dtype=np.uint8)
    for logical in range(LOGICAL_ORDER):
        start = logical * PHYSICAL_THICKNESS
        output[logical, start : start + ACTIVE_THICKNESS] = 1
    return output


def spectator_support() -> list[int]:
    return [coordinate(logical, ACTIVE_THICKNESS) for logical in range(LOGICAL_ORDER)]


def translate_vector(vector: np.ndarray, shift: int = 1) -> np.ndarray:
    reshaped = np.asarray(vector, dtype=np.uint8).reshape(
        LOGICAL_ORDER, PHYSICAL_THICKNESS
    )
    return np.roll(reshaped, shift=shift, axis=0).reshape(NUM_QUBITS)


def translation_orbit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [translate_vector(vector, shift) for shift in range(LOGICAL_ORDER)],
        dtype=np.uint8,
    )


def translation_matrix() -> np.ndarray:
    output = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    for column in range(NUM_QUBITS):
        unit = np.zeros(NUM_QUBITS, dtype=np.uint8)
        unit[column] = 1
        output[:, column] = translate_vector(unit)
    return output


def nilpotent_translation_power(power: int) -> np.ndarray:
    if not 0 <= power <= LOGICAL_ORDER:
        raise ValueError("power must be between zero and four")
    nilpotent = translation_matrix() ^ np.eye(NUM_QUBITS, dtype=np.uint8)
    output = np.eye(NUM_QUBITS, dtype=np.uint8)
    for _factor in range(power):
        output = (nilpotent @ output) % 2
    return output.astype(np.uint8)


def canonical_rowspace(matrix: np.ndarray) -> np.ndarray:
    reduced, _pivots = gf2_rref(np.asarray(matrix, dtype=np.uint8))
    return reduced


def packed(vector: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(vector))


def vector_from_mask(mask: int) -> np.ndarray:
    return np.fromiter(
        ((mask >> qubit) & 1 for qubit in range(NUM_QUBITS)),
        dtype=np.uint8,
        count=NUM_QUBITS,
    )


def stabilizer_sums(stabilizer: np.ndarray) -> tuple[int, ...]:
    basis = [packed(row) for row in canonical_rowspace(stabilizer)]
    sums = [0]
    for row in basis:
        sums.extend(value ^ row for value in tuple(sums))
    return tuple(sums)


def minimum_weight_basis(stabilizer: np.ndarray) -> np.ndarray:
    candidates = sorted(
        (mask.bit_count(), mask) for mask in stabilizer_sums(stabilizer) if mask
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


def embed_n28_stabilizer(stabilizer: np.ndarray) -> np.ndarray:
    old = np.asarray(stabilizer, dtype=np.uint8)
    if old.ndim != 2 or old.shape[1] != 28:
        raise ValueError("expected an n=28 stabilizer matrix")
    output = np.zeros((old.shape[0], NUM_QUBITS), dtype=np.uint8)
    for logical in range(LOGICAL_ORDER):
        old_start = logical * ACTIVE_THICKNESS
        new_start = logical * PHYSICAL_THICKNESS
        output[:, new_start : new_start + ACTIVE_THICKNESS] = old[
            :, old_start : old_start + ACTIVE_THICKNESS
        ]
    return canonical_rowspace(output)


def embed_n28_vector(vector: np.ndarray) -> np.ndarray:
    old = np.asarray(vector, dtype=np.uint8).reshape(4, 7)
    output = np.zeros((4, 8), dtype=np.uint8)
    output[:, :7] = old
    return output.reshape(NUM_QUBITS)


def nullspace_vectors(basis: np.ndarray) -> Iterable[np.ndarray]:
    """Enumerate a binary row-basis span once using Gray-code updates."""
    if not len(basis):
        return
    current = np.zeros(basis.shape[1], dtype=np.uint8)
    previous_gray = 0
    for value in range(1, 1 << len(basis)):
        gray = value ^ (value >> 1)
        changed = gray ^ previous_gray
        index = changed.bit_length() - 1
        current ^= basis[index]
        previous_gray = gray
        yield current.copy()


def rank_two_extensions(
    embedded_stabilizer: np.ndarray,
    *,
    maximum_check_weight: int = DEFAULT_MAXIMUM_CHECK_WEIGHT,
) -> Iterable[np.ndarray]:
    """Enumerate distinct compatible rank-two C4 stabilizer extensions."""
    if gf2_rank(embedded_stabilizer) != 12:
        raise ValueError("embedded seed must have stabilizer rank twelve")
    constraints = np.vstack(
        [
            logical_columns(),
            embedded_stabilizer,
            nilpotent_translation_power(2),
        ]
    )
    nullspace = gf2_nullspace(constraints)
    seen: set[bytes] = set()
    for vector in nullspace_vectors(nullspace):
        weight = int(np.count_nonzero(vector))
        if not 2 <= weight <= maximum_check_weight:
            continue
        orbit = translation_orbit(vector)
        if gf2_rank(orbit) != 2:
            continue
        if np.any((orbit @ orbit.T) % 2):
            continue
        extended = canonical_rowspace(np.vstack([embedded_stabilizer, orbit]))
        if gf2_rank(extended) != TARGET_STABILIZER_RANK:
            continue
        key = extended.tobytes()
        if key in seen:
            continue
        seen.add(key)
        yield extended


def random_nullspace_vector(basis: np.ndarray, rng: random.Random) -> np.ndarray:
    bits = rng.getrandbits(len(basis))
    if bits == 0:
        return np.zeros(basis.shape[1], dtype=np.uint8)
    selected = [index for index in range(len(basis)) if bits >> index & 1]
    return np.bitwise_xor.reduce(basis[selected], axis=0)


def extend_by_orbit_rank(
    stabilizer: np.ndarray,
    *,
    orbit_rank: int,
    rng: random.Random,
    maximum_check_weight: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Randomly add one isotropic C4 orbit with a prescribed rank."""
    if not 1 <= orbit_rank <= LOGICAL_ORDER:
        raise ValueError("orbit rank must be in 1..4")
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
        extended = canonical_rowspace(np.vstack([stabilizer, orbit]))
        if gf2_rank(extended) != current_rank + orbit_rank:
            continue
        return extended, vector
    return None


def coupled_extension_candidate(
    old_orbit_generators: tuple[np.ndarray, ...],
    *,
    omitted_orbit: int,
    rng: random.Random,
    maximum_check_weight: int = DEFAULT_MAXIMUM_CHECK_WEIGHT,
    attempts_per_orbit: int = 200,
) -> np.ndarray | None:
    """Keep two n=28 free orbits and jointly regenerate ranks two and four."""
    if len(old_orbit_generators) != 3:
        raise ValueError("expected three free n=28 orbit generators")
    kept = [
        embed_n28_vector(generator)
        for index, generator in enumerate(old_orbit_generators)
        if index != omitted_orbit
    ]
    base = canonical_rowspace(
        np.vstack([translation_orbit(generator) for generator in kept])
    )
    rank_two = extend_by_orbit_rank(
        base,
        orbit_rank=2,
        rng=rng,
        maximum_check_weight=maximum_check_weight,
        attempts=attempts_per_orbit,
    )
    if rank_two is None:
        return None
    intermediate, _rank_two_generator = rank_two
    rank_four = extend_by_orbit_rank(
        intermediate,
        orbit_rank=4,
        rng=rng,
        maximum_check_weight=maximum_check_weight,
        attempts=attempts_per_orbit,
    )
    if rank_four is None:
        return None
    output, _rank_four_generator = rank_four
    if gf2_rank(output) != TARGET_STABILIZER_RANK:
        raise AssertionError("coupled extension has the wrong rank")
    return output


def tanner_connected(stabilizer: np.ndarray) -> bool:
    return _tanner_connected(stabilizer, stabilizer)


def module_partitions(
    total: int = TARGET_STABILIZER_RANK,
    maximum_part: int = LOGICAL_ORDER,
) -> tuple[tuple[int, ...], ...]:
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
    attempts_per_orbit: int = 300,
) -> np.ndarray | None:
    if sum(module_type) != TARGET_STABILIZER_RANK:
        raise ValueError("module-type ranks must sum to fourteen")
    stabilizer = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    for orbit_rank in reversed(module_type):
        result = extend_by_orbit_rank(
            stabilizer,
            orbit_rank=orbit_rank,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if result is None:
            return None
        stabilizer, _generator = result
    if gf2_rank(stabilizer) != TARGET_STABILIZER_RANK:
        raise AssertionError("constructed stabilizer has the wrong rank")
    return stabilizer


def search_module_type_batch(
    *,
    module_type: tuple[int, ...],
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved: int,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    seen: set[bytes] = set()
    best: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
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
        if not tanner_connected(stabilizer):
            counters["disconnected"] += 1
            continue
        counters["connected"] += 1
        analysis = analyze_candidate(stabilizer)
        analysis["module_type"] = list(module_type)
        distance = int(analysis["certified_distance"])
        counters[f"distance_{distance}"] += 1
        if analysis["accepted"]:
            counters["accepted"] += 1
            survivors.append(analysis)
        best.append(analysis)
        best.sort(
            key=lambda item: (
                item["certified_distance"],
                -int(
                    item["distance"]["weight_counts_through_seven"].get(
                        str(item["certified_distance"]), 0
                    )
                ),
                -item["maximum_check_weight"],
            ),
            reverse=True,
        )
        del best[maximum_saved:]
    return {
        "schema_version": SCHEMA_VERSION,
        "module_type": list(module_type),
        "counters": dict(sorted(counters.items())),
        "survivors": survivors,
        "best": best,
    }


def decompose_4442(stabilizer: np.ndarray) -> tuple[tuple[int, np.ndarray], ...]:
    """Find three rank-four generators and one rank-two generator."""
    chosen = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    output: list[tuple[int, np.ndarray]] = []
    candidates = sorted(
        stabilizer_sums(stabilizer), key=lambda mask: (mask.bit_count(), mask)
    )
    for desired_rank in (4, 4, 4, 2):
        found = False
        for mask in candidates:
            if not mask:
                continue
            vector = vector_from_mask(mask)
            orbit = translation_orbit(vector)
            if gf2_rank(orbit) != desired_rank:
                continue
            current_rank = gf2_rank(chosen)
            extended = np.vstack([chosen, orbit])
            if gf2_rank(extended) != current_rank + desired_rank:
                continue
            output.append((desired_rank, vector))
            chosen = canonical_rowspace(extended)
            found = True
            break
        if not found:
            raise ValueError("failed to decompose stabilizer as module type (4,4,4,2)")
    if gf2_rank(chosen) != TARGET_STABILIZER_RANK:
        raise AssertionError("module decomposition has the wrong rank")
    return tuple(output)


def refinement_score(analysis: dict[str, Any]) -> tuple[int, int, int]:
    distance = int(analysis["certified_distance"])
    counts = analysis["distance"]["weight_counts_through_seven"]
    return (
        distance,
        -int(counts.get(str(distance), 0)),
        -sum(int(counts.get(str(weight), 0)) for weight in range(1, 6)),
    )


def locally_refine_4442(
    stabilizer: np.ndarray,
    *,
    rng: random.Random,
    iterations: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    current = canonical_rowspace(stabilizer)
    current_analysis = analyze_candidate(current)
    current_score = refinement_score(current_analysis)
    best = current_analysis
    counters: Counter[str] = Counter()
    for _iteration in range(iterations):
        counters["iterations"] += 1
        decomposition = decompose_4442(current)
        replaced = rng.randrange(len(decomposition))
        replaced_rank = decomposition[replaced][0]
        base = canonical_rowspace(
            np.vstack(
                [
                    translation_orbit(generator)
                    for index, (_rank, generator) in enumerate(decomposition)
                    if index != replaced
                ]
            )
        )
        result = extend_by_orbit_rank(
            base,
            orbit_rank=replaced_rank,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if result is None:
            counters["replacement_failed"] += 1
            continue
        neighbor, _generator = result
        if not tanner_connected(neighbor):
            counters["disconnected"] += 1
            continue
        analysis = analyze_candidate(neighbor)
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


def analyze_candidate(stabilizer: np.ndarray) -> dict[str, Any]:
    stabilizer = canonical_rowspace(stabilizer)
    logicals = logical_columns()
    rank = gf2_rank(stabilizer)
    translated = np.asarray(
        [translate_vector(row) for row in stabilizer], dtype=np.uint8
    )
    checks = {
        "rank_fourteen": rank == TARGET_STABILIZER_RANK,
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
    distance = exact_logical_distance(stabilizer)
    displayed_basis = minimum_weight_basis(stabilizer)
    row_weights = np.count_nonzero(displayed_basis, axis=1)
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
        "stabilizer_masks": [packed(row) for row in stabilizer],
        "logical_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in logicals
        ],
        "spectator_support": spectator_support(),
        "logical_translation": "(x,y) -> (x+1 mod 4,y)",
        "zx_pairing": ((logicals @ logicals.T) % 2).astype(int).tolist(),
    }
