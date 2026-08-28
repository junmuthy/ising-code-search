"""Fast orbit-replacement refinement for folded [[32,4,*]] CSS codes."""

from __future__ import annotations

import pathlib
import random
import sys
from collections import Counter
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c4_invariant_n32.search import (  # noqa: E402
    NUM_QUBITS,
    TARGET_STABILIZER_RANK,
    canonical_rowspace,
    decompose_4442,
    gf2_nullspace,
    gf2_rank,
    logical_columns,
    minimum_weight_basis,
    nilpotent_translation_power,
    packed,
    random_nullspace_vector,
    stabilizer_sums,
    translate_vector,
    translation_orbit,
)

from scan_frontier_folds import act_rows, pairing_is_permutation, tanner_connected
from scan_shifted_folds import shifted_fold


def vector_from_mask(mask: int) -> np.ndarray:
    return np.fromiter(
        ((int(mask) >> qubit) & 1 for qubit in range(NUM_QUBITS)),
        dtype=np.uint8,
        count=NUM_QUBITS,
    )


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray([vector_from_mask(mask) for mask in masks], dtype=np.uint8)


def exact_logical_distance(
    stabilizer: np.ndarray, logicals: np.ndarray
) -> dict[str, Any]:
    sums = stabilizer_sums(stabilizer)
    logical_masks = [packed(row) for row in logicals]
    best_weight = NUM_QUBITS + 1
    best_operator = 0
    best_label = 0
    counts: Counter[int] = Counter()
    for label in range(1, 1 << len(logical_masks)):
        logical = 0
        for index, mask in enumerate(logical_masks):
            if label >> index & 1:
                logical ^= mask
        for stabilizer_mask in sums:
            operator = logical ^ stabilizer_mask
            weight = operator.bit_count()
            if weight <= 7:
                counts[weight] += 1
            if weight < best_weight:
                best_weight = weight
                best_operator = operator
                best_label = label
    return {
        "distance": best_weight,
        "logical_label": best_label,
        "operator": best_operator,
        "operator_support": [
            qubit for qubit in range(NUM_QUBITS) if best_operator >> qubit & 1
        ],
        "weight_counts_through_seven": {
            str(weight): counts[weight] for weight in range(1, 8)
        },
        "stabilizers_per_coset": len(sums),
        "cosets_checked": (1 << len(logical_masks)) - 1,
    }


def translation_invariant(matrix: np.ndarray) -> bool:
    translated = np.asarray([translate_vector(row) for row in matrix], dtype=np.uint8)
    return gf2_rank(np.vstack([matrix, translated])) == gf2_rank(matrix)


def analyze_folded(stabilizer_x: np.ndarray, permutation: np.ndarray) -> dict[str, Any]:
    hx = canonical_rowspace(stabilizer_x)
    hz = canonical_rowspace(act_rows(hx, permutation))
    logical_z = logical_columns()
    logical_x = act_rows(logical_z, permutation)
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    pairing = (logical_z @ logical_x.T) % 2
    checks = {
        "rank_fourteen_x": rank_x == TARGET_STABILIZER_RANK,
        "rank_fourteen_z": rank_z == TARGET_STABILIZER_RANK,
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "logical_z_in_kernel_x": bool(not np.any((hx @ logical_z.T) % 2)),
        "logical_x_in_kernel_z": bool(not np.any((hz @ logical_x.T) % 2)),
        "logical_pairing_permutation": pairing_is_permutation(pairing),
        "c4_invariant_x": translation_invariant(hx),
        "c4_invariant_z": translation_invariant(hz),
        "tanner_connected": tanner_connected(hx, hz),
        "distinct_check_spaces": gf2_rank(np.vstack([hx, hz])) > rank_x,
    }
    distance_z = exact_logical_distance(hz, logical_z)
    # P is an involution and preserves weight, so the two distances agree.
    distance = int(distance_z["distance"])
    displayed = minimum_weight_basis(hx)
    row_weights = np.count_nonzero(displayed, axis=1).astype(int).tolist()
    return {
        "n": NUM_QUBITS,
        "k": NUM_QUBITS - rank_x - rank_z,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks": checks,
        "accepted_structurally": all(
            value for key, value in checks.items() if key != "distinct_check_spaces"
        ),
        "distance": distance,
        "distance_x": distance,
        "distance_z": distance,
        "distance_detail_z": distance_z,
        "accepted": all(checks.values()) and distance >= 6,
        "maximum_check_weight": max(row_weights, default=0),
        "row_weights_x": row_weights,
        "stabilizer_masks_x": [packed(row) for row in hx],
        "stabilizer_masks_z": [packed(row) for row in hz],
        "logical_supports_z": [
            np.flatnonzero(row).astype(int).tolist() for row in logical_z
        ],
        "logical_supports_x": [
            np.flatnonzero(row).astype(int).tolist() for row in logical_x
        ],
        "pairing": pairing.astype(int).tolist(),
        "permutation": permutation.astype(int).tolist(),
    }


def refinement_score(analysis: dict[str, Any]) -> tuple[int, int, int]:
    distance = int(analysis["distance"])
    counts = analysis["distance_detail_z"]["weight_counts_through_seven"]
    return (
        distance,
        -int(counts.get(str(distance), 0)),
        -sum(int(counts.get(str(weight), 0)) for weight in range(1, 6)),
    )


def extend_folded_by_orbit_rank(
    base: np.ndarray,
    *,
    orbit_rank: int,
    permutation: np.ndarray,
    rng: random.Random,
    maximum_check_weight: int,
    attempts: int,
) -> np.ndarray | None:
    folded_base = act_rows(base, permutation)
    constraints = [logical_columns(), folded_base]
    if orbit_rank < 4:
        constraints.append(nilpotent_translation_power(orbit_rank))
    nullspace = gf2_nullspace(np.vstack(constraints))
    current_rank = gf2_rank(base)
    for _attempt in range(attempts):
        vector = random_nullspace_vector(nullspace, rng)
        weight = int(np.count_nonzero(vector))
        if not 2 <= weight <= maximum_check_weight:
            continue
        orbit = translation_orbit(vector)
        if gf2_rank(orbit) != orbit_rank:
            continue
        candidate = canonical_rowspace(np.vstack([base, orbit]))
        if gf2_rank(candidate) != current_rank + orbit_rank:
            continue
        folded = act_rows(candidate, permutation)
        if np.any((candidate @ folded.T) % 2):
            continue
        return candidate
    return None


def refine_folded_4442(
    stabilizer_x: np.ndarray,
    permutation: np.ndarray,
    *,
    rng: random.Random,
    iterations: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    current = canonical_rowspace(stabilizer_x)
    current_analysis = analyze_folded(current, permutation)
    if not current_analysis["accepted_structurally"]:
        raise ValueError("starting folded code is not structurally valid")
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
        neighbor = extend_folded_by_orbit_rank(
            base,
            orbit_rank=replaced_rank,
            permutation=permutation,
            rng=rng,
            maximum_check_weight=maximum_check_weight,
            attempts=attempts_per_orbit,
        )
        if neighbor is None:
            counters["replacement_failed"] += 1
            continue
        if not tanner_connected(neighbor, act_rows(neighbor, permutation)):
            counters["disconnected"] += 1
            continue
        analysis = analyze_folded(neighbor, permutation)
        distance = int(analysis["distance"])
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


def seed_from_record(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = record["source"]
    stabilizer = matrix_from_masks(source["stabilizer_masks"])
    permutation = shifted_fold(tuple(record["shifts"]), int(record["epsilon"]))
    return stabilizer, permutation
