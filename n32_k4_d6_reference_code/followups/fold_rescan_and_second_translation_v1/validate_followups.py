#!/usr/bin/env python3
"""Independent qLDPC and circuit-structure validation of the cheap follow-ups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli


NUM_QUBITS = 32
LOGICAL_ORDER = 4
THICKNESS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--automorphisms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def matrix_from_supports(supports: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(supports), NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray(
        [[int(mask) >> data & 1 for data in range(NUM_QUBITS)] for mask in masks],
        dtype=np.uint8,
    )


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    return gf2_rank(left) == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def permutation_matrix_pairing(left: np.ndarray, right: np.ndarray) -> bool:
    pairing = (left @ right.T) % 2
    return bool(np.all(pairing.sum(axis=0) == 1) and np.all(pairing.sum(axis=1) == 1))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    source = json.loads(args.source.read_text())["result"]["best"]
    presentation = json.loads(args.presentation.read_text())
    folds = [json.loads(line) for line in args.folds.read_text().splitlines() if line.strip()]
    schedule = json.loads(args.schedule.read_text())
    automorphisms = json.loads(args.automorphisms.read_text())
    checks_x = matrix_from_supports(presentation["checks_x"])
    canonical_x = matrix_from_masks(source["stabilizer_masks_x"])
    logical_z = matrix_from_supports(source["logical_supports_z"])
    translation = np.asarray(
        [
            ((data // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + data % THICKNESS
            for data in range(NUM_QUBITS)
        ],
        dtype=int,
    )
    identity = np.arange(NUM_QUBITS)

    fold_results = []
    for index, fold in enumerate(folds):
        permutation = np.asarray(fold["permutation"], dtype=int)
        checks_z = act_rows(checks_x, permutation)
        logical_x = act_rows(logical_z, permutation)
        code = codes.CSSCode(checks_x, checks_z)
        conjugated = permutation[translation[np.argsort(permutation)]]
        expected_translation = np.linalg.matrix_power(
            np.eye(NUM_QUBITS, dtype=np.uint8)[translation], 1
        )
        # Direct array form of T^epsilon avoids interpreting matrix conventions.
        target = identity.copy()
        exponent = fold["epsilon"] % LOGICAL_ORDER
        for _ in range(exponent):
            target = translation[target]
        result = {
            "fold_index": index,
            "involution": bool(np.array_equal(permutation[permutation], identity)),
            "normalizes_C4_as_claimed": bool(np.array_equal(conjugated, target)),
            "css_orthogonal": not bool(np.any((checks_x @ checks_z.T) % 2)),
            "rank_x": gf2_rank(checks_x),
            "rank_z": gf2_rank(checks_z),
            "n": int(code.num_qubits),
            "k": int(code.dimension),
            "distance_x": int(code.get_distance_exact(Pauli.X)),
            "distance_z": int(code.get_distance_exact(Pauli.Z)),
            "logical_pairing_is_permutation": permutation_matrix_pairing(logical_z, logical_x),
        }
        fold_results.append(result)
        print(f"checkpoint qldpc fold={index + 1}/{len(folds)} dX={result['distance_x']} dZ={result['distance_z']}", flush=True)

    best_fold_index = int(schedule["fold_index"])
    permutation = np.asarray(folds[best_fold_index]["permutation"], dtype=int)
    checks_z = act_rows(checks_x, permutation)
    layers = schedule["layers"]
    times = {
        (gate["type"], int(gate["check"]), int(gate["data"])): time
        for time, layer in enumerate(layers)
        for gate in layer["gates"]
    }
    expected_edges = {
        ("X", check, int(data))
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row)
    } | {
        ("Z", check, int(data))
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row)
    }
    collision_free = True
    for layer in layers:
        ancillas = set()
        data_seen = set()
        for gate in layer["gates"]:
            ancilla = (gate["type"], int(gate["check"]))
            data = int(gate["data"])
            if ancilla in ancillas or data in data_seen:
                collision_free = False
            ancillas.add(ancilla)
            data_seen.add(data)
    clean_backaction = True
    for check_x in range(16):
        for check_z in range(16):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z])
            inversions = sum(
                times[("X", check_x, int(data))] < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                clean_backaction = False
    fold_time_reversal = all(
        times[("Z", check, int(permutation[data]))] == len(layers) - 1 - time
        for (kind, check, data), time in times.items()
        if kind == "X"
    )

    check_action_x = presentation["translation_action_on_x_checks"]
    epsilon = int(folds[best_fold_index]["epsilon"])
    check_action_z = list(range(16))
    for check in range(16):
        image = check
        for _ in range(epsilon % LOGICAL_ORDER):
            image = check_action_x[image]
        check_action_z[check] = image
    translation_invariant = True
    for layer in layers:
        gate_set = {
            (gate["type"], int(gate["check"]), int(gate["data"]))
            for gate in layer["gates"]
        }
        translated = {
            (
                kind,
                check_action_x[check] if kind == "X" else check_action_z[check],
                int(translation[data]),
            )
            for kind, check, data in gate_set
        }
        if translated != gate_set:
            translation_invariant = False

    degree_x = checks_x.sum(axis=0).astype(int)
    degree_z = checks_z.sum(axis=0).astype(int)
    combined_lower_bound = int(max(8, np.max(degree_x + degree_z)))
    schedule_result = {
        "all_edges_once": set(times) == expected_edges and len(times) == 256,
        "collision_free": collision_free,
        "clean_cross_ancilla_backaction": clean_backaction,
        "fold_P_plus_time_reversal": fold_time_reversal,
        "each_layer_C4_invariant": translation_invariant,
        "depth": len(layers),
        "combined_degree_lower_bound": combined_lower_bound,
        "depth_is_optimal_by_lower_bound": len(layers) == combined_lower_bound,
    }

    all_fold_checks = all(
        result["involution"]
        and result["normalizes_C4_as_claimed"]
        and result["css_orthogonal"]
        and result["rank_x"] == result["rank_z"] == 14
        and result["n"] == 32
        and result["k"] == 4
        and result["distance_x"] == result["distance_z"] == 6
        and result["logical_pairing_is_permutation"]
        for result in fold_results
    )
    output = {
        "schema_version": 1,
        "presentation_same_HX_rowspace_as_saved_code": same_rowspace(checks_x, canonical_x),
        "fold_count": len(folds),
        "all_24_folds_independently_qldpc_validated_as_32_4_6": all_fold_checks and len(folds) == 24,
        "fold_results": fold_results,
        "best_schedule": schedule_result,
        "second_translation_search": {
            "complete_exact_tanner_automorphism_count": automorphisms["counters"]["tanner_automorphisms"],
            "only_known_C4_powers": automorphisms["counters"]["tanner_automorphisms"] == 4,
            "found_second_translation": automorphisms["found_second_translation"],
            "found_order_16_translation": automorphisms["found_order_16_translation"],
        },
    }
    output["all_required_checks_pass"] = bool(
        output["presentation_same_HX_rowspace_as_saved_code"]
        and output["all_24_folds_independently_qldpc_validated_as_32_4_6"]
        and all(schedule_result.values())
        and output["second_translation_search"]["only_known_C4_powers"]
        and not output["second_translation_search"]["found_second_translation"]
        and not output["second_translation_search"]["found_order_16_translation"]
    )
    if not output["all_required_checks_pass"]:
        raise RuntimeError("independent follow-up validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "fold_results"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
