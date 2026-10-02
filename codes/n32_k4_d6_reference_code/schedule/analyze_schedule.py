#!/usr/bin/env python3
"""Build and verify syndrome schedules for the folded [[32,4,6]] code."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


NUM_QUBITS = 32
LOGICAL_ORDER = 4
THICKNESS = 8
RANK = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray(
        [[(mask >> qubit) & 1 for qubit in range(NUM_QUBITS)] for mask in masks],
        dtype=np.uint8,
    )


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


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def translate_row(row: np.ndarray, shift: int = 1) -> np.ndarray:
    return np.roll(row.reshape(LOGICAL_ORDER, THICKNESS), shift, axis=0).reshape(
        NUM_QUBITS
    )


def translation_orbit(row: np.ndarray) -> np.ndarray:
    return np.asarray([translate_row(row, shift) for shift in range(LOGICAL_ORDER)])


def stabilizer_sums(basis: np.ndarray) -> list[np.ndarray]:
    rows = [np.zeros(NUM_QUBITS, dtype=np.uint8)]
    for basis_row in basis:
        rows.extend(row ^ basis_row for row in tuple(rows))
    return rows


def minimum_weight_basis(basis: np.ndarray) -> np.ndarray:
    candidates = sorted(
        (row for row in stabilizer_sums(basis) if row.any()),
        key=lambda row: (
            int(row.sum()),
            sum(int(bit) << index for index, bit in enumerate(row)),
        ),
    )
    selected = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    for candidate in candidates:
        if gf2_rank(np.vstack([selected, candidate])) == gf2_rank(selected) + 1:
            selected = np.vstack([selected, candidate])
        if len(selected) == RANK:
            return selected
    raise RuntimeError("failed to extract a minimum-weight stabilizer basis")


def translation_symmetric_basis(basis: np.ndarray) -> np.ndarray:
    """Recover the productive (4,4,4,2) C4-module basis."""
    chosen = np.zeros((0, NUM_QUBITS), dtype=np.uint8)
    output: list[np.ndarray] = []
    candidates = sorted(
        stabilizer_sums(basis),
        key=lambda row: (
            int(row.sum()),
            sum(int(bit) << index for index, bit in enumerate(row)),
        ),
    )
    for desired_rank in (4, 4, 4, 2):
        for candidate in candidates:
            if not candidate.any():
                continue
            orbit = translation_orbit(candidate)
            if gf2_rank(orbit) != desired_rank:
                continue
            if gf2_rank(np.vstack([chosen, orbit])) != gf2_rank(chosen) + desired_rank:
                continue
            for row in orbit:
                if not any(np.array_equal(row, old) for old in output):
                    output.append(row.copy())
            chosen = np.vstack([chosen, orbit])
            break
        else:
            raise RuntimeError(f"failed to find a rank-{desired_rank} translated orbit")
    result = np.asarray(output, dtype=np.uint8)
    if len(result) != RANK or gf2_rank(result) != RANK:
        raise RuntimeError("translation-symmetric rows do not form a rank-14 basis")
    return result


def check_translation_action(checks: np.ndarray) -> list[int]:
    action = []
    for row in checks:
        translated = translate_row(row)
        matches = [index for index, other in enumerate(checks) if np.array_equal(translated, other)]
        if len(matches) != 1:
            raise RuntimeError("checks are not permuted by the physical C4 translation")
        action.append(matches[0])
    return action


def edge_coloring(checks: np.ndarray) -> list[list[tuple[int, int]]]:
    """Exact Delta-edge-coloring of a bipartite Tanner graph."""
    num_checks, num_qubits = checks.shape
    original = [(int(row), int(column)) for row, column in np.argwhere(checks)]
    left_degrees = Counter(row for row, _column in original)
    right_degrees = Counter(column for _row, column in original)
    delta = max(max(left_degrees.values()), max(right_degrees.values()))
    size = max(num_checks, num_qubits)
    multiplicity = Counter(original)
    real_multiplicity = Counter(original)
    left_deficit = [
        row for row in range(size) for _ in range(delta - left_degrees[row])
    ]
    right_deficit = [
        column for column in range(size) for _ in range(delta - right_degrees[column])
    ]
    random.Random(32).shuffle(right_deficit)
    for row, column in zip(left_deficit, right_deficit, strict=True):
        multiplicity[row, column] += 1
    layers: list[list[tuple[int, int]]] = []
    for _ in range(delta):
        match_right: list[int | None] = [None] * size

        def augment(row: int, seen: list[bool]) -> bool:
            for column in range(size):
                if not multiplicity[row, column] or seen[column]:
                    continue
                seen[column] = True
                if match_right[column] is None or augment(match_right[column], seen):
                    match_right[column] = row
                    return True
            return False

        for row in range(size):
            if not augment(row, [False] * size):
                raise RuntimeError("regularized Tanner graph lacks a perfect matching")
        matching = {row: column for column, row in enumerate(match_right)}
        layer: list[tuple[int, int]] = []
        for row in range(size):
            column = matching[row]
            multiplicity[row, column] -= 1
            if real_multiplicity[row, column]:
                real_multiplicity[row, column] -= 1
                layer.append((row, column))
        layers.append(sorted(layer))
    if any(real_multiplicity.values()):
        raise RuntimeError("edge coloring does not cover the Tanner graph exactly")
    return layers


def verify_collision_free(layers: list[dict[str, Any]]) -> None:
    observed: set[tuple[str, int, int]] = set()
    for layer in layers:
        ancillas: set[tuple[str, int]] = set()
        data: set[int] = set()
        for gate in layer["gates"]:
            key = (gate["type"], gate["check"], gate["data"])
            if key in observed:
                raise RuntimeError("a Tanner edge was scheduled twice")
            observed.add(key)
            ancilla = (gate["type"], gate["check"])
            if ancilla in ancillas or gate["data"] in data:
                raise RuntimeError("a schedule layer contains a collision")
            ancillas.add(ancilla)
            data.add(gate["data"])


def verify_clean_backaction(
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    layers: list[dict[str, Any]],
) -> None:
    times: dict[tuple[str, int, int], int] = {}
    for time, layer in enumerate(layers):
        for gate in layer["gates"]:
            times[(gate["type"], gate["check"], gate["data"])] = time
    for check_x in range(len(checks_x)):
        for check_z in range(len(checks_z)):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z])
            if len(overlap) % 2:
                raise RuntimeError("CSS check overlap is odd")
            inversions = sum(
                times[("X", check_x, int(data))]
                < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                raise RuntimeError("schedule leaves cross-ancilla back-action")


def schedule_payload(
    name: str,
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    layers: list[dict[str, Any]],
    permutation: np.ndarray,
    translation_action_x: list[int] | None,
    translation_action_z: list[int] | None,
    *,
    translation_time_permutation: list[int] | None,
) -> dict[str, Any]:
    verify_collision_free(layers)
    verify_clean_backaction(checks_x, checks_z, layers)
    edge_count = int(checks_x.sum() + checks_z.sum())
    if sum(len(layer["gates"]) for layer in layers) != edge_count:
        raise RuntimeError("schedule does not contain every Tanner edge")
    total_depth = len(layers)
    times = {
        (gate["type"], gate["check"], gate["data"]): time
        for time, layer in enumerate(layers)
        for gate in layer["gates"]
    }
    fold_symmetric = all(
        times[("Z", check, int(permutation[data]))] == total_depth - 1 - time
        for (kind, check, data), time in times.items()
        if kind == "X"
    )
    return {
        "name": name,
        "code": "[[32,4,6]]",
        "checks_per_css_type": len(checks_x),
        "independent_checks_per_css_type": RANK,
        "ancillas_total": 2 * len(checks_x),
        "check_supports_x": [np.flatnonzero(row).astype(int).tolist() for row in checks_x],
        "check_supports_z": [np.flatnonzero(row).astype(int).tolist() for row in checks_z],
        "check_weight_histogram": dict(sorted(Counter(map(int, checks_x.sum(axis=1))).items())),
        "data_degree_histogram_per_type": dict(
            sorted(Counter(map(int, checks_x.sum(axis=0))).items())
        ),
        "cnot_count": edge_count,
        "cnot_depth": total_depth,
        "translation_action_on_x_checks": translation_action_x,
        "translation_action_on_z_checks": translation_action_z,
        "translation_time_permutation_per_css_half": translation_time_permutation,
        "verified": {
            "rank_fourteen_each_type": gf2_rank(checks_x) == gf2_rank(checks_z) == RANK,
            "css_orthogonal": not bool(np.any((checks_x @ checks_z.T) % 2)),
            "collision_free": True,
            "all_tanner_edges_scheduled_once": True,
            "clean_cross_ancilla_backaction": True,
            "fold_P_plus_time_reversal_symmetry": fold_symmetric,
            "translation_preserves_check_sets": translation_action_x is not None
            and translation_action_z is not None,
            "even_data_degree_per_css_type": bool(np.all(checks_x.sum(axis=0) % 2 == 0))
            and bool(np.all(checks_z.sum(axis=0) % 2 == 0)),
        },
        "layers": layers,
    }


def minimal_schedule(
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, Any]:
    x_layers = edge_coloring(checks_x)
    if len(x_layers) != 8:
        raise RuntimeError("expected an optimal eight-layer one-type edge coloring")
    layers: list[dict[str, Any]] = []
    for layer in reversed(x_layers):
        layers.append(
            {
                "gates": [
                    {"type": "Z", "check": check, "data": int(permutation[data])}
                    for check, data in layer
                ]
            }
        )
    for layer in x_layers:
        layers.append(
            {
                "gates": [
                    {"type": "X", "check": check, "data": data}
                    for check, data in layer
                ]
            }
        )
    result = schedule_payload(
        "minimum-basis fold-symmetric sequential schedule",
        checks_x,
        checks_z,
        layers,
        permutation,
        None,
        None,
        translation_time_permutation=None,
    )
    result["optimality"] = {
        "one_css_type_depth": 8,
        "reason": "maximum check and data degree are eight; bipartite edge coloring attains eight",
        "full_sixteen_layer_cycle": "safe and fold-symmetric, but not claimed globally depth-optimal",
    }
    result["tradeoff"] = (
        "This basis is not C4-translation invariant and does not have even data degree, "
        "so it lacks the one-round STAR syndrome-parity guarantee."
    )
    return result


def efficient_schedule(
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    permutation: np.ndarray,
    action_x: list[int],
    action_z: list[int],
) -> dict[str, Any]:
    x_layers = edge_coloring(checks_x)
    if len(x_layers) != 10:
        raise RuntimeError("expected an optimal ten-layer one-type edge coloring")
    layers: list[dict[str, Any]] = []
    for layer in reversed(x_layers):
        layers.append(
            {
                "gates": [
                    {"type": "Z", "check": check, "data": int(permutation[data])}
                    for check, data in layer
                ]
            }
        )
    for layer in x_layers:
        layers.append(
            {
                "gates": [
                    {"type": "X", "check": check, "data": data}
                    for check, data in layer
                ]
            }
        )
    result = schedule_payload(
        "STAR-compatible translation-symmetric fold-symmetric sequential schedule",
        checks_x,
        checks_z,
        layers,
        permutation,
        action_x,
        action_z,
        translation_time_permutation=None,
    )
    result["optimality"] = {
        "one_css_type_depth": 10,
        "reason": "maximum check weight is ten; bipartite edge coloring attains ten",
        "full_twenty_layer_cycle": "safe and fold-symmetric, but not claimed globally depth-optimal",
    }
    return result


def covariant_schedule(
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    permutation: np.ndarray,
    action_x: list[int],
    action_z: list[int],
) -> dict[str, Any]:
    # Exact SAT certificate.  The 30 variables label the simultaneous C4 orbits
    # of X-check Tanner edges in the deterministic ordering below.
    time_permutation = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8]
    base_colors = [
        8, 6, 4, 5, 3, 2, 1, 0, 7, 6,
        4, 5, 2, 3, 1, 0, 8, 5, 6, 4,
        3, 2, 1, 0, 8, 5, 4, 0, 1, 0,
    ]
    qubit_action = [
        ((qubit // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + qubit % THICKNESS
        for qubit in range(NUM_QUBITS)
    ]
    edges = [
        (check, int(data))
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row)
    ]
    edge_set = set(edges)
    edge_info: dict[tuple[int, int], tuple[int, int]] = {}
    orbits: list[list[tuple[int, int]]] = []
    for edge in edges:
        if edge in edge_info:
            continue
        orbit: list[tuple[int, int]] = []
        current = edge
        while current not in orbit:
            if current not in edge_set:
                raise RuntimeError("translation left the X Tanner-edge set")
            orbit.append(current)
            current = (action_x[current[0]], qubit_action[current[1]])
        orbit_index = len(orbits)
        orbits.append(orbit)
        for phase, item in enumerate(orbit):
            edge_info[item] = (orbit_index, phase)
    if len(orbits) != len(base_colors) or any(len(orbit) != 4 for orbit in orbits):
        raise RuntimeError("unexpected translated Tanner-edge orbit structure")

    def permute_time(value: int, power: int) -> int:
        for _ in range(power):
            value = time_permutation[value]
        return value

    x_time = {
        edge: permute_time(base_colors[orbit], phase)
        for edge, (orbit, phase) in edge_info.items()
    }
    for edge, time in x_time.items():
        translated = (action_x[edge[0]], qubit_action[edge[1]])
        if x_time[translated] != time_permutation[time]:
            raise RuntimeError("X schedule is not C4-covariant in time")

    z_time = {
        (check, int(data)): 11 - x_time[(check, int(permutation[data]))]
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row)
    }
    for edge, time in z_time.items():
        translated = (action_z[edge[0]], qubit_action[edge[1]])
        if z_time[translated] != time_permutation[time]:
            raise RuntimeError("Z schedule is not C4-covariant in time")

    layers: list[dict[str, Any]] = []
    for time in range(12):
        layers.append(
            {
                "gates": [
                    {"type": "Z", "check": check, "data": data}
                    for (check, data), value in sorted(z_time.items())
                    if value == time
                ]
            }
        )
    for time in range(12):
        layers.append(
            {
                "gates": [
                    {"type": "X", "check": check, "data": data}
                    for (check, data), value in sorted(x_time.items())
                    if value == time
                ]
            }
        )
    result = schedule_payload(
        "strong C4-covariant and fold-symmetric sequential schedule",
        checks_x,
        checks_z,
        layers,
        permutation,
        action_x,
        action_z,
        translation_time_permutation=time_permutation,
    )
    result["covariance"] = {
        "physical_translation_T_permutes_relative_time_layers_by": time_permutation,
        "physical_fold_P_exchanges_css_types_and_reverses_all_24_layers": True,
        "dihedral_relation_on_relative_time_layers": "reflection * T * reflection = T^-1",
        "ten_and_eleven_layer_covariant_one-type_schedules": "excluded by exact finite-domain search",
        "twelve_layer_covariant_one-type_schedule": "verified by stored certificate",
    }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    record = json.loads(args.input.read_text())
    analysis = record["result"]["best"]
    canonical_x = matrix_from_masks(analysis["stabilizer_masks_x"])
    permutation = np.asarray(analysis["permutation"], dtype=int)
    minimum_x = minimum_weight_basis(canonical_x)
    minimum_z = act_rows(minimum_x, permutation)
    checks_x = translation_symmetric_basis(canonical_x)
    parity_row = np.bitwise_xor.reduce(checks_x, axis=0)
    checks_x = np.vstack([checks_x, parity_row])
    checks_z = act_rows(checks_x, permutation)
    action_x = check_translation_action(checks_x)
    action_z = check_translation_action(checks_z)

    if np.flatnonzero(parity_row).tolist() != [7, 15, 23, 31]:
        raise RuntimeError("unexpected parity-completion row")
    if not np.array_equal(act_rows(parity_row[None, :], permutation), parity_row[None, :]):
        raise RuntimeError("the parity-completion row is not fixed by P")
    if np.any((checks_x @ checks_z.T) % 2):
        raise RuntimeError("schedule checks violate CSS orthogonality")

    args.output.mkdir(parents=True)
    minimal = minimal_schedule(minimum_x, minimum_z, permutation)
    efficient = efficient_schedule(checks_x, checks_z, permutation, action_x, action_z)
    covariant = covariant_schedule(checks_x, checks_z, permutation, action_x, action_z)
    (args.output / "minimum-basis-fold-symmetric-16-layer.json").write_text(
        json.dumps(minimal, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "safe-fold-symmetric-20-layer.json").write_text(
        json.dumps(efficient, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "strong-c4-fold-symmetric-24-layer.json").write_text(
        json.dumps(covariant, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "code": "[[32,4,6]]",
        "source": str(args.input.resolve()),
        "translation_symmetric_parity_completed_presentation": {
            "checks_per_css_type": len(checks_x),
            "rank_per_css_type": gf2_rank(checks_x),
            "check_weight_histogram": dict(sorted(Counter(map(int, checks_x.sum(axis=1))).items())),
            "data_degree_histogram_per_type": dict(
                sorted(Counter(map(int, checks_x.sum(axis=0))).items())
            ),
            "parity_completion_support": np.flatnonzero(parity_row).astype(int).tolist(),
            "even_syndrome_parity": bool(np.all(checks_x.sum(axis=0) % 2 == 0)),
        },
        "schedules": {
            "minimum_basis": {
                "file": "minimum-basis-fold-symmetric-16-layer.json",
                "ancillas": minimal["ancillas_total"],
                "cnot_count": minimal["cnot_count"],
                "cnot_depth": minimal["cnot_depth"],
                "translation_symmetric": False,
                "even_syndrome_parity": False,
            },
            "star_compatible_translation_symmetric": {
                "file": "safe-fold-symmetric-20-layer.json",
                "ancillas": efficient["ancillas_total"],
                "cnot_count": efficient["cnot_count"],
                "cnot_depth": efficient["cnot_depth"],
                "translation_symmetric": True,
                "even_syndrome_parity": True,
            },
            "strong_temporal_symmetry": {
                "file": "strong-c4-fold-symmetric-24-layer.json",
                "ancillas": covariant["ancillas_total"],
                "cnot_count": covariant["cnot_count"],
                "cnot_depth": covariant["cnot_depth"],
                "translation_symmetric_in_checks_and_time": True,
                "even_syndrome_parity": True,
            },
        },
        "not_yet_certified": [
            "circuit-level fault distance under hook faults",
            "circuit-level logical error rate",
            "hardware routing and atom-movement time",
            "a clean simultaneous-X/Z schedule shorter than the safe sequential schedules",
        ],
    }
    (args.output / "schedule-analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
