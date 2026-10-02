#!/usr/bin/env python3
"""Build and verify the all-weight-eight C4-frame presentation of [[32,4,6]]."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli


NUM_QUBITS = 32
LOGICAL_ORDER = 4
THICKNESS = 8
CHECK_RANK = 14

CHECK_SUPPORTS_X = [
    [3, 6, 12, 13, 16, 21, 24, 26],
    [0, 2, 11, 14, 20, 21, 24, 29],
    [0, 5, 8, 10, 19, 22, 28, 29],
    [4, 5, 8, 13, 16, 18, 27, 30],
    [7, 16, 20, 21, 22, 25, 29, 31],
    [1, 5, 7, 15, 24, 28, 29, 30],
    [0, 4, 5, 6, 9, 13, 15, 23],
    [8, 12, 13, 14, 17, 21, 23, 31],
    [2, 4, 9, 12, 19, 22, 27, 29],
    [3, 5, 10, 12, 17, 20, 27, 30],
    [3, 6, 11, 13, 18, 20, 25, 28],
    [1, 4, 11, 14, 19, 21, 26, 28],
    [2, 6, 7, 15, 24, 26, 27, 29],
    [0, 2, 3, 5, 10, 14, 15, 23],
    [8, 10, 11, 13, 18, 22, 23, 31],
    [7, 16, 18, 19, 21, 26, 30, 31],
]

# Colors of the 32 simultaneous C4 orbits of X-check Tanner edges.  Orbit
# ordering is reconstructed deterministically below.
EDGE_ORBIT_COLORS = [
    4, 3, 6, 2, 0, 7, 5, 1,
    1, 7, 0, 3, 2, 6, 4, 5,
    2, 1, 7, 4, 0, 6, 3, 5,
    3, 7, 2, 0, 1, 4, 5, 6,
]

DEPENDENCY_SUPPORTS = [
    [0, 2, 4, 6, 8, 10, 12, 14],
    [1, 3, 5, 7, 9, 11, 13, 15],
]


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


def matrix_from_supports(supports: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(supports), NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


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
    rank_left = gf2_rank(left)
    rank_right = gf2_rank(right)
    return rank_left == rank_right == gf2_rank(np.vstack([left, right]))


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def translate_row(row: np.ndarray) -> np.ndarray:
    return np.roll(row.reshape(LOGICAL_ORDER, THICKNESS), 1, axis=0).reshape(
        NUM_QUBITS
    )


def translation_action(checks: np.ndarray) -> list[int]:
    action = []
    for row in checks:
        translated = translate_row(row)
        matches = [
            index for index, other in enumerate(checks) if np.array_equal(translated, other)
        ]
        if len(matches) != 1:
            raise RuntimeError("physical translation does not permute the checks")
        action.append(matches[0])
    return action


def build_edge_orbits(
    checks: np.ndarray, check_action: list[int]
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    qubit_action = [
        ((qubit // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + qubit % THICKNESS
        for qubit in range(NUM_QUBITS)
    ]
    edges = [
        (check, int(data))
        for check, row in enumerate(checks)
        for data in np.flatnonzero(row)
    ]
    edge_set = set(edges)
    orbits: list[list[tuple[int, int]]] = []
    edge_orbit: dict[tuple[int, int], int] = {}
    for edge in edges:
        if edge in edge_orbit:
            continue
        orbit: list[tuple[int, int]] = []
        current = edge
        while current not in orbit:
            if current not in edge_set:
                raise RuntimeError("translation left the Tanner-edge set")
            orbit.append(current)
            current = (check_action[current[0]], qubit_action[current[1]])
        if len(orbit) != LOGICAL_ORDER:
            raise RuntimeError("Tanner-edge orbit is not free under C4")
        if len({check for check, _data in orbit}) != len(orbit):
            raise RuntimeError("translated edge orbit collides at a check")
        if len({data for _check, data in orbit}) != len(orbit):
            raise RuntimeError("translated edge orbit collides at a data qubit")
        orbit_index = len(orbits)
        orbits.append(orbit)
        for item in orbit:
            edge_orbit[item] = orbit_index
    return orbits, edge_orbit


def collision_free(gates: list[dict[str, Any]]) -> bool:
    ancillas: set[tuple[str, int]] = set()
    data: set[int] = set()
    for gate in gates:
        ancilla = (gate["type"], gate["check"])
        if ancilla in ancillas or gate["data"] in data:
            return False
        ancillas.add(ancilla)
        data.add(gate["data"])
    return True


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    source = json.loads(args.input.read_text())
    winner = source["result"]["best"]
    canonical_x = matrix_from_masks(winner["stabilizer_masks_x"])
    canonical_z = matrix_from_masks(winner["stabilizer_masks_z"])
    permutation = np.asarray(winner["permutation"], dtype=int)

    checks_x = matrix_from_supports(CHECK_SUPPORTS_X)
    checks_z = act_rows(checks_x, permutation)
    action_x = translation_action(checks_x)
    action_z = translation_action(checks_z)
    orbits, edge_orbit = build_edge_orbits(checks_x, action_x)
    if len(orbits) != len(EDGE_ORBIT_COLORS):
        raise RuntimeError("edge-color certificate has the wrong size")

    x_layers: list[list[dict[str, Any]]] = []
    for color in range(8):
        gates = [
            {"type": "X", "check": check, "data": int(data)}
            for check, row in enumerate(checks_x)
            for data in np.flatnonzero(row).astype(int)
            if EDGE_ORBIT_COLORS[edge_orbit[(check, data)]] == color
        ]
        if len(gates) != 16 or not collision_free(gates):
            raise RuntimeError("invalid translation-invariant X edge color")
        x_layers.append(gates)

    full_layers: list[dict[str, Any]] = []
    for relative_time in range(8):
        x_color = 7 - relative_time
        full_layers.append(
            {
                "round": relative_time + 1,
                "type": "Z",
                "gates": [
                    {
                        "type": "Z",
                        "check": gate["check"],
                        "data": int(permutation[gate["data"]]),
                    }
                    for gate in x_layers[x_color]
                ],
            }
        )
    for relative_time, gates in enumerate(x_layers):
        full_layers.append(
            {
                "round": 9 + relative_time,
                "type": "X",
                "gates": gates,
            }
        )

    expected_edges = {
        ("X", check, int(data))
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row)
    } | {
        ("Z", check, int(data))
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row)
    }
    scheduled_edges = {
        (gate["type"], gate["check"], gate["data"])
        for layer in full_layers
        for gate in layer["gates"]
    }
    if expected_edges != scheduled_edges:
        raise RuntimeError("full schedule does not equal the two Tanner graphs")
    if not all(collision_free(layer["gates"]) for layer in full_layers):
        raise RuntimeError("full schedule contains a collision")

    times = {
        (gate["type"], gate["check"], gate["data"]): time
        for time, layer in enumerate(full_layers)
        for gate in layer["gates"]
    }
    fold_time_symmetric = all(
        times[("Z", check, int(permutation[data]))] == 15 - time
        for (kind, check, data), time in times.items()
        if kind == "X"
    )
    clean_backaction = True
    for check_x in range(16):
        for check_z in range(16):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z])
            if len(overlap) % 2:
                clean_backaction = False
            inversions = sum(
                times[("X", check_x, int(data))]
                < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                clean_backaction = False

    dependency_matrix = np.zeros((2, 16), dtype=np.uint8)
    for row, support in enumerate(DEPENDENCY_SUPPORTS):
        dependency_matrix[row, support] = 1
    dependencies_hold = bool(
        not np.any((dependency_matrix @ checks_x) % 2)
        and gf2_rank(dependency_matrix) == 2
        and 16 - gf2_rank(checks_x) == 2
    )

    code = codes.CSSCode(checks_x, checks_z)
    distance_x = int(code.get_distance_exact(Pauli.X))
    distance_z = int(code.get_distance_exact(Pauli.Z))
    degree_x = checks_x.sum(axis=0).astype(int)
    degree_z = checks_z.sum(axis=0).astype(int)

    presentation = {
        "schema_version": 1,
        "name": "all-weight-eight C4-frame presentation",
        "code": "[[32,4,6]]",
        "same_underlying_code_as_source": True,
        "source": str(args.input.resolve()),
        "checks_x": CHECK_SUPPORTS_X,
        "checks_z": [np.flatnonzero(row).astype(int).tolist() for row in checks_z],
        "check_count_per_css_type": 16,
        "rank_per_css_type": 14,
        "redundant_relations": DEPENDENCY_SUPPORTS,
        "check_weight_histogram_per_type": {"8": 16},
        "translation_action_on_x_checks": action_x,
        "translation_action_on_z_checks": action_z,
        "translation_check_orbits": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        "data_degree_histogram_per_type": dict(sorted(Counter(map(int, degree_x)).items())),
        "combined_xz_data_degree_histogram": dict(
            sorted(Counter(map(int, degree_x + degree_z)).items())
        ),
        "even_syndrome_parity": True,
        "dedicated_ancillas_total": 32,
        "reusable_ancillas_if_css_types_are_reset_sequentially": 16,
        "cnot_count": int(checks_x.sum() + checks_z.sum()),
        "safe_cnot_depth": 16,
        "one_css_type_cnot_depth": 8,
        "one_css_type_depth_optimal": True,
        "one_css_type_optimality_reason": "maximum check weight is eight",
        "combined_collision_lower_bound": int(max((degree_x + degree_z).max(), 8)),
        "fold_permutation": permutation.astype(int).tolist(),
    }
    schedule = {
        "schema_version": 1,
        "name": "safe all-weight-eight translation-invariant folded schedule",
        "presentation": "presentation.json",
        "instructions": "Apply the eight Z layers, then the eight X layers.",
        "cnot_depth": 16,
        "cnot_count": int(checks_x.sum() + checks_z.sum()),
        "x_edge_orbit_colors_zero_based": EDGE_ORBIT_COLORS,
        "each_one_type_layer_is_fixed_by_physical_C4_translation": True,
        "fold_P_exchanges_css_types_and_reverses_time": True,
        "layers": full_layers,
    }
    validation = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "qldpc": {
            "n": int(code.num_qubits),
            "k": int(code.dimension),
            "distance_x": distance_x,
            "distance_z": distance_z,
        },
        "checks": {
            "rank_x_fourteen": gf2_rank(checks_x) == CHECK_RANK,
            "rank_z_fourteen": gf2_rank(checks_z) == CHECK_RANK,
            "same_x_rowspace_as_saved_code": same_rowspace(checks_x, canonical_x),
            "same_z_rowspace_as_saved_code": same_rowspace(checks_z, canonical_z),
            "css_orthogonal": not bool(np.any((checks_x @ checks_z.T) % 2)),
            "all_check_weights_eight": bool(np.all(checks_x.sum(axis=1) == 8))
            and bool(np.all(checks_z.sum(axis=1) == 8)),
            "two_redundant_relations_verified": dependencies_hold,
            "xor_of_all_checks_zero": not bool(np.any(np.bitwise_xor.reduce(checks_x, axis=0))),
            "even_data_degree_each_type": bool(np.all(degree_x % 2 == 0))
            and bool(np.all(degree_z % 2 == 0)),
            "translation_preserves_check_sets": True,
        },
        "schedule": {
            "all_edges_once": expected_edges == scheduled_edges,
            "collision_free": all(collision_free(layer["gates"]) for layer in full_layers),
            "clean_cross_ancilla_backaction": clean_backaction,
            "translation_invariant_one_type_layers": True,
            "fold_P_plus_time_reversal": fold_time_symmetric,
            "eight_layers_per_css_type": len(x_layers) == 8,
            "sixteen_edges_per_one_type_layer": all(len(layer) == 16 for layer in x_layers),
        },
    }
    required = [
        validation["qldpc"] == {"n": 32, "k": 4, "distance_x": 6, "distance_z": 6},
        *validation["checks"].values(),
        *validation["schedule"].values(),
    ]
    validation["all_required_checks_pass"] = bool(all(required))
    if not validation["all_required_checks_pass"]:
        raise RuntimeError("presentation validation failed")

    args.output.mkdir(parents=True)
    for name, value in (
        ("presentation.json", presentation),
        ("safe-fold-symmetric-16-layer.json", schedule),
        ("validation.json", validation),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
