#!/usr/bin/env python3
"""Materialize and verify the depth-optimal simultaneous BB56 schedule."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


ELL = 4
M = 7
CELL_COUNT = ELL * M
NUM_DATA_QUBITS = 2 * CELL_COUNT
NUM_CHECKS_PER_TYPE = CELL_COUNT
DEPTH = 8

ELEMENTS = tuple(itertools.product(range(ELL), range(M)))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}

# H_X = [A(a) | A(b)] with these supports.
A_SUPPORT = ((0, 0), (0, 1), (1, 0), (1, 2))
B_SUPPORT = ((0, 0), (0, 6), (1, 0), (1, 5))

# Orbit order: L terms of a, then R terms of b.  Each value is the zero-based
# CNOT layer assigned to that entire 28-edge translation orbit.
EDGE_ORBITS = tuple((0, shift) for shift in A_SUPPORT) + tuple(
    (1, shift) for shift in B_SUPPORT
)
EDGE_ORBIT_COLORS = (2, 5, 1, 6, 4, 3, 0, 7)


def add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (left[0] + right[0]) % ELL, (left[1] + right[1]) % M


def neg(value: tuple[int, int]) -> tuple[int, int]:
    return (-value[0]) % ELL, (-value[1]) % M


def data_index(half: int, cell: tuple[int, int]) -> int:
    return half * CELL_COUNT + INDEX[cell]


def data_coordinates(data: int) -> tuple[int, tuple[int, int]]:
    half, within = divmod(int(data), CELL_COUNT)
    return half, ELEMENTS[within]


def term_name(shift: tuple[int, int]) -> str:
    x, y = shift
    parts = []
    if x:
        parts.append("x" if x == 1 else f"x^{x}")
    if y:
        signed = y if y <= M // 2 else y - M
        parts.append("y" if signed == 1 else f"y^{signed}")
    return "1" if not parts else "*".join(parts)


def expected_checks() -> tuple[list[set[int]], list[set[int]]]:
    checks_x: list[set[int]] = []
    checks_z: list[set[int]] = []
    for anchor in ELEMENTS:
        checks_x.append(
            {
                *(data_index(0, add(anchor, shift)) for shift in A_SUPPORT),
                *(data_index(1, add(anchor, shift)) for shift in B_SUPPORT),
            }
        )
        checks_z.append(
            {
                *(data_index(0, add(anchor, neg(shift))) for shift in B_SUPPORT),
                *(data_index(1, add(anchor, neg(shift))) for shift in A_SUPPORT),
            }
        )
    return checks_x, checks_z


def schedule_identifier(colors: tuple[int, ...]) -> str:
    payload = json.dumps(list(colors), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def materialize() -> dict[str, Any]:
    layers: list[list[dict[str, Any]]] = [[] for _ in range(DEPTH)]
    summaries: list[dict[str, Any]] = [{} for _ in range(DEPTH)]
    for orbit, (half, shift) in enumerate(EDGE_ORBITS):
        x_layer = EDGE_ORBIT_COLORS[orbit]
        z_layer = DEPTH - 1 - x_layer
        z_half = 1 - half
        z_shift = neg(shift)
        summaries[x_layer].update(
            {
                "x_half": "L" if half == 0 else "R",
                "x_displacement": list(shift),
                "x_term": term_name(shift),
            }
        )
        summaries[z_layer].update(
            {
                "z_half": "L" if z_half == 0 else "R",
                "z_displacement": list(z_shift),
                "z_term": term_name(z_shift),
            }
        )
        for check, anchor in enumerate(ELEMENTS):
            layers[x_layer].append(
                {
                    "type": "X",
                    "check": check,
                    "data": data_index(half, add(anchor, shift)),
                }
            )
            z_anchor = neg(anchor)
            layers[z_layer].append(
                {
                    "type": "Z",
                    "check": INDEX[z_anchor],
                    "data": data_index(z_half, neg(add(anchor, shift))),
                }
            )

    result = {
        "schema_version": 1,
        "name": "depth- and CNOT-hook-fault-optimal folded BB56 schedule",
        "schedule_id": schedule_identifier(EDGE_ORBIT_COLORS),
        "code": "[[56,8,6]]",
        "group": "C4 x C7",
        "qubit_index": "half*28 + 7*x + y",
        "check_index": "7*x + y",
        "x_cnot_direction": "X-check ancilla -> data",
        "z_cnot_direction": "data -> Z-check ancilla",
        "cnot_depth": DEPTH,
        "lower_bound_from_check_weight": DEPTH,
        "depth_optimal": True,
        "cnot_count": sum(len(layer) for layer in layers),
        "dedicated_x_ancillas": NUM_CHECKS_PER_TYPE,
        "dedicated_z_ancillas": NUM_CHECKS_PER_TYPE,
        "dedicated_ancillas": 2 * NUM_CHECKS_PER_TYPE,
        "gates_per_layer": NUM_DATA_QUBITS,
        "translation_invariant_layers": True,
        "collision_free": True,
        "idle_free_entangling_layers": True,
        "clean_cross_ancilla_backaction": True,
        "fold_plus_time_reversal": True,
        "fold_data_map": "(half,x,y) -> (1-half,-x,-y)",
        "fold_check_map": "(x,y) -> (-x,-y)",
        "edge_orbit_order": [
            {
                "half": "L" if half == 0 else "R",
                "displacement": list(shift),
                "term": term_name(shift),
            }
            for half, shift in EDGE_ORBITS
        ],
        "edge_orbit_colors_zero_based": list(EDGE_ORBIT_COLORS),
        "circuit_fault_certificate": "circuit_fault_certificate.json",
        "layers": [
            {"layer": index + 1, **summaries[index], "gates": gates}
            for index, gates in enumerate(layers)
        ],
    }
    verify_structure(result)
    return result


def verify_structure(schedule: dict[str, Any]) -> None:
    layers = schedule["layers"]
    if len(layers) != DEPTH:
        raise AssertionError("wrong CNOT depth")
    checks_x, checks_z = expected_checks()
    expected = {
        (kind, check, data)
        for kind, checks in (("X", checks_x), ("Z", checks_z))
        for check, support in enumerate(checks)
        for data in support
    }
    actual = {
        (gate["type"], int(gate["check"]), int(gate["data"]))
        for layer in layers
        for gate in layer["gates"]
    }
    if actual != expected or len(actual) != 448:
        raise AssertionError("the schedule does not cover every Tanner edge exactly once")

    times: dict[tuple[str, int, int], int] = {}
    layer_sets: list[set[tuple[str, int, int]]] = []
    for layer_index, layer in enumerate(layers):
        gates = layer["gates"]
        if len(gates) != NUM_DATA_QUBITS:
            raise AssertionError("a layer is not a 56-gate perfect matching")
        data_seen = {int(gate["data"]) for gate in gates}
        ancilla_seen = {(str(gate["type"]), int(gate["check"])) for gate in gates}
        if len(data_seen) != NUM_DATA_QUBITS or len(ancilla_seen) != NUM_DATA_QUBITS:
            raise AssertionError("data or ancilla collision")
        layer_set = set()
        for gate in gates:
            key = (str(gate["type"]), int(gate["check"]), int(gate["data"]))
            times[key] = layer_index
            layer_set.add(key)
        layer_sets.append(layer_set)

    for check_x in range(NUM_CHECKS_PER_TYPE):
        for check_z in range(NUM_CHECKS_PER_TYPE):
            overlap = checks_x[check_x] & checks_z[check_z]
            inversions = sum(
                times[("X", check_x, data)] < times[("Z", check_z, data)]
                for data in overlap
            )
            if inversions % 2:
                raise AssertionError("cross-ancilla back-action parity failure")

    for layer_index, layer in enumerate(layers):
        for gate in layer["gates"]:
            if gate["type"] != "X":
                continue
            check_cell = ELEMENTS[int(gate["check"])]
            half, data_cell = data_coordinates(int(gate["data"]))
            folded = (
                "Z",
                INDEX[neg(check_cell)],
                data_index(1 - half, neg(data_cell)),
            )
            if folded not in layer_sets[DEPTH - 1 - layer_index]:
                raise AssertionError("fold plus time-reversal symmetry failure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    expected = materialize()
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        path = args.output
    else:
        path = args.verify
        saved = json.loads(path.read_text(encoding="utf-8"))
        verify_structure(saved)
        if saved != expected:
            raise AssertionError("saved schedule differs from canonical reconstruction")
    print(
        json.dumps(
            {
                "verified": True,
                "path": str(path.resolve()),
                "schedule_id": expected["schedule_id"],
                "cnot_depth": expected["cnot_depth"],
                "cnot_count": expected["cnot_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
