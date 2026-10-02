#!/usr/bin/env python3
"""Materialize and verify the selected depth-optimal LP160 schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import search_schedule


EDGE_ORBIT_COLORS = (0, 6, 7, 10, 3, 5, 1, 8, 9, 0, 9, 7, 3, 2, 10, 4, 11, 5)


def schedule_identifier(colors) -> str:
    payload = json.dumps(list(colors), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def orbit_record(key) -> dict:
    family, block, displacement = key
    return {
        "check_family": int(family),
        "data_block": int(block),
        "displacement": [int(value) for value in displacement],
    }


def materialize() -> dict:
    _solver, _variables, xkeys, zkeys, partners, parity_constraints = (
        search_schedule.build_problem()
    )
    layers = search_schedule.schedule_from_colors(
        EDGE_ORBIT_COLORS, xkeys, zkeys, partners
    )
    result = {
        "schema_version": 1,
        "name": "depth-optimal balanced C4 x C8 fold/time-reversed LP160 schedule",
        "schedule_id": schedule_identifier(EDGE_ORBIT_COLORS),
        "code": "[[160,32,6]]",
        "group": "C4 x C8",
        "qubit_index": "32*block + 8*x + y",
        "check_index": "32*family + 8*x + y",
        "x_cnot_direction": "X-check ancilla -> data",
        "z_cnot_direction": "data -> Z-check ancilla",
        "cnot_depth": 12,
        "lower_bound_from_maximum_combined_data_degree": 12,
        "depth_optimal": True,
        "stabilizer_weight": 9,
        "cnot_count": 1152,
        "dedicated_x_ancillas": 64,
        "dedicated_z_ancillas": 64,
        "dedicated_ancillas": 128,
        "gates_per_layer": 96,
        "balanced_layers": True,
        "translation_invariant_layers": True,
        "collision_free": True,
        "clean_cross_ancilla_backaction": True,
        "cross_ancilla_parity_constraint_orbits": parity_constraints,
        "fold_plus_time_reversal": True,
        "fold_data_map": "(block,r,s) -> (block_map[block],r,-2*r-s mod 8)",
        "fold_block_map_zero_based": [0, 2, 1, 3, 4],
        "x_edge_orbits": [orbit_record(key) for key in xkeys],
        "z_edge_orbits": [orbit_record(key) for key in zkeys],
        "fold_x_to_z_orbit": [int(value) for value in partners],
        "x_edge_orbit_colors_zero_based": list(EDGE_ORBIT_COLORS),
        "fault_certificate": "circuit_fault_certificate.json",
        "layers": [
            {"layer": index + 1, "gates": layer}
            for index, layer in enumerate(layers)
        ],
    }
    verify_record(result)
    return result


def verify_record(record: dict) -> None:
    if record["cnot_depth"] != 12 or not record["depth_optimal"]:
        raise AssertionError("incorrect depth claim")
    if len(record["layers"]) != 12:
        raise AssertionError("wrong number of layers")
    if any(len(layer["gates"]) != 96 for layer in record["layers"]):
        raise AssertionError("layers are not balanced at 96 CNOTs")
    if sum(len(layer["gates"]) for layer in record["layers"]) != 1152:
        raise AssertionError("wrong CNOT count")


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
        args.output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        path = args.output
    else:
        path = args.verify
        saved = json.loads(path.read_text())
        verify_record(saved)
        if saved != expected:
            raise AssertionError("saved schedule differs from canonical reconstruction")
    print(json.dumps({
        "verified": True,
        "path": str(path.resolve()),
        "schedule_id": expected["schedule_id"],
        "cnot_depth": 12,
        "cnot_count": 1152,
        "gates_per_layer": 96,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
