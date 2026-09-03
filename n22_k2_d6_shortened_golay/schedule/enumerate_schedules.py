#!/usr/bin/env python3
"""Exhaust all cyclic, fold/time-reversed depth-eight syndrome schedules."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from n22_k2_d6_shortened_golay.code_data.code import (
    A_SUPPORT,
    B_SUPPORT,
    ELL,
    NUM_CHECKS_PER_TYPE,
    NUM_DATA_QUBITS,
    build_checks,
    fold_check_action,
    fold_permutation,
    validate_code,
)


DEPTH = 8
EDGE_TYPES = tuple((0, exponent) for exponent in A_SUPPORT) + tuple(
    (1, exponent) for exponent in B_SUPPORT
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--save-valid", type=int, default=32)
    return parser.parse_args()


def edge_orbits() -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    orbits: list[list[tuple[int, int]]] = []
    lookup: dict[tuple[int, int], int] = {}
    for orbit_index, (half, exponent) in enumerate(EDGE_TYPES):
        orbit = [
            (check, half * ELL + (check + exponent) % ELL)
            for check in range(ELL)
        ]
        orbits.append(orbit)
        for edge in orbit:
            lookup[edge] = orbit_index
    return orbits, lookup


def reverse_colors(colors: Iterable[int]) -> tuple[int, ...]:
    return tuple(DEPTH - 1 - int(color) for color in colors)


def schedule_id(colors: Iterable[int]) -> str:
    payload = json.dumps(list(colors), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def z_time(
    check: int,
    data: int,
    colors: tuple[int, ...],
    edge_lookup: dict[tuple[int, int], int],
) -> int:
    source_check = fold_check_action()[check]
    source_data = fold_permutation()[data]
    return DEPTH - 1 - colors[edge_lookup[(source_check, source_data)]]


def materialize_schedule(
    colors: tuple[int, ...],
    edge_lookup: dict[tuple[int, int], int],
) -> dict[str, Any] | None:
    checks_x, checks_z = build_checks()
    times_x = {
        (check, data): colors[edge_lookup[(check, data)]]
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row).astype(int)
    }
    times_z = {
        (check, data): z_time(check, data, colors, edge_lookup)
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row).astype(int)
    }
    for data in range(NUM_DATA_QUBITS):
        times = [time_index for (check, qubit), time_index in times_x.items() if qubit == data]
        times += [time_index for (check, qubit), time_index in times_z.items() if qubit == data]
        if len(times) != DEPTH or len(set(times)) != DEPTH:
            return None

    for check_x in range(NUM_CHECKS_PER_TYPE):
        for check_z in range(NUM_CHECKS_PER_TYPE):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z]).astype(int)
            inversions = sum(times_x[(check_x, data)] < times_z[(check_z, data)] for data in overlap)
            if inversions % 2:
                return None

    layers: list[dict[str, Any]] = []
    for layer_index in range(DEPTH):
        gates: list[dict[str, int | str]] = []
        for (check, data), time_index in times_x.items():
            if time_index == layer_index:
                gates.append({"type": "X", "check": int(check), "data": int(data)})
        for (check, data), time_index in times_z.items():
            if time_index == layer_index:
                gates.append({"type": "Z", "check": int(check), "data": int(data)})
        if len(gates) != NUM_DATA_QUBITS:
            raise AssertionError("optimal layer is not a perfect matching")
        if len({int(gate["data"]) for gate in gates}) != NUM_DATA_QUBITS:
            raise AssertionError("data collision")
        if len({(str(gate["type"]), int(gate["check"])) for gate in gates}) != NUM_DATA_QUBITS:
            raise AssertionError("ancilla collision")
        layers.append({"layer": layer_index, "gates": gates})

    return {
        "schema_version": 1,
        "name": "cyclic fold/time-reversed shortened-Golay syndrome schedule",
        "schedule_id": schedule_id(colors),
        "code": "[[22,2,6]]",
        "cnot_depth": DEPTH,
        "cnot_count": 176,
        "dedicated_ancillas": 22,
        "cyclic_C11_invariant_layers": True,
        "fold_plus_time_reversal": True,
        "collision_free": True,
        "idle_free_entangling_layers": True,
        "clean_cross_ancilla_backaction": True,
        "edge_types": [list(edge_type) for edge_type in EDGE_TYPES],
        "edge_orbit_colors_zero_based": list(colors),
        "zx_fold": list(fold_permutation()),
        "zx_fold_check_action": list(fold_check_action()),
        "layers": layers,
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    code_summary = validate_code(exhaustive=True)
    orbits, lookup = edge_orbits()
    if len(orbits) != DEPTH or any(len(orbit) != ELL for orbit in orbits):
        raise AssertionError("unexpected Tanner-edge orbit structure")

    started = time.perf_counter()
    tested = 0
    canonical = 0
    collision_free = 0
    clean = 0
    saved: list[str] = []
    records_path = args.output_dir / "valid_schedules.jsonl"
    with records_path.open("w", encoding="utf-8") as records:
        for color_value in itertools.permutations(range(DEPTH)):
            tested += 1
            colors = tuple(int(value) for value in color_value)
            if tested % args.checkpoint_every == 0:
                checkpoint = {
                    "tested": tested,
                    "time_reversal_canonical": canonical,
                    "collision_free_and_clean": clean,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
                write_json(args.output_dir / "checkpoint.json", checkpoint)
                print(f"checkpoint {json.dumps(checkpoint, sort_keys=True)}", flush=True)
            if colors > reverse_colors(colors):
                continue
            canonical += 1
            schedule = materialize_schedule(colors, lookup)
            if schedule is None:
                continue
            collision_free += 1
            clean += 1
            records.write(
                json.dumps(
                    {
                        "schedule_id": schedule["schedule_id"],
                        "edge_orbit_colors_zero_based": list(colors),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            records.flush()
            if clean == 1:
                write_json(args.output_dir / "baseline_schedule.json", schedule)
            if len(saved) < args.save_valid:
                path = args.output_dir / f"schedule-{clean:04d}-{schedule['schedule_id']}.json"
                write_json(path, schedule)
                saved.append(str(path))

    summary = {
        "schema_version": 1,
        "code": code_summary,
        "lower_bound_reason": "every data qubit participates in eight CNOTs per simultaneous X/Z round",
        "proven_minimum_cnot_depth": DEPTH,
        "edge_orbits": len(orbits),
        "raw_colorings": tested,
        "time_reversal_canonical_colorings": canonical,
        "collision_free_and_clean_schedules": clean,
        "saved_complete_schedules": saved,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
