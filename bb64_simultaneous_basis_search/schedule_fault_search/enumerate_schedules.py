#!/usr/bin/env python3
"""Enumerate clean full-translation-invariant depth-eight BB64 schedules."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bb64_simultaneous_basis_search.stim_fault_distance.model import (
    BASIS_PATH,
    NUM_CHECKS_PER_TYPE,
    NUM_DATA_QUBITS,
)
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import atomic_json, utc_now


Element = tuple[int, int]
ELEMENTS: tuple[Element, ...] = tuple((xx, yy) for xx in range(4) for yy in range(8))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
DEPTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-npz", type=Path, default=BASIS_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--save-valid", type=int, default=32)
    return parser.parse_args()


def add(left: Element, right: Element) -> Element:
    raw_x = left[0] + right[0]
    return raw_x % 4, (left[1] + right[1] + 4 * (raw_x // 4)) % 8


def translate_check(check: int, translation: Element) -> int:
    return INDEX[add(translation, ELEMENTS[check])]


def translate_data(data: int, translation: Element) -> int:
    half, within = divmod(data, 32)
    return 32 * half + INDEX[add(translation, ELEMENTS[within])]


def build_edge_orbits(checks: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    edges = {
        (check, int(data))
        for check, row in enumerate(checks)
        for data in np.flatnonzero(row)
    }
    orbits: list[list[tuple[int, int]]] = []
    edge_orbit: dict[tuple[int, int], int] = {}
    for edge in sorted(edges):
        if edge in edge_orbit:
            continue
        orbit = sorted(
            {
                (translate_check(edge[0], translation), translate_data(edge[1], translation))
                for translation in ELEMENTS
            }
        )
        if len(orbit) != 32 or not set(orbit).issubset(edges):
            raise RuntimeError("Tanner edges do not form free 32-element translation orbits")
        orbit_index = len(orbits)
        orbits.append(orbit)
        for member in orbit:
            edge_orbit[member] = orbit_index
    if len(orbits) != 8 or len(edge_orbit) != 256:
        raise RuntimeError(f"expected eight edge orbits covering 256 edges, got {len(orbits)}")
    return orbits, edge_orbit


def reverse_colors(colors: Iterable[int]) -> tuple[int, ...]:
    return tuple(DEPTH - 1 - int(color) for color in colors)


def schedule_id(colors: Iterable[int]) -> str:
    payload = json.dumps(list(colors), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def materialize_schedule(
    checks: np.ndarray,
    edge_orbit: dict[tuple[int, int], int],
    colors: tuple[int, ...],
) -> dict[str, Any] | None:
    # At a data qubit, the four X colors and four reversed Z colors must be
    # distinct. X/X, Z/Z, and ancilla collisions are excluded by orbit colors.
    for data in range(NUM_DATA_QUBITS):
        x_colors = [
            colors[edge_orbit[(check, data)]]
            for check in range(NUM_CHECKS_PER_TYPE)
            if checks[check, data]
        ]
        z_colors = [DEPTH - 1 - color for color in x_colors]
        if len(set(x_colors + z_colors)) != DEPTH:
            return None

    layers: list[dict[str, Any]] = []
    for layer_index in range(DEPTH):
        gates: list[dict[str, Any]] = []
        for check, row in enumerate(checks):
            for data_value in np.flatnonzero(row):
                data = int(data_value)
                color = colors[edge_orbit[(check, data)]]
                if color == layer_index:
                    gates.append({"type": "X", "check": check, "data": data})
                if DEPTH - 1 - color == layer_index:
                    gates.append({"type": "Z", "check": check, "data": data})
        if len(gates) != 64:
            raise RuntimeError("a depth-eight layer is not a perfect matching")
        layers.append({"round": layer_index + 1, "gates": gates})

    times = {
        (gate["type"], gate["check"], gate["data"]): layer_index
        for layer_index, layer in enumerate(layers)
        for gate in layer["gates"]
    }
    for check_x in range(NUM_CHECKS_PER_TYPE):
        for check_z in range(NUM_CHECKS_PER_TYPE):
            overlap = np.flatnonzero(checks[check_x] & checks[check_z])
            inversions = sum(
                times[("X", check_x, int(data))]
                < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                return None

    return {
        "schema_version": 1,
        "name": "clean full-translation-invariant BB64 identity-fold schedule",
        "schedule_id": schedule_id(colors),
        "code": "[[64,8,8]]",
        "cnot_depth": DEPTH,
        "cnot_count": 512,
        "dedicated_ancillas": 64,
        "identity_zx_fold": True,
        "fold_plus_time_reversal": True,
        "full_group_translation_invariant_layers": True,
        "collision_free": True,
        "idle_free_entangling_layers": True,
        "clean_cross_ancilla_backaction": True,
        "edge_orbit_colors_zero_based": list(colors),
        "layers": layers,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    archive = np.load(args.basis_npz)
    checks = np.asarray(archive["matrix_x"], dtype=np.uint8)
    orbits, edge_orbit = build_edge_orbits(checks)
    orbit_representatives = [list(orbit[0]) for orbit in orbits]
    started = time.perf_counter()
    tested = 0
    time_reversal_canonical = 0
    collision_free = 0
    clean = 0
    saved = []
    records_path = args.output_dir / "valid_schedules.jsonl"
    with records_path.open("w", encoding="utf-8") as records:
        for colors_value in itertools.permutations(range(DEPTH)):
            tested += 1
            colors = tuple(int(value) for value in colors_value)
            if colors > reverse_colors(colors):
                continue
            time_reversal_canonical += 1
            # Count collision-free colorings separately for diagnostics.
            collision_ok = True
            for data in range(NUM_DATA_QUBITS):
                x_colors = [
                    colors[edge_orbit[(check, data)]]
                    for check in range(NUM_CHECKS_PER_TYPE)
                    if checks[check, data]
                ]
                if len(set(x_colors + [DEPTH - 1 - color for color in x_colors])) != DEPTH:
                    collision_ok = False
                    break
            if not collision_ok:
                continue
            collision_free += 1
            schedule = materialize_schedule(checks, edge_orbit, colors)
            if schedule is None:
                continue
            clean += 1
            compact = {
                "schedule_id": schedule["schedule_id"],
                "edge_orbit_colors_zero_based": schedule["edge_orbit_colors_zero_based"],
            }
            records.write(json.dumps(compact, sort_keys=True) + "\n")
            records.flush()
            if len(saved) < args.save_valid:
                path = args.output_dir / f"schedule-{clean:04d}-{schedule['schedule_id']}.json"
                atomic_json(path, schedule)
                saved.append(str(path))
            if clean == 1:
                atomic_json(args.output_dir / "baseline_schedule.json", schedule)
            if tested % args.checkpoint_every == 0:
                checkpoint = {
                    "timestamp": utc_now(),
                    "tested": tested,
                    "time_reversal_canonical": time_reversal_canonical,
                    "collision_free": collision_free,
                    "clean": clean,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
                atomic_json(args.output_dir / "checkpoint.json", checkpoint)
                print(f"checkpoint {json.dumps(checkpoint, sort_keys=True)}", flush=True)
    summary = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "basis": str(args.basis_npz),
        "edge_orbits": len(orbits),
        "orbit_representatives": orbit_representatives,
        "tested": tested,
        "time_reversal_canonical": time_reversal_canonical,
        "collision_free": collision_free,
        "clean": clean,
        "saved_complete_schedules": saved,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
