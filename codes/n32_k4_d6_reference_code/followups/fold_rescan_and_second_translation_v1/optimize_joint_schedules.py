#!/usr/bin/env python3
"""Search clean, C4-invariant, fold/time-reversal joint schedules for all folds."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import z3


NUM_QUBITS = 32
NUM_CHECKS = 16
LOGICAL_ORDER = 4
THICKNESS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-depth", type=int, default=12)
    parser.add_argument("--maximum-depth", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def matrix_from_supports(supports: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(supports), NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def act_rows(matrix: np.ndarray, permutation: list[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def build_edge_orbits(
    checks: np.ndarray, check_action: list[int]
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    data_action = [
        ((data // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + data % THICKNESS
        for data in range(NUM_QUBITS)
    ]
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
        orbit = []
        current = edge
        while current not in orbit:
            if current not in edges:
                raise RuntimeError("translation leaves Tanner graph")
            orbit.append(current)
            current = (check_action[current[0]], data_action[current[1]])
        if len(orbit) != LOGICAL_ORDER:
            raise RuntimeError("non-free Tanner edge orbit")
        index = len(orbits)
        orbits.append(orbit)
        for member in orbit:
            edge_orbit[member] = index
    return orbits, edge_orbit


def solve_depth(
    checks_x: np.ndarray,
    permutation: list[int],
    edge_orbit: dict[tuple[int, int], int],
    orbit_count: int,
    depth: int,
    timeout_seconds: float,
) -> tuple[str, list[int] | None, float]:
    checks_z = act_rows(checks_x, permutation)
    variables = [z3.Int(f"orbit_{index}") for index in range(orbit_count)]
    solver = z3.Solver()
    solver.set(timeout=max(1, round(timeout_seconds * 1000)))
    for variable in variables:
        solver.add(variable >= 0, variable < depth)

    # X/X collisions.  Z/Z follows because P is a permutation and the Z
    # schedule is defined as the folded time reverse of X.
    for check in range(NUM_CHECKS):
        incident = [
            variables[edge_orbit[(check, int(data))]]
            for data in np.flatnonzero(checks_x[check])
        ]
        solver.add(z3.Distinct(*incident))
    for data in range(NUM_QUBITS):
        incident = [
            variables[edge_orbit[(check, data)]]
            for check in range(NUM_CHECKS)
            if checks_x[check, data]
        ]
        solver.add(z3.Distinct(*incident))

    # At target data q, a Z edge is the fold of an X edge incident on P(q).
    for data in range(NUM_QUBITS):
        x_incident = [
            variables[edge_orbit[(check, data)]]
            for check in range(NUM_CHECKS)
            if checks_x[check, data]
        ]
        source = permutation[data]
        z_incident = [
            depth - 1 - variables[edge_orbit[(check, source)]]
            for check in range(NUM_CHECKS)
            if checks_x[check, source]
        ]
        solver.add(z3.Distinct(*(x_incident + z_incident)))

    # Exact clean-cross-ancilla parity condition used by the saved analyzer.
    for check_x in range(NUM_CHECKS):
        for check_z in range(NUM_CHECKS):
            comparisons = []
            for data in range(NUM_QUBITS):
                if checks_x[check_x, data] and checks_z[check_z, data]:
                    x_time = variables[edge_orbit[(check_x, data)]]
                    source = permutation[data]
                    z_time = depth - 1 - variables[edge_orbit[(check_z, source)]]
                    comparisons.append(x_time < z_time)
            if len(comparisons) % 2:
                raise RuntimeError("input checks are not CSS orthogonal")
            if comparisons:
                parity = comparisons[0]
                for comparison in comparisons[1:]:
                    parity = z3.Xor(parity, comparison)
                solver.add(z3.Not(parity))

    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    if status != z3.sat:
        return str(status), None, elapsed
    model = solver.model()
    colors = [model.eval(variable).as_long() for variable in variables]
    return "sat", colors, elapsed


def build_schedule(
    checks_x: np.ndarray,
    permutation: list[int],
    edge_orbit: dict[tuple[int, int], int],
    colors: list[int],
    depth: int,
) -> dict[str, Any]:
    checks_z = act_rows(checks_x, permutation)
    layers: list[dict[str, Any]] = []
    for time_index in range(depth):
        gates = []
        for check, row in enumerate(checks_x):
            for data_value in np.flatnonzero(row):
                data = int(data_value)
                if colors[edge_orbit[(check, data)]] == time_index:
                    gates.append({"type": "X", "check": check, "data": data})
                source = data
                target = permutation[source]
                if depth - 1 - colors[edge_orbit[(check, source)]] == time_index:
                    gates.append({"type": "Z", "check": check, "data": target})
        layers.append({"round": time_index + 1, "gates": gates})

    expected = {
        ("X", check, int(data))
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row)
    } | {
        ("Z", check, int(data))
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row)
    }
    observed = {
        (gate["type"], gate["check"], gate["data"])
        for layer in layers
        for gate in layer["gates"]
    }
    if expected != observed:
        raise RuntimeError("schedule edge coverage failure")
    for layer in layers:
        ancillas = set()
        data_seen = set()
        for gate in layer["gates"]:
            ancilla = (gate["type"], gate["check"])
            if ancilla in ancillas or gate["data"] in data_seen:
                raise RuntimeError("schedule collision")
            ancillas.add(ancilla)
            data_seen.add(gate["data"])
    times = {
        (gate["type"], gate["check"], gate["data"]): time_index
        for time_index, layer in enumerate(layers)
        for gate in layer["gates"]
    }
    for check_x in range(NUM_CHECKS):
        for check_z in range(NUM_CHECKS):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z])
            inversions = sum(
                times[("X", check_x, int(data))]
                < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                raise RuntimeError("cross-ancilla backaction failure")
    return {
        "schema_version": 1,
        "name": "clean simultaneous C4-invariant fold/time-reversal schedule",
        "cnot_depth": depth,
        "cnot_count": len(expected),
        "dedicated_ancillas": 32,
        "translation_invariant_layers": True,
        "fold_P_plus_time_reversal": True,
        "clean_cross_ancilla_backaction": True,
        "edge_orbit_colors_zero_based": colors,
        "layers": layers,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    presentation = json.loads(args.presentation.read_text())
    checks_x = matrix_from_supports(presentation["checks_x"])
    check_action = presentation["translation_action_on_x_checks"]
    orbits, edge_orbit = build_edge_orbits(checks_x, check_action)
    folds = [json.loads(line) for line in args.folds.read_text().splitlines() if line.strip()]
    results = []
    best: tuple[int, int, list[int]] | None = None
    started = time.perf_counter()
    for fold_index, fold in enumerate(folds):
        attempts = []
        for depth in range(args.minimum_depth, args.maximum_depth + 1):
            status, colors, seconds = solve_depth(
                checks_x,
                fold["permutation"],
                edge_orbit,
                len(orbits),
                depth,
                args.timeout_seconds,
            )
            attempts.append({"depth": depth, "status": status, "seconds": round(seconds, 6)})
            if colors is not None:
                if best is None or depth < best[0]:
                    best = (depth, fold_index, colors)
                break
        results.append({"fold_index": fold_index, "attempts": attempts})
        print(
            f"checkpoint fold={fold_index + 1}/{len(folds)} "
            f"best={None if best is None else best[0]} "
            f"last={attempts[-1]['depth']}:{attempts[-1]['status']} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )

    schedules = []
    if best is not None:
        best_depth = best[0]
        for result in results:
            sat_attempt = next((item for item in result["attempts"] if item["status"] == "sat"), None)
            if sat_attempt is None or sat_attempt["depth"] != best_depth:
                continue
            fold_index = result["fold_index"]
            status, colors, seconds = solve_depth(
                checks_x,
                folds[fold_index]["permutation"],
                edge_orbit,
                len(orbits),
                best_depth,
                max(args.timeout_seconds, 60),
            )
            if status != "sat" or colors is None:
                raise RuntimeError("failed to reproduce saved satisfiable depth")
            schedule = build_schedule(
                checks_x,
                folds[fold_index]["permutation"],
                edge_orbit,
                colors,
                best_depth,
            )
            schedule["fold_index"] = fold_index
            schedule["fold"] = folds[fold_index]
            schedule["solver_seconds_reproduction"] = round(seconds, 6)
            schedules.append(schedule)
            break
        (args.output / "best-schedule.json").write_text(
            json.dumps(schedules[0], indent=2, sort_keys=True) + "\n"
        )

    status_histogram = Counter(
        attempt["status"]
        for result in results
        for attempt in result["attempts"]
    )
    summary = {
        "schema_version": 1,
        "scope": "all 24 connected distance-six affine folds, C4-invariant layers, fold P plus time reversal, and exact clean-backaction constraints",
        "fold_count": len(folds),
        "translated_x_edge_orbits": len(orbits),
        "depth_range": [args.minimum_depth, args.maximum_depth],
        "timeout_seconds_per_attempt": args.timeout_seconds,
        "best_depth": None if best is None else best[0],
        "best_fold_index": None if best is None else best[1],
        "solver_status_histogram": dict(sorted(status_histogram.items())),
        "results": results,
        "seconds": round(time.perf_counter() - started, 6),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
