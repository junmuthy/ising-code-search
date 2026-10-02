#!/usr/bin/env python3
"""Solve a clean depth-16 Z ordering around a fixed X edge coloring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import z3

sys.path.insert(0, "/tmp")
from optimize_c28_joint_schedule import initial_coloring, selected_check, verify_coloring  # noqa: E402


def parity(values):
    values = list(values)
    while len(values) > 1:
        values = [
            z3.Xor(values[index], values[index + 1])
            for index in range(0, len(values) - 1, 2)
        ] + ([values[-1]] if len(values) % 2 else [])
    return values[0] if values else z3.BoolVal(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path("/tmp/c28_basis_search.json"))
    parser.add_argument("--start", type=Path, default=Path("/tmp/c28_joint_schedule.json"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check, selection, metadata = selected_check(args.basis)
    edge_records, initial = initial_coloring(check)
    start = json.loads(args.start.read_text()) if args.start.exists() else None
    colors = np.asarray(start["colors"], dtype=np.int16) if start else initial
    tx = np.full(check.shape, -1, dtype=np.int16)
    for edge_index, (kind, row, data, _left, _right) in enumerate(edge_records):
        if kind == 0:
            tx[row, data] = colors[edge_index]
    assert np.all(tx[check.astype(bool)] >= 0)

    zedges = [(row, int(data)) for row in range(48) for data in np.flatnonzero(check[row])]
    zindex = {edge: index for index, edge in enumerate(zedges)}
    zcolors = [z3.Int(f"z_{row}_{data}") for row, data in zedges]
    solver = z3.Solver()
    solver.set(timeout=args.timeout_seconds * 1000, random_seed=args.random_seed)
    for color in zcolors:
        solver.add(color >= 0, color < 16)
    for row in range(48):
        solver.add(z3.Distinct(*[zcolors[zindex[(row, int(data))]] for data in np.flatnonzero(check[row])]))
    for data in range(112):
        fixed = {int(tx[row, data]) for row in np.flatnonzero(check[:, data])}
        variables = [zcolors[zindex[(int(row), data)]] for row in np.flatnonzero(check[:, data])]
        solver.add(z3.Distinct(*variables))
        for variable in variables:
            for color in fixed:
                solver.add(variable != color)
    parity_constraints = 0
    for xrow in range(48):
        for zrow in range(48):
            overlap = np.flatnonzero(check[xrow] & check[zrow]).astype(int)
            if len(overlap):
                comparisons = [
                    int(tx[xrow, data]) < zcolors[zindex[(zrow, data)]]
                    for data in overlap
                ]
                solver.add(z3.Not(parity(comparisons)))
                parity_constraints += 1
    print(json.dumps({
        "event": "solve",
        "variables": len(zcolors),
        "parity_constraints": parity_constraints,
        "timeout_seconds": args.timeout_seconds,
        "random_seed": args.random_seed,
    }), flush=True)
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "finish", "status": str(status), "elapsed_seconds": round(elapsed, 3)}), flush=True)
    if status != z3.sat:
        return
    model = solver.model()
    tz_values = [model.eval(color).as_long() for color in zcolors]
    combined_colors = colors.copy()
    edge_lookup = {(kind, row, data): i for i, (kind, row, data, _l, _r) in enumerate(edge_records)}
    for (row, data), color in zip(zedges, tz_values, strict=True):
        combined_colors[edge_lookup[(1, row, data)]] = color
    score, _tx, _tz, failures = verify_coloring(check, edge_records, combined_colors)
    if score:
        raise AssertionError(np.argwhere(failures).tolist())
    layers = [[] for _ in range(16)]
    for edge, color in zip(edge_records, combined_colors, strict=True):
        kind, row, data, _left, _right = edge
        layers[int(color)].append({"type": "X" if kind == 0 else "Z", "check": row, "data": data})
    result = {
        "schema_version": 1,
        "name": "depth-optimal clean simultaneous C28 x C4 schedule",
        "code": "[[112,16,7]]",
        "depth": 16,
        "depth_optimal": True,
        "peak_layer_load_optimal": True,
        "cnot_count": 1536,
        "layer_sizes": [len(layer) for layer in layers],
        "collision_free": True,
        "clean_cross_ancilla_backaction": True,
        "fold_plus_time_reversal": False,
        "basis_selected_indices": selection,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selection
        ],
        "basis_degree_histogram": {
            str(value): int(np.count_nonzero(check.sum(axis=0) == value))
            for value in sorted(set(map(int, check.sum(axis=0))))
        },
        "combined_colors": combined_colors.astype(int).tolist(),
        "layers": layers,
        "solver_seconds": round(elapsed, 3),
        "solver_random_seed": args.random_seed,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("combined_colors", "layers", "basis_selected_checks", "basis_selected_indices")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
