#!/usr/bin/env python3
"""Find a clean joint schedule by ordering every local Z edge before every X edge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, "/tmp")
from optimize_c28_joint_schedule import initial_coloring, selected_check, verify_coloring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    depth = args.depth
    check, selection, metadata = selected_check(Path("/tmp/c28_basis_search.json"))
    edge_records, _colors = initial_coloring(check)
    edge_count = len(edge_records)
    slot_count = edge_count * depth
    variable_count = slot_count + 112
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    degrees = check.sum(axis=0).astype(int)
    for data, degree in enumerate(degrees):
        lower[slot_count + data] = degree - 1
        upper[slot_count + data] = depth - degree - 1
    integrality = np.ones(variable_count, dtype=np.uint8)

    rows = []
    cols = []
    vals = []
    lows = []
    highs = []

    def constraint(entries, low=-np.inf, high=np.inf):
        row = len(lows)
        for column, value in entries:
            rows.append(row)
            cols.append(column)
            vals.append(value)
        lows.append(low)
        highs.append(high)

    def slot(edge, time_index):
        return edge * depth + time_index

    for edge in range(edge_count):
        constraint(((slot(edge, t), 1) for t in range(depth)), 1, 1)
    by_ancilla = {}
    by_data = {data: [] for data in range(112)}
    for edge, (kind, check_index, data, _left, _right) in enumerate(edge_records):
        by_ancilla.setdefault((kind, check_index), []).append(edge)
        by_data[data].append(edge)
    for incident in by_ancilla.values():
        for t in range(depth):
            constraint(((slot(edge, t), 1) for edge in incident), high=1)
    for incident in by_data.values():
        for t in range(depth):
            constraint(((slot(edge, t), 1) for edge in incident), high=1)
    for edge, (kind, _check_index, data, _left, _right) in enumerate(edge_records):
        q = slot_count + data
        if kind == 1:
            constraint([*((slot(edge, t), t) for t in range(depth)), (q, -1)], high=0)
        else:
            constraint([*((slot(edge, t), -t) for t in range(depth)), (q, 1)], high=-1)

    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lows), variable_count)).tocsr()
    print(json.dumps({"event": "solve", "depth": depth, "binary_slot_variables": slot_count, "threshold_variables": 112, "constraints": len(lows), "nonzeros": matrix.nnz}), flush=True)
    started = time.perf_counter()
    result = milp(
        np.zeros(variable_count),
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(lows), np.asarray(highs)),
        options={"time_limit": args.timeout_seconds, "presolve": True, "mip_rel_gap": 0},
    )
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "finish", "success": bool(result.success), "status": int(result.status), "message": result.message, "elapsed_seconds": round(elapsed, 3), "mip_node_count": getattr(result, "mip_node_count", None)}), flush=True)
    if result.x is None:
        return
    colors = np.empty(edge_count, dtype=np.int16)
    for edge in range(edge_count):
        colors[edge] = int(np.argmax(result.x[edge * depth : (edge + 1) * depth]))
    score, tx, tz, failures = verify_coloring(check, edge_records, colors)
    if score:
        raise AssertionError(f"parity failures despite precedence: {score}")
    for data in range(112):
        if int(tz[:, data].max()) >= int(tx[tx[:, data] >= 0, data].min()):
            raise AssertionError(f"precedence failure at data {data}")
    layers = [[] for _ in range(depth)]
    for edge, color in zip(edge_records, colors, strict=True):
        kind, row, data, _left, _right = edge
        layers[int(color)].append({"type": "X" if kind == 0 else "Z", "check": row, "data": data})
    payload = {
        "schema_version": 1,
        "name": "clean simultaneous local-Z-before-X syndrome schedule",
        "code": "[[112,16,7]]",
        "depth": depth,
        "cnot_count": edge_count,
        "layer_sizes": [len(layer) for layer in layers],
        "collision_free": True,
        "all_tanner_edges_once": True,
        "clean_cross_ancilla_backaction": True,
        "local_z_before_x": True,
        "basis_selected_indices": selection,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selection
        ],
        "colors": colors.astype(int).tolist(),
        "layers": layers,
        "solver": {"name": "scipy.optimize.milp/HiGHS", "seconds": elapsed, "status": int(result.status), "message": result.message},
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("colors", "layers", "basis_selected_indices", "basis_selected_checks")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
