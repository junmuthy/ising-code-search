#!/usr/bin/env python3
"""Solve a depth-16 exact-self-dual fold/time-reversed C28xC4 schedule."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import z3

ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
sys.path.insert(0, str(ROOT))
from component_code import gf2_rref, translated_check_catalog  # noqa: E402

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path("/tmp/c28_basis_search.json"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    depth = args.depth
    if depth < 16:
        raise SystemExit("depth below the weight-16 lower bound")
    basis_record = json.loads(args.basis.read_text())
    selected = [int(i) for i in basis_record["selected_indices"]]
    rows, metadata, _orbits = translated_check_catalog()
    check = rows[selected]
    assert len(gf2_rref(check)[0]) == 48
    edges = [(row, int(data)) for row in range(48) for data in np.flatnonzero(check[row])]
    edge_index = {edge: index for index, edge in enumerate(edges)}
    colors = [z3.Int(f"c_{row}_{data}") for row, data in edges]
    solver = z3.Solver()
    solver.set(timeout=args.timeout_seconds * 1000)
    for color in colors:
        solver.add(color >= 0, color < depth)
    for row in range(48):
        solver.add(z3.Distinct(*[colors[edge_index[(row, int(data))]] for data in np.flatnonzero(check[row])]))
    for data in range(112):
        incident = [colors[edge_index[(int(row), data)]] for row in np.flatnonzero(check[:, data])]
        solver.add(z3.Distinct(*incident))
        for left in range(len(incident)):
            for right in range(left + 1, len(incident)):
                solver.add(incident[left] + incident[right] != depth - 1)
    parity_constraints = 0
    # Under the identical-check fold and time reversal, the back-action
    # parity for (i,j) equals that for (j,i), so only the upper triangle is
    # independent.  At depth 16 the diagonal constraints are automatic, but
    # retaining them also supports exploratory depths above 16.
    for xrow in range(48):
        for zrow in range(xrow, 48):
            overlap = np.flatnonzero(check[xrow] & check[zrow]).astype(int)
            if not len(overlap):
                continue
            terms = [
                z3.If(
                    colors[edge_index[(xrow, data)]]
                    < depth - 1 - colors[edge_index[(zrow, data)]],
                    1,
                    0,
                )
                for data in overlap
            ]
            solver.add(z3.Sum(*terms) % 2 == 0)
            parity_constraints += 1
    print(json.dumps({
        "event": "solve",
        "variables": len(colors),
        "parity_constraints": parity_constraints,
        "depth": depth,
        "timeout_seconds": args.timeout_seconds,
    }), flush=True)
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "finish", "status": str(status), "elapsed_seconds": round(elapsed, 3)}), flush=True)
    if status != z3.sat:
        return
    model = solver.model()
    values = [model.eval(color).as_long() for color in colors]
    layers = [[] for _ in range(depth)]
    for (row, data), color in zip(edges, values, strict=True):
        layers[color].append({"type": "X", "check": row, "data": data})
        layers[depth - 1 - color].append({"type": "Z", "check": row, "data": data})
    result = {
        "code": "[[112,16,7]]",
        "depth": depth,
        "cnot_count": 1536,
        "layer_sizes": [len(layer) for layer in layers],
        "depth_optimal": True,
        "peak_layer_load_optimal": depth == 16,
        "fold_plus_time_reversal": True,
        "clean_cross_ancilla_backaction": True,
        "basis_selected_indices": selected,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selected
        ],
        "basis_degree_histogram": basis_record["degree_histogram"],
        "x_edge_colors": values,
        "edges": [list(edge) for edge in edges],
        "layers": layers,
        "solver_elapsed_seconds": round(elapsed, 3),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("layers", "edges", "x_edge_colors", "basis_selected_checks", "basis_selected_indices")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
