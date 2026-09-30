#!/usr/bin/env python3
"""Sample X edge colorings and solve exactly for a compatible clean Z coloring."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import z3

from optimize_c28_joint_schedule import initial_coloring, selected_check, verify_coloring


def xor_all(values):
    values = list(values)
    if not values:
        return z3.BoolVal(False)
    while len(values) > 1:
        values = [
            z3.Xor(values[i], values[i + 1])
            if i + 1 < len(values)
            else values[i]
            for i in range(0, len(values), 2)
        ]
    return values[0]


def x_kempe_move(edge_records, colors, rng):
    xedges = [i for i, edge in enumerate(edge_records) if edge[0] == 0]
    start = rng.choice(xedges)
    first = int(colors[start])
    second = rng.randrange(15)
    if second >= first:
        second += 1
    incidence = {}
    for edge_index in xedges:
        edge = edge_records[edge_index]
        color = int(colors[edge_index])
        incidence[edge[3], color] = edge_index
        incidence[edge[4], color] = edge_index
    todo = [start]
    component = set()
    while todo:
        edge_index = todo.pop()
        if edge_index in component:
            continue
        component.add(edge_index)
        for node in edge_records[edge_index][3:5]:
            for color in (first, second):
                neighbor = incidence.get((node, color))
                if neighbor is not None and neighbor not in component:
                    todo.append(neighbor)
    for edge_index in component:
        colors[edge_index] = second if colors[edge_index] == first else first


def solve_z(check, edge_records, colors, depth, timeout_ms, random_seed):
    tx = np.full(check.shape, -1, dtype=np.int16)
    for edge_index, (kind, row, data, _left, _right) in enumerate(edge_records):
        if kind == 0:
            tx[row, data] = colors[edge_index]
    zedges = [(row, int(data)) for row in range(48) for data in np.flatnonzero(check[row])]
    zindex = {edge: index for index, edge in enumerate(zedges)}
    variables = [z3.Int(f"z_{row}_{data}") for row, data in zedges]
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=random_seed)
    for variable in variables:
        solver.add(variable >= 0, variable < depth)
    for row in range(48):
        solver.add(z3.Distinct(*[variables[zindex[row, int(data)]] for data in np.flatnonzero(check[row])]))
    for data in range(112):
        x_used = [int(tx[row, data]) for row in np.flatnonzero(check[:, data])]
        z_used = [variables[zindex[int(row), data]] for row in np.flatnonzero(check[:, data])]
        solver.add(z3.Distinct(*(x_used + z_used)))
    for xrow in range(48):
        for zrow in range(48):
            overlap = np.flatnonzero(check[xrow] & check[zrow])
            if len(overlap):
                solver.add(z3.Not(xor_all(
                    int(tx[xrow, data]) < variables[zindex[zrow, int(data)]]
                    for data in overlap
                )))
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    if status != z3.sat:
        return str(status), elapsed, None
    model = solver.model()
    answer = colors.copy()
    lookup = {(kind, row, data): i for i, (kind, row, data, _left, _right) in enumerate(edge_records)}
    for edge, variable in zip(zedges, variables, strict=True):
        answer[lookup[1, edge[0], edge[1]]] = model.eval(variable).as_long()
    score, _tx, _tz, _failures = verify_coloring(check, edge_records, answer)
    assert score == 0
    return "sat", elapsed, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--moves-per-trial", type=int, default=100)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=1121607)
    parser.add_argument("--output", type=Path, default=Path("/tmp/c28_clean_d16.json"))
    args = parser.parse_args()
    check, selection, metadata = selected_check(Path("/tmp/c28_basis_search.json"))
    edge_records, base = initial_coloring(check)
    rng = random.Random(args.seed)
    counts = {"sat": 0, "unsat": 0, "unknown": 0}
    started = time.perf_counter()
    for trial in range(args.trials):
        colors = base.copy()
        permutation = list(range(16))
        rng.shuffle(permutation)
        for edge_index, edge in enumerate(edge_records):
            if edge[0] == 0:
                colors[edge_index] = permutation[int(colors[edge_index])]
        for _ in range(args.moves_per_trial * trial):
            x_kempe_move(edge_records, colors, rng)
        status, seconds, answer = solve_z(
            check, edge_records, colors, args.depth, round(args.timeout_seconds * 1000), rng.randrange(2**30)
        )
        counts[status] += 1
        print(json.dumps({"trial": trial, "status": status, "solve_seconds": round(seconds, 3), "counts": counts, "total_seconds": round(time.perf_counter() - started, 3)}), flush=True)
        if answer is not None:
            layers = [[] for _ in range(args.depth)]
            for edge, color in zip(edge_records, answer, strict=True):
                kind, row, data, _left, _right = edge
                layers[int(color)].append({"type": "X" if kind == 0 else "Z", "check": row, "data": data})
            payload = {
                "schema_version": 1,
                "code": "[[112,16,7]]",
                "depth": args.depth,
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
                "colors": answer.astype(int).tolist(),
                "layers": layers,
                "search": {"trial": trial, "seed": args.seed, "moves_per_trial": args.moves_per_trial, "solve_seconds": seconds},
            }
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"event": "found", "output": str(args.output), "layer_sizes": payload["layer_sizes"]}), flush=True)
            return


if __name__ == "__main__":
    main()
