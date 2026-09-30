#!/usr/bin/env python3
"""Screen alternative weight-16 bases and edge orders for hook fault distance."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from batch_c28_fixed_x import x_kempe_move
from optimize_c28_joint_schedule import initial_coloring
from probe_c28_fault_distance import build_from_layers
from search_c28_basis import greedy_trial, integer_rows

ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
sys.path.insert(0, str(ROOT))
from component_code import translated_check_catalog


def schedule_layers(records, colors, reversed_x):
    one = {(row, data): int(colors[i]) for i, (kind, row, data, _l, _r) in enumerate(records) if kind == 0}
    layers = [[] for _ in range(32)]
    for (row, data), color in one.items():
        layers[color].append({"type": "Z", "check": row, "data": data})
        xtime = 31 - color if reversed_x else 16 + color
        layers[xtime].append({"type": "X", "check": row, "data": data})
    return layers


def heuristic(layers):
    circuit = build_from_layers("X", layers, noisy=True)
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=8,
        dont_explore_edges_with_degree_above=12,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    return len(errors), [str(error) for error in errors]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=716112)
    parser.add_argument("--output", type=Path, default=Path("/tmp/c28_fault_schedule_best.json"))
    args = parser.parse_args()
    rows, metadata, _orbits = translated_check_catalog()
    values = integer_rows(rows)
    rng = random.Random(args.seed)
    best = None
    started = time.perf_counter()
    attempted = 0
    for trial in range(args.trials):
        selection, degrees = greedy_trial(rows, values, rng, 4)
        if int(degrees.max()) > 8:
            continue
        check = rows[selection]
        records, colors = initial_coloring(check)
        for _ in range(rng.randrange(500)):
            x_kempe_move(records, colors, rng)
        for reversed_x in (False, True):
            attempted += 1
            layers = schedule_layers(records, colors, reversed_x)
            upper, errors = heuristic(layers)
            record = {
                "upper": upper,
                "trial": trial,
                "reversed_x": reversed_x,
                "selection": selection,
                "selected_checks": [
                    {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
                    for i in selection
                ],
                "degree_histogram": {str(d): int(np.count_nonzero(degrees == d)) for d in sorted(set(map(int, degrees)))},
                "colors": colors.astype(int).tolist(),
                "layers": layers,
                "heuristic_errors": errors,
            }
            if best is None or upper > best["upper"]:
                best = record
                args.output.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"event": "best", "upper": upper, "trial": trial, "reversed_x": reversed_x, "degree_histogram": record["degree_histogram"], "attempted": attempted, "elapsed_seconds": round(time.perf_counter()-started, 3)}), flush=True)
            elif attempted % 10 == 0:
                print(json.dumps({"event": "progress", "trial": trial, "upper": upper, "best": best["upper"], "attempted": attempted, "elapsed_seconds": round(time.perf_counter()-started, 3)}), flush=True)
    print(json.dumps({"event": "finish", "best_upper": best["upper"] if best else None, "attempted": attempted, "elapsed_seconds": round(time.perf_counter()-started, 3)}), flush=True)


if __name__ == "__main__":
    main()
