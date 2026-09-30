#!/usr/bin/env python3
"""Target failed cross-ancilla parities with Kempe-chain schedule repairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from optimize_c28_joint_schedule import (  # noqa: E402
    incidence,
    initial_coloring,
    kempe_component,
    objective,
    selected_check,
    verify_coloring,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path("/tmp/c28_basis_search.json"))
    parser.add_argument("--start", type=Path, default=Path("/tmp/c28_joint_schedule.json"))
    parser.add_argument("--moves", type=int, default=500000)
    parser.add_argument("--proposals", type=int, default=12)
    parser.add_argument("--seed", type=int, default=716281)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check, selection, metadata = selected_check(args.basis)
    edge_records, _initial = initial_coloring(check)
    saved = json.loads(args.start.read_text())
    colors = np.asarray(saved["colors"], dtype=np.int16)
    if len(colors) != len(edge_records):
        raise SystemExit("start colors do not match edge list")
    current_score, _tx, _tz, failures = verify_coloring(check, edge_records, colors)
    best_score = current_score
    best_colors = colors.copy()
    rng = random.Random(args.seed)
    edge_lookup = {(kind, row, data): i for i, (kind, row, data, _l, _r) in enumerate(edge_records)}
    started = time.perf_counter()
    print(json.dumps({"event": "start", "parity_failures": current_score}), flush=True)
    for move in range(1, args.moves + 1):
        failed = np.argwhere(failures)
        if not len(failed):
            break
        xrow, zrow = map(int, failed[rng.randrange(len(failed))])
        overlap = np.flatnonzero(check[xrow] & check[zrow]).astype(int)
        by_node_color = incidence(edge_records, colors)
        proposal_records = []
        for _ in range(args.proposals):
            data = int(overlap[rng.randrange(len(overlap))])
            kind, row = (0, xrow) if rng.randrange(2) == 0 else (1, zrow)
            start_edge = edge_lookup[(kind, row, data)]
            first = int(colors[start_edge])
            second = rng.randrange(15)
            if second >= first:
                second += 1
            component = kempe_component(edge_records, colors, start_edge, first, second, by_node_color)
            for edge_index in component:
                colors[edge_index] = second if colors[edge_index] == first else first
            new_score, _a, _b, new_failures = objective(check, edge_records, colors)
            for edge_index in component:
                colors[edge_index] = second if colors[edge_index] == first else first
            proposal_records.append((new_score, rng.random(), component, first, second, new_failures))
        proposal_records.sort(key=lambda item: (item[0], item[1]))
        new_score, _tie, component, first, second, new_failures = proposal_records[0]
        temperature = max(0.01, 0.5 * (1.0 - move / args.moves))
        if new_score <= current_score or rng.random() < math.exp((current_score - new_score) / temperature):
            for edge_index in component:
                colors[edge_index] = second if colors[edge_index] == first else first
            current_score = new_score
            failures = new_failures
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            print(json.dumps({
                "event": "best",
                "move": move,
                "parity_failures": best_score,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
            if best_score == 0:
                break
    score, _tx, _tz, failures = verify_coloring(check, edge_records, best_colors)
    layers = [[] for _ in range(16)]
    for edge, color in zip(edge_records, best_colors, strict=True):
        kind, row, data, _left, _right = edge
        layers[int(color)].append({"type": "X" if kind == 0 else "Z", "check": row, "data": data})
    result = {
        "code": "[[112,16,7]]",
        "depth": 16,
        "cnot_count": len(edge_records),
        "layer_sizes": [len(layer) for layer in layers],
        "parity_failures": score,
        "clean_cross_ancilla_backaction": score == 0,
        "basis_selected_indices": selection,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selection
        ],
        "colors": best_colors.astype(int).tolist(),
        "failed_cross_pairs": np.argwhere(failures).astype(int).tolist(),
        "layers": layers,
        "moves": move,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("colors", "layers", "basis_selected_checks", "basis_selected_indices", "failed_cross_pairs")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
