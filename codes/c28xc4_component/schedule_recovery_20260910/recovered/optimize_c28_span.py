#!/usr/bin/env python3
"""Kempe-search a 16-edge-coloring with compact color spans at data qubits."""

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
from optimize_c28_joint_schedule import initial_coloring, selected_check


def x_edges(edge_records):
    return [i for i, edge in enumerate(edge_records) if edge[0] == 0]


def objective(edge_records, colors):
    at_data = [[] for _ in range(112)]
    for i in x_edges(edge_records):
        at_data[edge_records[i][2]].append(int(colors[i]))
    spans = np.asarray([max(values) - min(values) for values in at_data], dtype=int)
    excess = np.maximum(spans - 7, 0)
    return (int(excess.sum()), int(spans.max()), int((excess * excess).sum())), spans


def component(edge_records, colors, start, first, second):
    incidence = {}
    for i in x_edges(edge_records):
        color = int(colors[i])
        incidence[edge_records[i][3], color] = i
        incidence[edge_records[i][4], color] = i
    todo = [start]
    found = set()
    while todo:
        i = todo.pop()
        if i in found:
            continue
        found.add(i)
        for node in edge_records[i][3:5]:
            for color in (first, second):
                neighbor = incidence.get((node, color))
                if neighbor is not None and neighbor not in found:
                    todo.append(neighbor)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=241127)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/c28_span_coloring.json"))
    args = parser.parse_args()
    check, selection, _metadata = selected_check(Path("/tmp/c28_basis_search.json"))
    records, colors = initial_coloring(check)
    if args.start:
        colors = np.asarray(json.loads(args.start.read_text())["colors"], dtype=np.int16)
    edges = x_edges(records)
    rng = random.Random(args.seed)
    current, spans = objective(records, colors)
    best = current
    best_colors = colors.copy()
    started = time.perf_counter()
    print(json.dumps({"event": "start", "score": current}), flush=True)
    for move in range(1, args.moves + 1):
        start = rng.choice(edges)
        first = int(colors[start])
        second = rng.randrange(15)
        if second >= first:
            second += 1
        changed = component(records, colors, start, first, second)
        for i in changed:
            colors[i] = second if colors[i] == first else first
        new, new_spans = objective(records, colors)
        delta = (new[0] - current[0]) + 0.05 * (new[2] - current[2])
        temperature = max(0.001, args.temperature * (1 - move / args.moves))
        if new <= current or rng.random() < math.exp(-max(0, delta) / temperature):
            current, spans = new, new_spans
        else:
            for i in changed:
                colors[i] = second if colors[i] == first else first
        if current < best:
            best = current
            best_colors = colors.copy()
            print(json.dumps({"event": "best", "move": move, "score": best, "elapsed_seconds": round(time.perf_counter()-started, 3)}), flush=True)
            if best[0] == 0:
                break
    score, spans = objective(records, best_colors)
    payload = {
        "score": score,
        "colors": best_colors.astype(int).tolist(),
        "spans": spans.astype(int).tolist(),
        "basis_selected_indices": selection,
        "moves": move,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "colors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
