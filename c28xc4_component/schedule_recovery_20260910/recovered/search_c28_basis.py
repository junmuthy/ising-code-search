#!/usr/bin/env python3
"""Search the translated weight-16 catalog for a lower-degree check basis."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
sys.path.insert(0, str(ROOT))
from component_code import CHECK_RANK, translated_check_catalog  # noqa: E402


def integer_rows(rows):
    return [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in rows]


def reduce_value(value, pivots):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            return value
        value ^= pivots[pivot]
    return 0


def add_pivot(value, pivots):
    value = reduce_value(value, pivots)
    if not value:
        return False
    pivots[value.bit_length() - 1] = value
    return True


def score(degrees):
    over8 = np.maximum(degrees - 8, 0)
    return (
        int(over8.sum()),
        int(degrees.max()),
        int(np.sum(degrees.astype(np.int64) ** 2)),
        int(degrees.max() - degrees.min()),
    )


def greedy_trial(rows, values, rng, rcl):
    pivots = {}
    degrees = np.zeros(rows.shape[1], dtype=np.int16)
    selected = []
    remaining = list(range(len(rows)))
    for _step in range(CHECK_RANK):
        candidates = []
        for index in remaining:
            reduced = reduce_value(values[index], pivots)
            if reduced:
                new_degrees = degrees + rows[index]
                candidates.append((score(new_degrees), rng.random(), index, reduced))
        if not candidates:
            raise RuntimeError("independent set did not extend")
        candidates.sort()
        limit = min(rcl, len(candidates))
        _candidate_score, _tie, chosen, reduced = candidates[rng.randrange(limit)]
        add_pivot(reduced, pivots)
        degrees += rows[chosen]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected, degrees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1121607)
    parser.add_argument("--rcl", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, metadata, orbit_sizes = translated_check_catalog()
    values = integer_rows(rows)
    rng = random.Random(args.seed)
    best = None
    started = time.perf_counter()
    for trial in range(1, args.trials + 1):
        selected, degrees = greedy_trial(rows, values, rng, args.rcl)
        current = score(degrees)
        if best is None or current < best[0]:
            best = (current, selected, degrees.copy(), trial)
            print(json.dumps({
                "event": "best",
                "trial": trial,
                "score": current,
                "degree_histogram": {str(d): int(np.count_nonzero(degrees == d)) for d in sorted(set(map(int, degrees)))},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
        if current[0] == 0:
            break
    assert best is not None
    result = {
        "score": list(best[0]),
        "trial": best[3],
        "trials_requested": args.trials,
        "seed": args.seed,
        "rcl": args.rcl,
        "selected_indices": best[1],
        "selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in best[1]
        ],
        "degrees": best[2].astype(int).tolist(),
        "degree_histogram": {str(d): int(np.count_nonzero(best[2] == d)) for d in sorted(set(map(int, best[2])))},
        "orbit_sizes": list(orbit_sizes),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
