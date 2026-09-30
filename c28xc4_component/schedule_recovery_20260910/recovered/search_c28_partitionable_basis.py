#!/usr/bin/env python3
"""Jointly search degree-eight bases and exact clean early/late partitions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/judah_unmuth/gala-code-search/c28xc4_component")
from build_c28_partition_schedule import find_partition  # noqa: E402
from component_code import translated_check_catalog  # noqa: E402
from search_c28_basis import greedy_trial, integer_rows, score  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=160728)
    parser.add_argument("--rcl", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, metadata, orbit_sizes = translated_check_catalog()
    values = integer_rows(rows)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    degree8 = 0
    for trial in range(1, args.trials + 1):
        selected, degrees = greedy_trial(rows, values, rng, args.rcl)
        if int(degrees.max()) > 8:
            continue
        degree8 += 1
        check = rows[selected]
        status, partition, parity_constraints, solver_seconds = find_partition(check, 10)
        if degree8 % 100 == 0:
            print(json.dumps({
                "event": "checkpoint",
                "trial": trial,
                "degree8_bases": degree8,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
        if partition is None:
            continue
        result = {
            "status": status,
            "trial": trial,
            "degree8_bases_tested": degree8,
            "trials_requested": args.trials,
            "seed": args.seed,
            "rcl": args.rcl,
            "score": list(score(degrees)),
            "selected_indices": selected,
            "selected_checks": [
                {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
                for i in selected
            ],
            "degrees": degrees.astype(int).tolist(),
            "degree_histogram": {
                str(value): int(np.count_nonzero(degrees == value))
                for value in sorted(set(map(int, degrees)))
            },
            "early_data": np.flatnonzero(partition).astype(int).tolist(),
            "late_data": np.flatnonzero(~partition).astype(int).tolist(),
            "partition_parity_constraints": parity_constraints,
            "partition_solver_seconds": round(solver_seconds, 6),
            "orbit_sizes": list(orbit_sizes),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("selected_indices", "selected_checks", "degrees", "early_data", "late_data")}, indent=2, sort_keys=True))
        return
    result = {
        "status": "not_found",
        "trials": args.trials,
        "degree8_bases_tested": degree8,
        "seed": args.seed,
        "rcl": args.rcl,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
