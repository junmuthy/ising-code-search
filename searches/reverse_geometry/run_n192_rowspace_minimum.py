#!/usr/bin/env python3
"""Run exact minimum-compatible-F row-space ZX pilots at ``n=192``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from searches.reverse_geometry.n192 import NUM_HALVES, NUM_QUBITS, TOP_DEGREE, X_ORDER, Y_ORDER
from searches.reverse_geometry.rowspace import run_minimum_f_rowspace_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=100)
    parser.add_argument("--f-trials", type=int, default=3)
    parser.add_argument("--seconds-per-f-solve", type=float, default=2)
    parser.add_argument("--seconds-per-g-solve", type=float, default=2)
    parser.add_argument("--seed", type=int, default=260833)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    source = json.loads(args.source.read_text())
    output_dir.mkdir(parents=True)
    runs = []
    for index, witness in enumerate(source["witnesses"][: args.witnesses]):
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[witness["support"]] = 1
        runs.append(
            {
                "witness_index": index,
                "weight": witness["weight"],
                "support": witness["support"],
                "search": run_minimum_f_rowspace_pilot(
                    seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER),
                    f_trials=args.f_trials,
                    seconds_per_f_solve=args.seconds_per_f_solve,
                    seconds_per_g_solve=args.seconds_per_g_solve,
                    random_seed=args.seed + 1000 * index,
                ),
            }
        )
    counters: dict[str, int] = {}
    for run in runs:
        for key, value in run["search"]["counters"].items():
            counters[key] = counters.get(key, 0) + value
    result = {
        "source": str(args.source),
        "witnesses_tested": len(runs),
        "f_trials": args.f_trials,
        "seconds_per_f_solve": args.seconds_per_f_solve,
        "seconds_per_g_solve": args.seconds_per_g_solve,
        "counters": dict(sorted(counters.items())),
        "runs": runs,
    }
    output = output_dir / "minimum-rowspace-pilot.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "witnesses_tested": len(runs),
                "counters": result["counters"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
