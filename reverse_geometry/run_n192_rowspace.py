#!/usr/bin/env python3
"""Run sparse-F, affine-solved-G row-space ZX pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reverse_geometry.n192 import NUM_HALVES, NUM_QUBITS, TOP_DEGREE, X_ORDER, Y_ORDER
from reverse_geometry.rowspace import run_rowspace_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=3)
    parser.add_argument("--f-trials", type=int, default=100)
    parser.add_argument("--seconds-per-solve", type=float, default=1)
    parser.add_argument("--seed", type=int, default=260831)
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
        runs.append({
            "witness_index": index,
            "weight": witness["weight"],
            "support": witness["support"],
            "search": run_rowspace_pilot(
                seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER),
                f_trials=args.f_trials,
                seconds_per_solve=args.seconds_per_solve,
                random_seed=args.seed + index,
            ),
        })
    result = {
        "source": str(args.source),
        "witnesses_tested": len(runs),
        "f_trials": sum(run["search"]["f_trials"] for run in runs),
        "affine_solutions": sum(len(run["search"]["records"]) for run in runs),
        "rowspace_zx_hits": sum(
            run["search"]["counters"].get("rowspace_zx_hits", 0) for run in runs
        ),
        "structural_hits": sum(len(run["search"]["structural_hits"]) for run in runs),
        "weight_four_survivors": sum(
            run["search"]["counters"].get("weight_four_survivors", 0)
            for run in runs
        ),
        "runs": runs,
    }
    output = output_dir / "rowspace-pilot.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "witnesses_tested",
        "f_trials",
        "affine_solutions",
        "rowspace_zx_hits",
        "structural_hits",
        "weight_four_survivors",
    )} | {"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
