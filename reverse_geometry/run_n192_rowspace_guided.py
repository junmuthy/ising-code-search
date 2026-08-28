#!/usr/bin/env python3
"""Run logical-compatibility-guided row-space ZX pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reverse_geometry.n192 import NUM_HALVES, NUM_QUBITS, TOP_DEGREE, X_ORDER, Y_ORDER
from reverse_geometry.rowspace import run_guided_rowspace_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=3)
    parser.add_argument(
        "--indices",
        help="comma-separated witness indices; overrides --witnesses",
    )
    parser.add_argument("--f-supports", type=int, default=100)
    parser.add_argument("--compatibility-restarts", type=int, default=5)
    parser.add_argument("--seconds-per-solve", type=float, default=1)
    parser.add_argument("--seed", type=int, default=260832)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    source = json.loads(args.source.read_text())
    output_dir.mkdir(parents=True)
    runs = []
    indices = (
        [int(value) for value in args.indices.split(",")]
        if args.indices
        else list(range(min(args.witnesses, len(source["witnesses"]))))
    )
    for index in indices:
        witness = source["witnesses"][index]
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[witness["support"]] = 1
        runs.append({
            "witness_index": index,
            "weight": witness["weight"],
            "support": witness["support"],
            "search": run_guided_rowspace_pilot(
                seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER),
                f_supports=args.f_supports,
                compatibility_restarts=args.compatibility_restarts,
                seconds_per_solve=args.seconds_per_solve,
                random_seed=args.seed + index,
            ),
        })
    result = {
        "source": str(args.source),
        "witnesses_tested": len(runs),
        "compatible_f_supports": sum(
            len(run["search"]["compatibility"]["supports"]) for run in runs
        ),
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
    output = output_dir / "guided-rowspace-pilot.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "witnesses_tested",
        "compatible_f_supports",
        "affine_solutions",
        "rowspace_zx_hits",
        "structural_hits",
        "weight_four_survivors",
    )} | {"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
