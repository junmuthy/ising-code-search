#!/usr/bin/env python3
"""Run six-term meet-in-the-middle synthesis on saved logical geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from searches.reverse_geometry.n192 import (
    NUM_HALVES,
    NUM_QUBITS,
    TOP_DEGREE,
    X_ORDER,
    Y_ORDER,
    search_mitm_six_term_generators,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--maximum-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--weights", type=int, nargs="+", default=[4, 5, 6])
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
            "search": search_mitm_six_term_generators(
                seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER),
                restarts=args.restarts,
                maximum_records=args.maximum_records,
                random_seed=args.seed + index,
                term_weights=tuple(args.weights),
            ),
        })
    result = {
        "source": str(args.source),
        "witnesses_tested": len(runs),
        "records": sum(len(run["search"]["records"]) for run in runs),
        "structural_pilot_hits": sum(
            run["search"]["num_structural_pilot_hits"] for run in runs
        ),
        "runs": runs,
    }
    output = output_dir / "six-term-mitm.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "witnesses_tested": result["witnesses_tested"],
        "records": result["records"],
        "structural_pilot_hits": result["structural_pilot_hits"],
        "counters": [run["search"]["counters"] for run in runs],
    }, indent=2))


if __name__ == "__main__":
    main()
