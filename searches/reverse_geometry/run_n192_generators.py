#!/usr/bin/env python3
"""Run the first sparse fold-tied generator pilot on saved geometry."""

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
    search_sparse_fold_tied_generators,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=3)
    parser.add_argument("--solutions", type=int, default=100)
    parser.add_argument("--seconds-per-solution", type=float, default=5)
    parser.add_argument("--minimum-terms", type=int, default=1)
    parser.add_argument("--maximum-terms", type=int, default=6)
    parser.add_argument("--require-no-zero-columns", action="store_true")
    parser.add_argument("--require-nonzero-top-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=260827)
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
                "support": witness["support"],
                "search": search_sparse_fold_tied_generators(
                    seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER),
                    minimum_terms=args.minimum_terms,
                    maximum_terms=args.maximum_terms,
                    solutions=args.solutions,
                    seconds_per_solution=args.seconds_per_solution,
                    random_seed=args.seed + index,
                    require_no_zero_columns=args.require_no_zero_columns,
                    require_nonzero_top_augmentation=(
                        args.require_nonzero_top_augmentation
                    ),
                ),
            }
        )
    result = {
        "source": str(args.source),
        "witnesses_tested": len(runs),
        "solutions_found": sum(run["search"]["solutions_found"] for run in runs),
        "css_hits": sum(run["search"]["num_css_hits"] for run in runs),
        "structural_pilot_hits": sum(
            run["search"]["num_structural_pilot_hits"] for run in runs
        ),
        "runs": runs,
    }
    output = output_dir / "fold-tied-generator-pilot.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "witnesses_tested": result["witnesses_tested"],
        "solutions_found": result["solutions_found"],
        "css_hits": result["css_hits"],
        "structural_pilot_hits": result["structural_pilot_hits"],
    }, indent=2))


if __name__ == "__main__":
    main()
