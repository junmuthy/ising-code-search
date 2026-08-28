#!/usr/bin/env python3
"""Enumerate all natural-S3 check-coordinate folds for one data fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reverse_geometry.n192 import (
    NUM_HALVES,
    NUM_QUBITS,
    TOP_DEGREE,
    X_ORDER,
    Y_ORDER,
    search_mitm_six_term_generators,
    top_permutations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--maximum-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    source = json.loads(args.source.read_text())
    output_dir.mkdir(parents=True)
    runs = []
    for witness_index, witness in enumerate(source["witnesses"][: args.witnesses]):
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[witness["support"]] = 1
        shaped = seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER)
        for fold_index, check_fold in enumerate(top_permutations()):
            runs.append({
                "witness_index": witness_index,
                "check_fold_index": fold_index,
                "check_top_images": list(check_fold),
                "search": search_mitm_six_term_generators(
                    shaped,
                    restarts=args.restarts,
                    maximum_records=args.maximum_records,
                    random_seed=args.seed + 101 * witness_index + fold_index,
                    term_weights=(4, 5, 6),
                    check_top_images=check_fold,
                ),
            })
    result = {
        "source": str(args.source),
        "witnesses_tested": min(args.witnesses, len(source["witnesses"])),
        "check_folds_per_witness": len(top_permutations()),
        "records": sum(len(run["search"]["records"]) for run in runs),
        "structural_pilot_hits": sum(
            run["search"]["num_structural_pilot_hits"] for run in runs
        ),
        "runs": runs,
    }
    output = output_dir / "check-fold-mitm.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "witnesses_tested": result["witnesses_tested"],
        "check_folds_per_witness": result["check_folds_per_witness"],
        "records": result["records"],
        "structural_pilot_hits": result["structural_pilot_hits"],
        "nonempty_spaces": [
            [run["witness_index"], run["check_fold_index"], len(run["search"]["records"])]
            for run in runs if run["search"]["records"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
