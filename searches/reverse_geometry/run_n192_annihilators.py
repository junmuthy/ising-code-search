#!/usr/bin/env python3
"""Compute fold-tied annihilator capacities for saved n=192 geometries."""

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
    fold_tied_annihilator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    source = json.loads(args.source.read_text())
    output_dir.mkdir(parents=True)
    records = []
    for index, witness in enumerate(source["witnesses"][: args.witnesses]):
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[witness["support"]] = 1
        records.append({
            "witness_index": index,
            "weight": witness["weight"],
            "top_projection_weights": witness.get("top_projection_weights"),
            "support": witness["support"],
            "annihilator": fold_tied_annihilator(
                seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER)
            ),
        })
    result = {
        "source": str(args.source),
        "witnesses_tested": len(records),
        "rank_capable": sum(
            record["annihilator"]["rank_capacity_survives_k32_target"]
            for record in records
        ),
        "records": records,
    }
    output = output_dir / "annihilator-capacities.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "witnesses_tested": result["witnesses_tested"],
        "rank_capable": result["rank_capable"],
        "nullities": [record["annihilator"]["constraint_nullity"] for record in records],
        "rank_capacities": [record["annihilator"]["rank_capacity_x_upper_bound"] for record in records],
    }, indent=2))


if __name__ == "__main__":
    main()
