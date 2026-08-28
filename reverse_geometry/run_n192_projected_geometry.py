#!/usr/bin/env python3
"""Search general n=192 logical seeds with both top projections active."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverse_geometry.n192 import search_projected_geometry_witnesses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--trials", type=int, default=1_000_000)
    parser.add_argument("--target-witnesses", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    result = search_projected_geometry_witnesses(
        random_seed=args.seed,
        trials=args.trials,
        target_witnesses=args.target_witnesses,
    )
    output = output_dir / "projected-geometry.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "trials_completed": result["trials_completed"],
        "projection_survivors": result["projection_survivors"],
        "witnesses_found": result["witnesses_found"],
        "seconds": result["seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
