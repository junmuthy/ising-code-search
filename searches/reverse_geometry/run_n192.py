#!/usr/bin/env python3
"""Run the isolated natural-S3 reverse-geometry search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searches.reverse_geometry.n192 import geometry_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--witnesses", type=int, default=100)
    parser.add_argument("--annihilator-witnesses", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    result = geometry_run(
        random_seed=args.seed,
        witnesses=args.witnesses,
        annihilator_witnesses=args.annihilator_witnesses,
    )
    output = output_dir / "geometry-and-annihilator.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "output": str(output),
        "witnesses_found": result["witnesses_found"],
        "geometry_survives": result["geometry_survives"],
        "annihilator_stage_survives": result["annihilator_stage_survives"],
        "generator_stage_ready": result["generator_stage_ready"],
        "annihilator_nullities": [
            record["constraint_nullity"] for record in result["annihilator_records"]
        ],
        "rank_capacities_x": [
            record["rank_capacity_x_upper_bound"]
            for record in result["annihilator_records"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
