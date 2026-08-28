#!/usr/bin/env python3
"""Run the exact and sampled ``n=128`` batched half-grid geometry screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gala_search.batched_half_grid import (
    exact_geometry_certificate,
    sampled_geometry_screen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--samples-per-fold", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/half-grid-batched") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    result = {
        "exact_certificate": exact_geometry_certificate(),
        "sampled_screen": sampled_geometry_screen(
            seed=args.seed, samples_per_fold=args.samples_per_fold
        ),
    }
    output = output_dir / "geometry-certificate.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "exact_no_go": result["exact_certificate"]["pairing_singular_for_every_allowed_seed"],
        "generator_search_pruned": result["exact_certificate"]["generator_search_pruned"],
        "sampled_supports": result["sampled_screen"]["tested"],
        "maximum_sampled_rank": result["sampled_screen"]["maximum_sampled_rank"],
        "full_rank_hits": result["sampled_screen"]["full_rank_hits"],
    }, indent=2))


if __name__ == "__main__":
    main()
