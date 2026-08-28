#!/usr/bin/env python3
"""Run the exhaustive self-dual ``C4 x C5`` BB search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c4xc5_search.search import search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--maximum-certifications", type=int, default=0)
    parser.add_argument("--maximum-saved-near-misses", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/c4xc5-search") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    result = search(
        maximum_certifications=args.maximum_certifications,
        maximum_saved_near_misses=args.maximum_saved_near_misses,
    )
    structural_path = output_dir / "structural-candidates.jsonl"
    with structural_path.open("w") as stream:
        for record in result.pop("structural_candidates"):
            stream.write(json.dumps(record) + "\n")
    summary = {
        **result,
        "structural_candidates_path": str(structural_path),
        "output": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
