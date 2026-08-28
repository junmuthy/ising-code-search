#!/usr/bin/env python3
"""Run the bounded general two-polynomial ``C4 x C5`` search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c4xc5_search.general import search_general


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/c4xc5-search") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    result = search_general(
        trials=args.trials,
        seed_restarts=args.seed_restarts,
        random_seed=args.seed,
    )
    records_path = output_dir / "paired-logical-candidates.jsonl"
    with records_path.open("w") as stream:
        for record in result.pop("records"):
            stream.write(json.dumps(record) + "\n")
    summary = {
        **result,
        "records_path": str(records_path),
        "output": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
