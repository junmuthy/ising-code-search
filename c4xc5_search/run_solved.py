"""CLI for the seed-constrained ``C4 x C5`` search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c4xc5_search.solved import search_solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--solver-restarts", type=int, default=3)
    parser.add_argument("--pairs-per-seed", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=40503)
    args = parser.parse_args()

    output = Path("results/c4xc5-search") / args.run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    result = search_solved(
        seeds=args.seeds,
        solver_restarts=args.solver_restarts,
        pairs_per_seed=args.pairs_per_seed,
        random_seed=args.random_seed,
    )
    summary = {**result, "output": str(output)}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output / "survivors.jsonl").open("w") as handle:
        for record in result["survivors"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output / "near-misses.jsonl").open("w") as handle:
        for record in result["near_misses"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    printable = {
        key: value
        for key, value in summary.items()
        if key not in ("survivors", "near_misses")
    }
    printable["survivors"] = len(result["survivors"])
    printable["near_misses"] = len(result["near_misses"])
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
