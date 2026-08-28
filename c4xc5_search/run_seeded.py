"""CLI for the seed-first ``C4 x C5`` BB search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c4xc5_search.seeded import search_seeded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=40502)
    args = parser.parse_args()

    output = Path("results/c4xc5-search") / args.run_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    result = search_seeded(trials=args.trials, random_seed=args.random_seed)
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
