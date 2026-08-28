#!/usr/bin/env python3
"""Run the paper-faithful abelian ``[[32,4,>=6]]`` searches."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import Any

from gala_search.abelian_bb32 import search_c4xc4
from gala_search.abelian_single_row import search_l4_group

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--family",
        action="append",
        choices=("c8", "c2xc4", "c4xc4"),
        help="repeat to select families; defaults to all three",
    )
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--no-distance", action="store_true")
    parser.add_argument("--maximum-saved-near-misses", type=int, default=20)
    return parser.parse_args()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = (
        PROJECT_DIR / "results" / "single-row" / "abelian-n32-k4" / args.run_name
    )
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    families = args.family or ["c8", "c2xc4", "c4xc4"]
    records = []
    for family in families:
        if family == "c4xc4":
            record = search_c4xc4(
                maximum_check_weight=args.maximum_check_weight,
                certify_distance=not args.no_distance,
                maximum_saved_near_misses=args.maximum_saved_near_misses,
            )
        else:
            record = search_l4_group(
                family,
                maximum_check_weight=args.maximum_check_weight,
                certify_distance=not args.no_distance,
                maximum_saved_near_misses=args.maximum_saved_near_misses,
            )
        records.append(record)
        write_json(output_dir / f"{family}.json", record)
        print(json.dumps({key: value for key, value in record.items() if key not in {"accepted", "near_misses"}}, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "target": "[[32,4,>=6]]",
        "arguments": vars(args),
        "families": families,
        "accepted_candidates": sum(len(record["accepted"]) for record in records),
        "output": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
