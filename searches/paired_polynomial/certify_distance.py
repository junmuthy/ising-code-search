#!/usr/bin/env python3
"""Certify the exact distance of one paired-polynomial GALA candidate."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from gala_search.core import WeightEightCandidate, certify_distance

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--solver", default="HIGHS")
    parser.add_argument(
        "--stop-at",
        type=int,
        help="stop after finding a rigorously verified logical at or below this weight",
    )
    parser.add_argument(
        "--logical-index",
        action="append",
        type=int,
        dest="logical_indices",
        help="certify only selected logical index; repeat as needed (default: all eight)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_DIR / "results" / "certifications.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = WeightEightCandidate(args.m, args.r, args.s, args.t)
    result = certify_distance(
        candidate,
        solver=args.solver,
        logical_indices=args.logical_indices,
        stop_at=args.stop_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as output:
        temporary_path = pathlib.Path(output.name)
        if args.output.exists():
            output.write(args.output.read_text())
        output.write(json.dumps(result, sort_keys=True) + "\n")
    temporary_path.replace(args.output)


if __name__ == "__main__":
    main()
