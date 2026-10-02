#!/usr/bin/env python3
"""Certify saved floating-k single-row candidates by dimension stratum."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from gala_search.single_row_floating import certify_floating_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--k", type=int, action="append")
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--stop-after-hit", action="store_true")
    return parser.parse_args()


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    candidates = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if args.k:
        dimensions = set(args.k)
        candidates = [item for item in candidates if item["k"] in dimensions]
    candidates.sort(
        key=lambda item: (
            -item["k"],
            item["maximum_check_weight"],
            item["coefficient_weight"],
        )
    )
    if args.maximum is not None:
        candidates = candidates[: args.maximum]
    results = []
    for index, candidate in enumerate(candidates, start=1):
        result = certify_floating_candidate(candidate)
        results.append(result)
        write_jsonl(args.output, results)
        if index % 25 == 0 or result["accepted"] or index == len(candidates):
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(candidates),
                        "k": candidate["k"],
                        "distance": result["certified_distance"],
                        "accepted": result["accepted"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.stop_after_hit and result["accepted"]:
            break


if __name__ == "__main__":
    main()
