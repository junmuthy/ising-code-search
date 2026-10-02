#!/usr/bin/env python3
"""Certify only untested candidates from a saved floating-k GL(2,2) run."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from typing import Any

from gala_search.single_row_floating import certify_floating_candidate


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    fold = candidate["data_fold"]
    return (
        tuple(candidate["coefficients"]),
        tuple(fold["fibre_images"]),
        tuple(fold["fibre_shifts"]),
    )


def atomic_json(path: pathlib.Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=pathlib.Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--existing",
        action="append",
        default=[],
        help="Existing certification JSONL filename relative to the run directory",
    )
    args = parser.parse_args()
    output = args.run_directory / args.output_name
    progress = output.with_suffix(".progress.json")
    summary = output.with_suffix(".summary.json")
    for path in (output, progress, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    candidates = read_jsonl(args.run_directory / "structural-candidates.jsonl")
    tested: set[tuple[Any, ...]] = set()
    existing_records = 0
    for filename in args.existing:
        for record in read_jsonl(args.run_directory / filename):
            tested.add(candidate_key(record["candidate"]))
            existing_records += 1
    remaining = [
        candidate for candidate in candidates if candidate_key(candidate) not in tested
    ]
    started = time.perf_counter()
    accepted = 0
    distances: dict[str, int] = {}
    with output.open("x") as handle:
        for completed, candidate in enumerate(remaining, start=1):
            result = certify_floating_candidate(candidate)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            accepted += int(result["accepted"])
            distance = str(result["certified_distance"])
            distances[distance] = distances.get(distance, 0) + 1
            checkpoint = {
                "status": "running",
                "completed": completed,
                "total": len(remaining),
                "accepted": accepted,
                "distance_counts": distances,
                "seconds": round(time.perf_counter() - started, 6),
            }
            atomic_json(progress, checkpoint)
            print(json.dumps(checkpoint, sort_keys=True), flush=True)
    result_summary = {
        "status": "complete",
        "structural_candidates": len(candidates),
        "existing_records": existing_records,
        "existing_unique_candidates": len(tested),
        "new_certifications": len(remaining),
        "accepted": accepted,
        "distance_counts": distances,
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(summary, result_summary)
    atomic_json(progress, result_summary)
    print(json.dumps(result_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
