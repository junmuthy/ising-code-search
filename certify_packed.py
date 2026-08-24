#!/usr/bin/env python3
"""Certify distances of algebraically accepted packed BB candidates."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any

from gala_search.packed import (
    PackedCandidate,
    PackedSearchCandidate,
    PackedWeightTwelveCandidate,
    certify_packed_distance,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_atomic(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary_path = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _certify_safely(
    candidate: PackedSearchCandidate, solver: str, target_distance: int
) -> dict[str, Any]:
    try:
        return certify_packed_distance(
            candidate, solver=solver, target_distance=target_distance
        )
    except Exception as error:
        return {
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "complete": False,
            "threshold_passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "packed-weight8.jsonl",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "packed-certifications.jsonl",
    )
    parser.add_argument("--m", nargs="+", type=int, default=[7])
    parser.add_argument(
        "--family", choices=("packed_weight8", "packed_weight12")
    )
    parser.add_argument("--target-distance", type=int, default=6)
    parser.add_argument("--solver", default="HIGHS")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.target_distance < 1:
        raise SystemExit("--target-distance must be positive")

    search_records = _load_jsonl(args.input)
    existing = _load_jsonl(args.output)
    records_by_id = {record["candidate_id"]: record for record in existing}
    finished_ids = {
        record["candidate_id"]
        for record in existing
        if record.get("threshold_passed")
        or (
            record.get("distance_upper_bound") is not None
            and record["distance_upper_bound"] < args.target_distance
        )
    }
    selected = [
        record
        for record in search_records
        if record.get("accepted")
        and record["candidate"]["m"] in args.m
        and (
            args.family is None
            or record["candidate"].get("family") == args.family
        )
        and record["candidate_id"] not in finished_ids
    ]
    selected.sort(
        key=lambda record: (
            -record.get("total_seconds", 0),
            record["candidate_id"],
        )
    )
    if args.limit is not None:
        selected = selected[: args.limit]

    print(
        f"Certifying {len(selected)} packed candidates at target d >= "
        f"{args.target_distance}; workers={args.workers}",
        flush=True,
    )
    if selected:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for record in selected:
                data = record["candidate"]
                if data.get("family") == "packed_weight12":
                    candidate = PackedWeightTwelveCandidate(
                        data["m"],
                        data["r0"],
                        data["r1"],
                        data["r2"],
                        data["p1"],
                        data["p2"],
                    )
                else:
                    candidate = PackedCandidate(
                        data["m"], data["r"], data["s"], data["u"], data["v"]
                    )
                futures[
                    executor.submit(
                        _certify_safely,
                        candidate,
                        args.solver,
                        args.target_distance,
                    )
                ] = candidate

            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                candidate = futures[future]
                result = future.result()
                records_by_id[candidate.candidate_id] = result
                if completed % 10 == 0 or completed == len(selected):
                    passing = sum(
                        bool(record.get("threshold_passed"))
                        for record in records_by_id.values()
                    )
                    print(
                        f"Completed {completed}/{len(selected)}; "
                        f"d >= {args.target_distance}: {passing}",
                        flush=True,
                    )
                _write_jsonl_atomic(args.output, list(records_by_id.values()))

    statuses = Counter(
        "pass"
        if record.get("threshold_passed")
        else "error"
        if record.get("error")
        else f"d<={record.get('distance_upper_bound', '?')}"
        for record in records_by_id.values()
    )
    print(f"Wrote {len(records_by_id)} records to {args.output}", flush=True)
    print(f"Certification counts: {dict(statuses)}", flush=True)


if __name__ == "__main__":
    main()
