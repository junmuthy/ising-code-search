#!/usr/bin/env python3
"""Batch-certify promising candidates from a GALA search result file."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import tempfile
import traceback
from typing import Any

from gala_search.core import WeightEightCandidate, certify_distance

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
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
    candidate: WeightEightCandidate, solver: str, stop_at: int | None
) -> dict[str, Any]:
    try:
        return certify_distance(candidate, solver=solver, stop_at=stop_at)
    except Exception as error:
        return {
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "canonical_support": [list(term) for term in candidate.equivalence_key],
            "complete": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=pathlib.Path, default=DEFAULT_RESULTS_DIR / "weight8.jsonl"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "certifications.jsonl",
    )
    parser.add_argument("--m", nargs="+", type=int, default=[7])
    parser.add_argument("--min-bound", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--solver", default="HIGHS")
    parser.add_argument(
        "--stop-at",
        type=int,
        help="stop each candidate after proving an upper bound at or below this weight",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    search_records = _load_jsonl(args.input)
    existing = _load_jsonl(args.output)
    existing_ids = {
        record["candidate_id"]
        for record in existing
        if record.get("complete")
        or (
            args.stop_at is not None
            and record.get("ilp_distance_upper_bound") is not None
            and record["ilp_distance_upper_bound"] <= args.stop_at
        )
    }
    selected = [
        record
        for record in search_records
        if record.get("accepted")
        and record["candidate"]["m"] in args.m
        and (record.get("distance_upper_bound") or -1) >= args.min_bound
        and record["candidate_id"] not in existing_ids
    ]
    selected.sort(
        key=lambda record: (
            record.get("distance_upper_bound") or -1,
            record.get("tanner_girth") or -1,
            -record.get("num_four_cycles", 10**12),
            record["candidate_id"],
        ),
        reverse=True,
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    print(
        f"Certifying {len(selected)} candidates with m={args.m}, "
        f"bound >= {args.min_bound}, workers={args.workers}",
        flush=True,
    )

    records_by_id = {record["candidate_id"]: record for record in existing}
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for record in selected:
            data = record["candidate"]
            candidate = WeightEightCandidate(data["m"], data["r"], data["s"], data["t"])
            futures[
                executor.submit(_certify_safely, candidate, args.solver, args.stop_at)
            ] = candidate
        for future in concurrent.futures.as_completed(futures):
            candidate = futures[future]
            result = future.result()
            records_by_id[candidate.candidate_id] = result
            completed += 1
            distance = (
                result.get("certified_distance")
                or result.get("ilp_distance_upper_bound")
                or result.get("error", "failed")
            )
            print(
                f"Completed {completed}/{len(selected)}: {candidate.candidate_id} -> {distance}",
                flush=True,
            )
            _write_jsonl_atomic(args.output, list(records_by_id.values()))

    print(f"Wrote {len(records_by_id)} certification records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
