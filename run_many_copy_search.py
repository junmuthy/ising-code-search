#!/usr/bin/env python3
"""Search connected weight-12 codes containing many 32-qubit Ising grids."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any, Iterable

import diskcache

from gala_search.many_copy import (
    MANY_COPY_SCHEMA_VERSION,
    ManyCopyCandidate,
    analyze_many_copy_candidate,
    iter_many_copy_candidates,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "many-copy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--copies-per-half", type=int, default=2)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-symmetry-quotient", action="store_true")
    parser.add_argument("--skip-graph-metrics", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--cache-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR / "cache"
    )
    return parser.parse_args()


def _cache_key(candidate: ManyCopyCandidate, include_graph_metrics: bool) -> str:
    return (
        f"many-copy-v{MANY_COPY_SCHEMA_VERSION}:{candidate.candidate_id}:"
        f"graph={int(include_graph_metrics)}"
    )


def _analyze_safely(
    candidate: ManyCopyCandidate, include_graph_metrics: bool
) -> dict[str, Any]:
    try:
        return analyze_many_copy_candidate(
            candidate, include_graph_metrics=include_graph_metrics
        )
    except Exception as error:
        return {
            "schema_version": MANY_COPY_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reasons": ["worker_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _write_jsonl_atomic(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.m < 3 or args.copies_per_half < 1:
        raise SystemExit("invalid search size or worker count")
    candidates = sorted(
        iter_many_copy_candidates(
            m=args.m,
            copies_per_half=args.copies_per_half,
            quotient_symmetries=not args.no_symmetry_quotient,
        )
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        candidates = candidates[: args.limit]
    output_path = args.output or (
        DEFAULT_RESULTS_DIR
        / f"weight12-r{args.copies_per_half}-m{args.m}.jsonl"
    )
    include_graph_metrics = not args.skip_graph_metrics
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "copies_per_half": args.copies_per_half,
                "m": args.m,
                "workers": args.workers,
                "graph_metrics": include_graph_metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(args.cache_dir))
    records_by_id: dict[str, dict[str, Any]] = {}
    pending: list[ManyCopyCandidate] = []
    for candidate in candidates:
        key = _cache_key(candidate, include_graph_metrics)
        if key in cache:
            records_by_id[candidate.candidate_id] = cache[key]
        else:
            pending.append(candidate)
    print(
        json.dumps(
            {"cache_hits": len(candidates) - len(pending), "pending": len(pending)},
            sort_keys=True,
        ),
        flush=True,
    )

    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _analyze_safely, candidate, include_graph_metrics
                ): candidate
                for candidate in pending
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                candidate = futures[future]
                record = future.result()
                records_by_id[candidate.candidate_id] = record
                cache[_cache_key(candidate, include_graph_metrics)] = record
                if completed % 25 == 0 or completed == len(pending):
                    accepted = sum(
                        bool(record.get("accepted"))
                        for record in records_by_id.values()
                    )
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "pending": len(pending),
                                "accepted": accepted,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    records = [records_by_id[candidate.candidate_id] for candidate in candidates]
    _write_jsonl_atomic(output_path, records)
    cache.close()
    statuses = Counter(
        "accepted"
        if record.get("accepted")
        else record.get("rejection_reasons", ["unknown"])[0]
        for record in records
    )
    accepted_records = [record for record in records if record.get("accepted")]
    best_four_cycles = min(
        (
            record["num_four_cycles_per_css_tanner_graph"]
            for record in accepted_records
            if record.get("num_four_cycles_per_css_tanner_graph") is not None
        ),
        default=None,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": len(records),
                "statuses": statuses,
                "best_four_cycles": best_four_cycles,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
