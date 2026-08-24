#!/usr/bin/env python3
"""Search packed self-dual BB codes with a ``C_8 x C_4`` logical quotient."""

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

from gala_search.packed import (
    PACKED_SCHEMA_VERSION,
    PackedSearchCandidate,
    analyze_packed_candidate,
    iter_packed_candidates,
    iter_packed_weight_twelve_candidates,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def _cache_key(candidate: PackedSearchCandidate, include_graph_metrics: bool) -> str:
    return (
        f"packed-v{PACKED_SCHEMA_VERSION}:{candidate.candidate_id}:"
        f"graph={int(include_graph_metrics)}"
    )


def _analyze_safely(
    candidate: PackedSearchCandidate, include_graph_metrics: bool
) -> dict[str, Any]:
    try:
        return analyze_packed_candidate(
            candidate, include_graph_metrics=include_graph_metrics
        )
    except Exception as error:
        return {
            "schema_version": PACKED_SCHEMA_VERSION,
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
        temporary_path = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[7])
    parser.add_argument(
        "--family", choices=("weight8", "weight12"), default="weight8"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "packed-cache",
    )
    parser.add_argument("--no-symmetry-quotient", action="store_true")
    parser.add_argument("--skip-graph-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    iterator = (
        iter_packed_candidates
        if args.family == "weight8"
        else iter_packed_weight_twelve_candidates
    )
    candidates = sorted(
        candidate
        for m in args.m
        for candidate in iterator(
            m, quotient_symmetries=not args.no_symmetry_quotient
        )
    )
    if args.output is None:
        args.output = DEFAULT_RESULTS_DIR / f"packed-{args.family}.jsonl"
    include_graph_metrics = not args.skip_graph_metrics
    print(
        f"Searching {len(candidates)} packed {args.family} candidates for "
        f"m={sorted(set(args.m))}; "
        f"workers={args.workers}, graph_metrics={include_graph_metrics}",
        flush=True,
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(args.cache_dir))
    records_by_id: dict[str, dict[str, Any]] = {}
    pending: list[PackedSearchCandidate] = []
    for candidate in candidates:
        key = _cache_key(candidate, include_graph_metrics)
        if key in cache:
            records_by_id[candidate.candidate_id] = cache[key]
        else:
            pending.append(candidate)

    print(
        f"Cache hits: {len(candidates) - len(pending)}; pending: {len(pending)}",
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
                        f"Completed {completed}/{len(pending)} pending; "
                        f"accepted so far: {accepted}",
                        flush=True,
                    )

    records = [records_by_id[candidate.candidate_id] for candidate in candidates]
    _write_jsonl_atomic(args.output, records)
    cache.close()

    statuses = Counter(
        "accepted"
        if record.get("accepted")
        else record.get("rejection_reasons", ["unknown"])[0]
        for record in records
    )
    print(f"Wrote {len(records)} records to {args.output}", flush=True)
    print(f"Status counts: {dict(statuses)}", flush=True)


if __name__ == "__main__":
    main()
