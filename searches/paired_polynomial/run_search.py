#!/usr/bin/env python3
"""Run a cached, parallel search over paired-polynomial GALA codes."""

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

from gala_search.core import (
    SEARCH_SCHEMA_VERSION,
    WeightEightCandidate,
    analyze_candidate,
    iter_weight_eight_candidates,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def _cache_key(candidate: WeightEightCandidate, trials: int, base_seed: int) -> str:
    return f"v{SEARCH_SCHEMA_VERSION}:{candidate.candidate_id}:trials={trials}:seed={base_seed}"


def _analyze_safely(
    candidate: WeightEightCandidate,
    trials: int,
    base_seed: int,
    include_graph_metrics: bool,
) -> dict[str, Any]:
    try:
        return analyze_candidate(
            candidate,
            distance_trials=trials,
            base_seed=base_seed,
            include_graph_metrics=include_graph_metrics,
        )
    except Exception as error:  # preserve a failed candidate without aborting the sweep
        return {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reasons": ["worker_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "distance_trials": trials,
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
    parser.add_argument("--m", nargs="+", type=int, default=[7, 11, 13], help="odd C_m orders")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="BP+OSD randomized distance trials per algebraically accepted candidate",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed for reproducible trials")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="number of worker processes",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "weight8.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "cache",
    )
    parser.add_argument(
        "--no-symmetry-quotient",
        action="store_true",
        help="retain every distinct normalized polynomial support",
    )
    parser.add_argument(
        "--skip-graph-metrics",
        action="store_true",
        help="skip Tanner connectivity, girth, and four-cycle measurements",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 0:
        raise SystemExit("--trials must be nonnegative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    candidates = sorted(
        candidate
        for m in args.m
        for candidate in iter_weight_eight_candidates(
            m, quotient_symmetries=not args.no_symmetry_quotient
        )
    )
    print(
        f"Searching {len(candidates)} candidates for m={sorted(set(args.m))}; "
        f"trials={args.trials}, workers={args.workers}",
        flush=True,
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(str(args.cache_dir))
    records_by_id: dict[str, dict[str, Any]] = {}
    pending: list[WeightEightCandidate] = []
    for candidate in candidates:
        key = _cache_key(candidate, args.trials, args.seed)
        if key in cache:
            records_by_id[candidate.candidate_id] = cache[key]
        else:
            pending.append(candidate)

    print(f"Cache hits: {len(candidates) - len(pending)}; pending: {len(pending)}", flush=True)
    completed = 0
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _analyze_safely,
                    candidate,
                    args.trials,
                    args.seed,
                    not args.skip_graph_metrics,
                ): candidate
                for candidate in pending
            }
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                record = future.result()
                records_by_id[candidate.candidate_id] = record
                cache[_cache_key(candidate, args.trials, args.seed)] = record
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    accepted = sum(bool(item.get("accepted")) for item in records_by_id.values())
                    print(
                        f"Completed {completed}/{len(pending)} pending; "
                        f"accepted so far: {accepted}",
                        flush=True,
                    )

    records = [records_by_id[candidate.candidate_id] for candidate in candidates]
    _write_jsonl_atomic(args.output, records)
    cache.close()

    statuses = Counter(
        "accepted" if record.get("accepted") else record.get("rejection_reasons", ["unknown"])[0]
        for record in records
    )
    accepted_records = [record for record in records if record.get("accepted")]
    best_by_m = {
        m: max(
            (
                record["distance_upper_bound"]
                for record in accepted_records
                if record["candidate"]["m"] == m
                and record.get("distance_upper_bound") is not None
            ),
            default=None,
        )
        for m in sorted(set(args.m))
    }
    print(f"Wrote {len(records)} records to {args.output}", flush=True)
    print(f"Status counts: {dict(statuses)}", flush=True)
    print(f"Largest retained distance upper bounds by m: {best_by_m}", flush=True)


if __name__ == "__main__":
    main()
