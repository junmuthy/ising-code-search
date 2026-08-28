"""Checkpointed runner for the faithful ``D4 x C4``, ``n=32`` search."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import time
import traceback
from collections import Counter
from typing import Any

from d4_single_row.search import (
    SCHEMA_VERSION,
    search_constraint_space,
    select_seed_witnesses,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "results" / "single-row" / "d4-c4-n32"


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def task(
    task_index: int,
    witness: dict[str, Any],
    maximum_check_weight: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    try:
        search = search_constraint_space(
            witness["seed_support"],
            maximum_check_weight=maximum_check_weight,
            maximum_saved_near_misses=maximum_saved_near_misses,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness": witness,
            "search": search,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness": witness,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed-witnesses", type=int, default=16)
    parser.add_argument("--seed", type=int, default=440401)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--maximum-saved-near-misses", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = RESULTS_ROOT / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    parts = output / "task-results"
    parts.mkdir()
    progress_path = output / "progress.json"
    witnesses = select_seed_witnesses(
        count=args.seed_witnesses, random_seed=args.seed
    )
    write_jsonl(output / "seed-witnesses.jsonl", witnesses)
    started = time.perf_counter()

    def checkpoint(value: dict[str, Any]) -> None:
        record = {**value, "seconds": round(time.perf_counter() - started, 6)}
        atomic_json(progress_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint(
        {"status": "starting", "completed_tasks": 0, "total_tasks": len(witnesses)}
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                index,
                witness,
                args.maximum_check_weight,
                args.maximum_saved_near_misses,
            ): index
            for index, witness in enumerate(witnesses)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            index = int(record["task_index"])
            records_by_index[index] = record
            atomic_json(parts / f"task-{index:04d}.json", record)
            checkpoint(
                {
                    "status": "generator_search",
                    "completed_tasks": completed,
                    "total_tasks": len(witnesses),
                    "structural_candidates": sum(
                        item.get("search", {}).get("counters", {}).get(
                            "structural_candidates", 0
                        )
                        for item in records_by_index.values()
                    ),
                    "distance_hits": sum(
                        item.get("search", {}).get("counters", {}).get(
                            "distance_at_least_six", 0
                        )
                        for item in records_by_index.values()
                    ),
                    "worker_errors": sum(
                        "error" in item for item in records_by_index.values()
                    ),
                }
            )
    records = [records_by_index[index] for index in range(len(witnesses))]
    write_jsonl(output / "generator-spaces.jsonl", records)
    survivors = [
        hit
        for record in records
        for hit in record.get("search", {}).get("survivors", [])
    ]
    near_misses = [
        hit
        for record in records
        for hit in record.get("search", {}).get("near_misses", [])
    ]
    write_jsonl(output / "survivors.jsonl", survivors)
    write_jsonl(output / "near-misses.jsonl", near_misses)
    dimensions: Counter[int] = Counter()
    aggregate: Counter[str] = Counter()
    for record in records:
        for dimension, count in record.get("search", {}).get(
            "dimension_counts", {}
        ).items():
            dimensions[int(dimension)] += int(count)
        aggregate.update(record.get("search", {}).get("counters", {}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "[[32,k,>=6]] with k>=4 and one protected C4 row",
        "representation": "faithful four-point D4 permutation module times C4",
        "protograph": "L=2, J=1 two-block GALA",
        "method": "exact identity-ZX-fold seed-constrained spaces",
        "arguments": vars(args),
        "witnesses": len(witnesses),
        "worker_errors": sum("error" in item for item in records),
        "aggregate_counters": dict(sorted(aggregate.items())),
        "dimension_counts": dict(sorted(dimensions.items())),
        "survivors": len(survivors),
        "saved_near_misses": len(near_misses),
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint(
        {
            "status": "complete",
            "completed_tasks": len(witnesses),
            "total_tasks": len(witnesses),
            "survivors": len(survivors),
            "summary_path": str(output / "summary.json"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
