"""Checkpointed runner for direct C4-invariant Lagrangian searches."""

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

from searches.c4_invariant_lagrangian.search import SCHEMA_VERSION, search_batch

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results" / "c4-invariant-lagrangian"


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
    batch_index: int,
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    try:
        return {
            "schema_version": SCHEMA_VERSION,
            "batch_index": batch_index,
            "search": search_batch(
                trials=trials,
                random_seed=random_seed,
                maximum_check_weight=maximum_check_weight,
                attempts_per_orbit=attempts_per_orbit,
                maximum_saved_near_misses=maximum_saved_near_misses,
            ),
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "batch_index": batch_index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--trials-per-batch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=470401)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=200)
    parser.add_argument("--maximum-saved-near-misses", type=int, default=10)
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
    parts = output / "batch-results"
    parts.mkdir()
    started = time.perf_counter()
    progress_path = output / "progress.json"

    def checkpoint(value: dict[str, Any]) -> None:
        record = {**value, "seconds": round(time.perf_counter() - started, 6)}
        atomic_json(progress_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint(
        {
            "status": "starting",
            "completed_batches": 0,
            "total_batches": args.batches,
            "completed_trials": 0,
            "total_trials": args.batches * args.trials_per_batch,
        }
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                batch_index,
                args.trials_per_batch,
                args.seed + 104729 * batch_index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
                args.maximum_saved_near_misses,
            ): batch_index
            for batch_index in range(args.batches)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            batch_index = int(record["batch_index"])
            records_by_index[batch_index] = record
            atomic_json(parts / f"batch-{batch_index:05d}.json", record)
            aggregate: Counter[str] = Counter()
            for item in records_by_index.values():
                aggregate.update(item.get("search", {}).get("counters", {}))
            checkpoint(
                {
                    "status": "searching",
                    "completed_batches": completed,
                    "total_batches": args.batches,
                    "completed_trials": aggregate["trials"],
                    "total_trials": args.batches * args.trials_per_batch,
                    "constructed": aggregate["constructed"],
                    "connected": aggregate["connected"],
                    "accepted": aggregate["accepted"],
                    "distance_counts": {
                        key.removeprefix("distance_"): value
                        for key, value in sorted(aggregate.items())
                        if key.startswith("distance_")
                    },
                    "worker_errors": sum(
                        "error" in item for item in records_by_index.values()
                    ),
                }
            )
    records = [records_by_index[index] for index in range(args.batches)]
    write_jsonl(output / "batches.jsonl", records)
    survivors = [
        candidate
        for record in records
        for candidate in record.get("search", {}).get("survivors", [])
    ]
    unique_survivors = {
        tuple(candidate["stabilizer_masks"]): candidate for candidate in survivors
    }
    ordered_survivors = sorted(
        unique_survivors.values(),
        key=lambda item: (
            -item["certified_distance"],
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        ),
    )
    near_misses = [
        candidate
        for record in records
        for candidate in record.get("search", {}).get("near_misses", [])
    ]
    near_misses.sort(
        key=lambda item: (
            -item["certified_distance"],
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        )
    )
    write_jsonl(output / "survivors.jsonl", ordered_survivors)
    write_jsonl(output / "near-misses.jsonl", near_misses[:100])
    aggregate = Counter()
    for record in records:
        aggregate.update(record.get("search", {}).get("counters", {}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "[[28,4,>=6]] with one protected C4 row",
        "method": "random C4-invariant Lagrangians in logical-complement",
        "arguments": vars(args),
        "aggregate_counters": dict(sorted(aggregate.items())),
        "unique_survivors": len(ordered_survivors),
        "best_distance": max(
            [candidate["certified_distance"] for candidate in ordered_survivors]
            + [candidate["certified_distance"] for candidate in near_misses]
            + [0]
        ),
        "worker_errors": sum("error" in item for item in records),
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint(
        {
            "status": "complete",
            "completed_batches": args.batches,
            "total_batches": args.batches,
            "completed_trials": aggregate["trials"],
            "total_trials": args.batches * args.trials_per_batch,
            "accepted": aggregate["accepted"],
            "unique_survivors": len(ordered_survivors),
            "best_distance": summary["best_distance"],
            "summary_path": str(output / "summary.json"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
