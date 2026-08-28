"""Checkpointed fresh search across rank-14 C4 cyclic-module types."""

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

from c4_invariant_n32.search import (
    SCHEMA_VERSION,
    module_partitions,
    search_module_type_batch,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "results" / "c4-invariant-n32"


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
    index: int,
    module_type: tuple[int, ...],
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved: int,
) -> dict[str, Any]:
    try:
        return {
            "index": index,
            "result": search_module_type_batch(
                module_type=module_type,
                trials=trials,
                random_seed=random_seed,
                maximum_check_weight=maximum_check_weight,
                attempts_per_orbit=attempts_per_orbit,
                maximum_saved=maximum_saved,
            ),
        }
    except Exception as error:
        return {
            "index": index,
            "module_type": list(module_type),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trials-per-type", type=int, default=500)
    parser.add_argument("--seed", type=int, default=320402)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=300)
    parser.add_argument("--saved-per-type", type=int, default=10)
    parser.add_argument(
        "--module-type",
        action="append",
        default=[],
        help="Optional comma-separated module type, for example 4,4,4,2.",
    )
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
    parts = output / "module-results"
    parts.mkdir()
    all_types = list(module_partitions())
    if args.module_type:
        types = [tuple(int(part) for part in value.split(",")) for value in args.module_type]
        invalid = [module_type for module_type in types if module_type not in all_types]
        if invalid:
            raise SystemExit(f"invalid rank-14 module types: {invalid}")
    else:
        types = all_types
    records_by_index: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    def checkpoint(status: str) -> None:
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("result", {}).get("counters", {}))
        progress = {
            "status": status,
            "completed_module_types": len(records_by_index),
            "total_module_types": len(types),
            "trials_per_type": args.trials_per_type,
            "constructed": counters["constructed"],
            "connected": counters["connected"],
            "distance_five": counters["distance_5"],
            "distance_six_or_more": sum(
                count
                for name, count in counters.items()
                if name.startswith("distance_") and int(name.split("_")[1]) >= 6
            ),
            "accepted": counters["accepted"],
            "worker_errors": sum("error" in item for item in records_by_index.values()),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(output / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    checkpoint("starting")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                index,
                module_type,
                args.trials_per_type,
                args.seed + 104729 * index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
                args.saved_per_type,
            ): index
            for index, module_type in enumerate(types)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["index"])
            records_by_index[index] = record
            atomic_json(parts / f"module-{index:03d}.json", record)
            checkpoint("searching")
    records = [records_by_index[index] for index in range(len(types))]
    write_jsonl(output / "module-types.jsonl", records)
    survivors = [
        candidate
        for record in records
        for candidate in record.get("result", {}).get("survivors", [])
    ]
    best = [
        candidate
        for record in records
        for candidate in record.get("result", {}).get("best", [])
    ]
    best.sort(
        key=lambda item: (
            -item["certified_distance"],
            int(
                item["distance"]["weight_counts_through_seven"].get(
                    str(item["certified_distance"]), 0
                )
            ),
            item["maximum_check_weight"],
        )
    )
    write_jsonl(output / "survivors.jsonl", survivors)
    write_jsonl(output / "best-candidates.jsonl", best[:200])
    counters: Counter[str] = Counter()
    for record in records:
        counters.update(record.get("result", {}).get("counters", {}))
    viable_types = [
        record["result"]["module_type"]
        for record in records
        if record.get("result", {}).get("counters", {}).get("constructed", 0)
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "fresh C4-invariant [[32,4,>=6]] modules",
        "arguments": vars(args),
        "module_types": len(types),
        "viable_module_types": viable_types,
        "aggregate_counters": dict(sorted(counters.items())),
        "accepted": len(survivors),
        "best_distance": max(
            [item["certified_distance"] for item in survivors + best] + [0]
        ),
        "worker_errors": sum("error" in record for record in records),
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint("complete")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
