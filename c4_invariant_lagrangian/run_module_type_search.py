"""Checkpointed search across all C4 cyclic-module types at ``[[28,4,*]]``."""

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

from c4_invariant_lagrangian.search import (
    SCHEMA_VERSION,
    module_partitions,
    search_module_type_batch,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
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
    index: int,
    module_type: tuple[int, ...],
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved_near_misses: int,
) -> dict[str, Any]:
    try:
        result = search_module_type_batch(
            module_type=module_type,
            trials=trials,
            random_seed=random_seed,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
            maximum_saved_near_misses=maximum_saved_near_misses,
        )
        return {"index": index, "result": result}
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
    parser.add_argument("--seed", type=int, default=470403)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=400)
    parser.add_argument("--saved-near-misses-per-type", type=int, default=10)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument(
        "--include-free-type",
        action="store_true",
        help="Also rerun the already-sampled (4,4,4) free module type.",
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
    types = list(module_partitions())
    if not args.include_free_type:
        types = [module_type for module_type in types if module_type != (4, 4, 4)]
    started = time.perf_counter()
    records_by_index: dict[int, dict[str, Any]] = {}

    def checkpoint(status: str) -> None:
        complete = len(records_by_index)
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("result", {}).get("counters", {}))
        progress = {
            "status": status,
            "completed_module_types": complete,
            "total_module_types": len(types),
            "trials_per_type": args.trials_per_type,
            "constructed": counters["constructed"],
            "connected": counters["connected"],
            "distance_five": counters["distance_5"],
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
                args.saved_near_misses_per_type,
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
        survivor
        for record in records
        for survivor in record.get("result", {}).get("survivors", [])
    ]
    near_misses = [
        candidate
        for record in records
        for candidate in record.get("result", {}).get("near_misses", [])
    ]
    near_misses.sort(
        key=lambda item: (
            -item["certified_distance"],
            item["maximum_check_weight"],
            item["module_type"],
        )
    )
    write_jsonl(output / "survivors.jsonl", survivors)
    write_jsonl(output / "near-misses.jsonl", near_misses)
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
        "target": "[[28,4,>=6]] all cyclic C4 module types",
        "arguments": vars(args),
        "module_types": len(types),
        "viable_module_types": viable_types,
        "aggregate_counters": dict(sorted(counters.items())),
        "accepted": len(survivors),
        "best_distance": max(
            [item["certified_distance"] for item in survivors + near_misses] + [0]
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
