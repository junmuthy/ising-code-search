"""Checkpointed nonidentity-fold ``D4 x C4``, ``n=32`` search."""

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
    d4_involution_permutations,
    search_exact_fold_space,
    select_fold_seed_witnesses,
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
    witness_index: int,
    witness: dict[str, Any],
    check_fold_index: int,
    check_top_permutation: tuple[int, ...],
    maximum_check_weight: int,
    maximum_saved_near_misses: int,
    maximum_nullity: int,
) -> dict[str, Any]:
    try:
        search = search_exact_fold_space(
            witness["seed_support"],
            data_top_permutation=witness["data_top_permutation"],
            check_top_permutation=check_top_permutation,
            maximum_check_weight=maximum_check_weight,
            maximum_saved_near_misses=maximum_saved_near_misses,
            maximum_nullity=maximum_nullity,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness_index": witness_index,
            "check_fold_index": check_fold_index,
            "witness": witness,
            "search": search,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness_index": witness_index,
            "check_fold_index": check_fold_index,
            "witness": witness,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--witnesses-per-data-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=440402)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--maximum-saved-near-misses", type=int, default=10)
    parser.add_argument("--maximum-nullity", type=int, default=22)
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
    started = time.perf_counter()

    def checkpoint(value: dict[str, Any]) -> None:
        record = {**value, "seconds": round(time.perf_counter() - started, 6)}
        atomic_json(progress_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint({"status": "selecting_witnesses", "completed_spaces": 0})
    witnesses = select_fold_seed_witnesses(
        witnesses_per_data_fold=args.witnesses_per_data_fold,
        random_seed=args.seed,
    )
    check_folds = d4_involution_permutations()
    identity = tuple(range(4))
    tasks = []
    for witness_index, witness in enumerate(witnesses):
        for check_fold_index, check_top_permutation in enumerate(check_folds):
            if (
                tuple(witness["data_top_permutation"]) == identity
                and check_top_permutation == identity
            ):
                continue
            tasks.append(
                (
                    len(tasks),
                    witness_index,
                    witness,
                    check_fold_index,
                    check_top_permutation,
                )
            )
    write_jsonl(output / "fold-seed-witnesses.jsonl", witnesses)
    checkpoint(
        {
            "status": "starting",
            "completed_spaces": 0,
            "total_spaces": len(tasks),
            "feasible_data_folds": len(
                {tuple(item["data_top_permutation"]) for item in witnesses}
            ),
            "geometrically_impossible_data_folds": len(check_folds)
            - len({tuple(item["data_top_permutation"]) for item in witnesses}),
        }
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                *item,
                args.maximum_check_weight,
                args.maximum_saved_near_misses,
                args.maximum_nullity,
            ): item[0]
            for item in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            index = int(record["task_index"])
            records_by_index[index] = record
            atomic_json(parts / f"space-{index:05d}.json", record)
            checkpoint(
                {
                    "status": "fold_space_search",
                    "completed_spaces": completed,
                    "total_spaces": len(tasks),
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
                    "nullity_skips": sum(
                        item.get("search", {}).get("status")
                        == "nullity_above_exact_limit"
                        for item in records_by_index.values()
                    ),
                    "worker_errors": sum(
                        "error" in item for item in records_by_index.values()
                    ),
                }
            )
    records = [records_by_index[index] for index in range(len(tasks))]
    write_jsonl(output / "fold-spaces.jsonl", records)
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
    aggregate: Counter[str] = Counter()
    dimensions: Counter[int] = Counter()
    nullities: Counter[int] = Counter()
    for record in records:
        search = record.get("search", {})
        aggregate.update(search.get("counters", {}))
        if "constraint_nullity" in search:
            nullities[int(search["constraint_nullity"])] += 1
        for dimension, count in search.get("dimension_counts", {}).items():
            dimensions[int(dimension)] += int(count)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "[[32,k,>=6]] with k>=4 and one protected C4 row",
        "representation": "faithful four-point D4 permutation module times C4",
        "protograph": "L=2, J=1 two-block GALA",
        "method": "exhaustive involutive D4 data/check-fold constraint spaces",
        "arguments": vars(args),
        "witnesses": len(witnesses),
        "feasible_data_folds": len(
            {tuple(item["data_top_permutation"]) for item in witnesses}
        ),
        "geometrically_impossible_data_folds": len(check_folds)
        - len({tuple(item["data_top_permutation"]) for item in witnesses}),
        "fold_spaces": len(records),
        "worker_errors": sum("error" in item for item in records),
        "nullity_counts": dict(sorted(nullities.items())),
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
            "completed_spaces": len(records),
            "total_spaces": len(records),
            "survivors": len(survivors),
            "summary_path": str(output / "summary.json"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
