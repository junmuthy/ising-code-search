#!/usr/bin/env python3
"""Checkpointed parallel local refinement of folded [[32,4,5]] seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import random
import tempfile
import time
import traceback
from collections import Counter
from typing import Any

from folded_refinement import refine_folded_4442, seed_from_record


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def task(
    index: int,
    record: dict[str, Any],
    iterations: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    try:
        stabilizer, permutation = seed_from_record(record)
        result = refine_folded_4442(
            stabilizer,
            permutation,
            rng=random.Random(random_seed),
            iterations=iterations,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
        )
        return {
            "index": index,
            "source_start_index": record["start_index"],
            "fold_index": record["fold_index"],
            "fold": {
                "epsilon": record["epsilon"],
                "shifts": record["shifts"],
                "pairing": record["pairing"],
            },
            "result": result,
        }
    except Exception as error:
        return {
            "index": index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def select_diverse(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record["distinct_check_spaces"] and record["tanner_connected"]
    ]
    selected = []
    used_starts = set()
    for record in eligible:
        if record["start_index"] in used_starts:
            continue
        selected.append(record)
        used_starts.add(record["start_index"])
        if len(selected) == maximum:
            return selected
    for record in eligible:
        if record in selected:
            continue
        selected.append(record)
        if len(selected) == maximum:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=324242)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=300)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    parts = args.output / "start-results"
    parts.mkdir()
    all_records = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    selected = select_diverse(all_records, args.starts)
    if not selected:
        raise SystemExit("no distinct connected folded seeds in input")
    atomic_json(
        args.output / "selected-seeds.json",
        [
            {
                "start_index": item["start_index"],
                "fold_index": item["fold_index"],
                "epsilon": item["epsilon"],
                "shifts": item["shifts"],
            }
            for item in selected
        ],
    )
    started = time.perf_counter()
    records_by_index: dict[int, dict[str, Any]] = {}

    def checkpoint(status: str) -> None:
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("result", {}).get("counters", {}))
        progress = {
            "status": status,
            "completed_starts": len(records_by_index),
            "total_starts": len(selected),
            "iterations_per_start": args.iterations,
            "best_distance": max(
                [
                    int(record.get("result", {}).get("best", {}).get("distance", 0))
                    for record in records_by_index.values()
                ]
                + [0]
            ),
            "accepted": counters["accepted"],
            "worker_errors": sum("error" in item for item in records_by_index.values()),
            "counters": dict(sorted(counters.items())),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    checkpoint("starting")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                index,
                record,
                args.iterations,
                args.seed + 104729 * index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
            ): index
            for index, record in enumerate(selected)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            index = int(result["index"])
            records_by_index[index] = result
            atomic_json(parts / f"start-{index:03d}.json", result)
            checkpoint("refining")
    records = [records_by_index[index] for index in range(len(selected))]
    write_jsonl(args.output / "starts.jsonl", records)
    best = [record["result"]["best"] for record in records if "result" in record]
    best.sort(
        key=lambda item: (
            -int(item["distance"]),
            int(
                item["distance_detail_z"]["weight_counts_through_seven"].get(
                    str(item["distance"]), 0
                )
            ),
            int(item["maximum_check_weight"]),
        )
    )
    write_jsonl(args.output / "best-candidates.jsonl", best)
    survivors = [item for item in best if item["accepted"]]
    write_jsonl(args.output / "survivors.jsonl", survivors)
    counters: Counter[str] = Counter()
    for record in records:
        counters.update(record.get("result", {}).get("counters", {}))
    summary = {
        "target": "folded (4,4,4,2) local refinement for [[32,4,>=6]]",
        "input": str(args.input),
        "arguments": {
            **vars(args),
            "input": str(args.input),
            "output": str(args.output),
        },
        "starts": len(selected),
        "counters": dict(sorted(counters.items())),
        "accepted": len(survivors),
        "best_distance": max([int(item["distance"]) for item in best] + [0]),
        "worker_errors": sum("error" in record for record in records),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output / "summary.json", summary)
    checkpoint("complete")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
