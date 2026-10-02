#!/usr/bin/env python3
"""Checkpointed parallel local refinement of folded ``[[n,2,d]]`` seeds."""

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

from searches.inverse_c2_minimum.local_refinement import load_seed, refine_seed


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = pathlib.Path(output.name)
        for value in values:
            output.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def discover_seeds(
    inputs: list[pathlib.Path],
    minimum_distance: int,
    geometry_name: str,
    module_type: tuple[int, ...],
) -> list[pathlib.Path]:
    paths: set[pathlib.Path] = set()
    for root in inputs:
        if root.is_file():
            candidates = [root]
        else:
            candidates = root.rglob("*-best.json")
        for path in candidates:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                analysis = record["analysis"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if (
                int(analysis.get("k", -1)) == 2
                and int(analysis.get("distance", 0)) >= minimum_distance
                and analysis.get("fold", {}).get("geometry", {}).get("name")
                == geometry_name
                and tuple(int(value) for value in record.get("module_type", ()))
                == module_type
            ):
                paths.add(path)
    return sorted(paths)


def task(
    index: int,
    source: pathlib.Path,
    repetition: int,
    iterations: int,
    attempts_per_replacement: int,
    maximum_check_weight: int,
    random_seed: int,
    replacement_width: int,
    downhill_distance_floor: int,
    downhill_probability: float,
    minimum_start_distance: int,
) -> dict[str, Any]:
    try:
        seed = load_seed(source)
        result = refine_seed(
            seed,
            rng=random.Random(random_seed),
            iterations=iterations,
            attempts_per_replacement=attempts_per_replacement,
            maximum_check_weight=maximum_check_weight,
            replacement_width=replacement_width,
            downhill_distance_floor=downhill_distance_floor,
            downhill_probability=downhill_probability,
            minimum_start_distance=minimum_start_distance,
        )
        return {
            "index": index,
            "source": str(source),
            "fold_index": seed["fold"].index,
            "module_type": list(seed["module_type"]),
            "repetition": repetition,
            "result": result,
        }
    except Exception as error:
        return {
            "index": index,
            "source": str(source),
            "repetition": repetition,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-seed-distance", type=int, default=4)
    parser.add_argument(
        "--geometry-name", default="n16-w6-s2-2-s1-0"
    )
    parser.add_argument("--module-type", default="2,2,2,1")
    parser.add_argument("--repetitions-per-seed", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--attempts-per-replacement", type=int, default=400)
    parser.add_argument("--maximum-check-weight", type=int, default=8)
    parser.add_argument("--replacement-width", type=int, choices=(1, 2), default=1)
    parser.add_argument("--downhill-distance-floor", type=int, default=4)
    parser.add_argument("--downhill-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=160206)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    parts = args.output / "start-results"
    parts.mkdir()
    required_module_type = tuple(
        int(value) for value in args.module_type.split(",") if value
    )
    if any(value not in (1, 2) for value in required_module_type):
        raise SystemExit("--module-type entries must be one or two")
    seeds = discover_seeds(
        args.input,
        args.minimum_seed_distance,
        args.geometry_name,
        required_module_type,
    )
    if not seeds:
        raise SystemExit("no eligible [[n,2,d]] seed files found")
    task_specs = [
        (source, repetition)
        for source in seeds
        for repetition in range(args.repetitions_per_seed)
    ]
    atomic_json(args.output / "selected-seeds.json", [str(path) for path in seeds])
    started = time.perf_counter()
    records_by_index: dict[int, dict[str, Any]] = {}

    def checkpoint(status: str) -> None:
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("result", {}).get("counters", {}))
        distances = [
            int(record.get("result", {}).get("best", {}).get("distance", 0))
            for record in records_by_index.values()
        ]
        progress = {
            "status": status,
            "seeds": len(seeds),
            "completed_starts": len(records_by_index),
            "total_starts": len(task_specs),
            "iterations_per_start": args.iterations,
            "best_distance": max(distances + [0]),
            "distance_counts": dict(sorted(Counter(distances).items())),
            "accepted": sum(
                bool(record.get("result", {}).get("accepted"))
                for record in records_by_index.values()
            ),
            "worker_errors": sum("error" in record for record in records_by_index.values()),
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
                source,
                repetition,
                args.iterations,
                args.attempts_per_replacement,
                args.maximum_check_weight,
                args.seed + 104729 * index,
                args.replacement_width,
                args.downhill_distance_floor,
                args.downhill_probability,
                args.minimum_seed_distance,
            ): index
            for index, (source, repetition) in enumerate(task_specs)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["index"])
            records_by_index[index] = record
            atomic_json(parts / f"start-{index:03d}.json", record)
            checkpoint("refining")
    records = [records_by_index[index] for index in range(len(task_specs))]
    write_jsonl(args.output / "starts.jsonl", records)
    best = [record["result"]["best"] for record in records if "result" in record]
    best.sort(
        key=lambda item: (
            -int(item["distance"]),
            int(item["x_distance"]["weight_counts_through_seven"].get(str(item["distance"]), 0))
            + int(item["z_distance"]["weight_counts_through_seven"].get(str(item["distance"]), 0)),
        )
    )
    write_jsonl(args.output / "best-candidates.jsonl", best)
    survivors = [item for item in best if int(item["distance"]) >= 6]
    write_jsonl(args.output / "survivors.jsonl", survivors)
    counters: Counter[str] = Counter()
    for record in records:
        counters.update(record.get("result", {}).get("counters", {}))
    summary = {
        "target": "C2- and ZX-folded [[n,2,>=6]] local orbit refinement",
        "arguments": {
            **vars(args),
            "input": [str(path) for path in args.input],
            "output": str(args.output),
        },
        "seeds": len(seeds),
        "starts": len(task_specs),
        "best_distance": max([int(item["distance"]) for item in best] + [0]),
        "accepted": len(survivors),
        "worker_errors": sum("error" in record for record in records),
        "counters": dict(sorted(counters.items())),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output / "summary.json", summary)
    checkpoint("complete")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
