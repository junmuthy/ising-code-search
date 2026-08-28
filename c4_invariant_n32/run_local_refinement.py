"""Checkpointed orbit-replacement refinement of n=32 distance-five codes."""

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

import numpy as np

from c4_invariant_n32.search import (
    NUM_QUBITS,
    SCHEMA_VERSION,
    locally_refine_4442,
    vector_from_mask,
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
    saved: dict[str, Any],
    iterations: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    try:
        stabilizer = np.asarray(
            [vector_from_mask(int(mask)) for mask in saved["stabilizer_masks"]],
            dtype=np.uint8,
        ).reshape(-1, NUM_QUBITS)
        result = locally_refine_4442(
            stabilizer,
            rng=random.Random(random_seed),
            iterations=iterations,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
        )
        return {"index": index, "result": result}
    except Exception as error:
        return {
            "index": index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=320403)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=200)
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
    parts = output / "start-results"
    parts.mkdir()
    saved = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    saved = [item for item in saved if item["certified_distance"] == 5]
    saved.sort(
        key=lambda item: (
            int(item["distance"]["weight_counts_through_seven"].get("5", 0)),
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        )
    )
    saved = saved[: args.starts]
    if not saved:
        raise SystemExit("input contains no distance-five starts")
    records_by_index: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    def checkpoint(status: str) -> None:
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("result", {}).get("counters", {}))
        progress = {
            "status": status,
            "completed_starts": len(records_by_index),
            "total_starts": len(saved),
            "iterations_per_start": args.iterations,
            "best_distance": max(
                [
                    record.get("result", {}).get("best", {}).get(
                        "certified_distance", 0
                    )
                    for record in records_by_index.values()
                ]
                + [0]
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
                item,
                args.iterations,
                args.seed + 104729 * index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
            ): index
            for index, item in enumerate(saved)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["index"])
            records_by_index[index] = record
            atomic_json(parts / f"start-{index:03d}.json", record)
            checkpoint("refining")
    records = [records_by_index[index] for index in range(len(saved))]
    write_jsonl(output / "starts.jsonl", records)
    survivors = [
        record["result"]["best"]
        for record in records
        if record.get("result", {}).get("accepted")
    ]
    best = [
        record["result"]["best"] for record in records if "result" in record
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
    write_jsonl(output / "best-candidates.jsonl", best)
    counters: Counter[str] = Counter()
    for record in records:
        counters.update(record.get("result", {}).get("counters", {}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "local (4,4,4,2) refinement for [[32,4,>=6]]",
        "arguments": {**vars(args), "input": str(args.input)},
        "starts": len(saved),
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
