"""Checkpointed local orbit-replacement search from distance-five codes."""

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

from c4_invariant_lagrangian.search import (
    NUM_QUBITS,
    SCHEMA_VERSION,
    analyze_candidate,
    locally_refine,
    vector_from_mask,
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
    start_index: int,
    candidate: dict[str, Any],
    iterations: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
) -> dict[str, Any]:
    try:
        stabilizer = np.asarray(
            [vector_from_mask(int(mask)) for mask in candidate["stabilizer_masks"]],
            dtype=np.uint8,
        ).reshape(-1, NUM_QUBITS)
        result = locally_refine(
            stabilizer,
            rng=random.Random(random_seed),
            iterations=iterations,
            maximum_check_weight=maximum_check_weight,
            attempts_per_orbit=attempts_per_orbit,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "start_index": start_index,
            "initial_distance": candidate["certified_distance"],
            "result": result,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "start_index": start_index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=470402)
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
    saved_candidates = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    # Older random-search artifacts predate the low-weight spectrum and
    # minimum-weight-basis diagnostics used by the refinement objective.
    # Reconstruct and independently reanalyze every starting rowspace here.
    candidates = []
    for saved in saved_candidates:
        stabilizer = np.asarray(
            [vector_from_mask(int(mask)) for mask in saved["stabilizer_masks"]],
            dtype=np.uint8,
        ).reshape(-1, NUM_QUBITS)
        candidates.append(analyze_candidate(stabilizer))
    candidates = [item for item in candidates if item["certified_distance"] == 5]
    candidates.sort(
        key=lambda item: (
            int(item["distance"]["weight_counts_through_seven"].get("5", 0)),
            item["maximum_check_weight"],
            item["stabilizer_masks"],
        )
    )
    candidates = candidates[: args.starts]
    if not candidates:
        raise SystemExit("input contains no distance-five candidates")
    started = time.perf_counter()
    progress_path = output / "progress.json"

    def checkpoint(value: dict[str, Any]) -> None:
        record = {**value, "seconds": round(time.perf_counter() - started, 6)}
        atomic_json(progress_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint(
        {
            "status": "starting",
            "completed_starts": 0,
            "total_starts": len(candidates),
            "iterations_per_start": args.iterations,
        }
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                index,
                candidate,
                args.iterations,
                args.seed + 104729 * index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            index = int(record["start_index"])
            records_by_index[index] = record
            atomic_json(parts / f"start-{index:05d}.json", record)
            checkpoint(
                {
                    "status": "refining",
                    "completed_starts": completed,
                    "total_starts": len(candidates),
                    "accepted": sum(
                        item.get("result", {}).get("accepted", False)
                        for item in records_by_index.values()
                    ),
                    "best_distance": max(
                        [
                            item.get("result", {}).get("best", {}).get(
                                "certified_distance", 0
                            )
                            for item in records_by_index.values()
                        ]
                        + [0]
                    ),
                    "worker_errors": sum(
                        "error" in item for item in records_by_index.values()
                    ),
                }
            )
    records = [records_by_index[index] for index in range(len(candidates))]
    write_jsonl(output / "starts.jsonl", records)
    accepted = [
        record["result"]["best"]
        for record in records
        if record.get("result", {}).get("accepted")
    ]
    best = sorted(
        [record["result"]["best"] for record in records if "result" in record],
        key=lambda item: (
            -item["certified_distance"],
            int(item["distance"]["weight_counts_through_seven"].get(
                str(item["certified_distance"]), 0
            )),
            item["maximum_check_weight"],
        ),
    )
    write_jsonl(output / "survivors.jsonl", accepted)
    write_jsonl(output / "best-candidates.jsonl", best[:100])
    counters = Counter()
    for record in records:
        counters.update(record.get("result", {}).get("counters", {}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "[[28,4,>=6]] local C4-orbit refinement",
        "arguments": {**vars(args), "input": str(args.input)},
        "starts": len(candidates),
        "aggregate_counters": dict(sorted(counters.items())),
        "accepted": len(accepted),
        "best_distance": max(
            [item["certified_distance"] for item in best] + [0]
        ),
        "worker_errors": sum("error" in record for record in records),
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint(
        {
            "status": "complete",
            "completed_starts": len(candidates),
            "total_starts": len(candidates),
            "accepted": len(accepted),
            "best_distance": summary["best_distance"],
            "summary_path": str(output / "summary.json"),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
