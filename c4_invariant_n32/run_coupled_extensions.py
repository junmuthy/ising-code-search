"""Checkpointed coupled-extension search from saved ``[[28,4,5]]`` seeds."""

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
    decompose_free_orbits,
    vector_from_mask as vector_from_n28_mask,
)
from c4_invariant_n32.search import (
    SCHEMA_VERSION,
    analyze_candidate,
    canonical_rowspace,
    coupled_extension_candidate,
    tanner_connected,
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


def score(candidate: dict[str, Any]) -> tuple[int, int, int]:
    distance = int(candidate["certified_distance"])
    counts = candidate["distance"]["weight_counts_through_seven"]
    return (
        distance,
        -int(counts.get(str(distance), 0)),
        -candidate["maximum_check_weight"],
    )


def task(
    index: int,
    saved: dict[str, Any],
    trials: int,
    random_seed: int,
    maximum_check_weight: int,
    attempts_per_orbit: int,
    maximum_saved: int,
) -> dict[str, Any]:
    try:
        rng = random.Random(random_seed)
        old = np.asarray(
            [vector_from_n28_mask(int(mask)) for mask in saved["stabilizer_masks"]],
            dtype=np.uint8,
        )
        generators = decompose_free_orbits(old)
        counters: Counter[str] = Counter()
        seen: set[bytes] = set()
        best: list[dict[str, Any]] = []
        survivors: list[dict[str, Any]] = []
        started = time.perf_counter()
        omitted_order = list(range(3))
        rng.shuffle(omitted_order)
        for trial in range(trials):
            counters["trials"] += 1
            omitted = omitted_order[trial % len(omitted_order)]
            candidate = coupled_extension_candidate(
                generators,
                omitted_orbit=omitted,
                rng=rng,
                maximum_check_weight=maximum_check_weight,
                attempts_per_orbit=attempts_per_orbit,
            )
            if candidate is None:
                counters["construction_failed"] += 1
                continue
            counters["constructed"] += 1
            key = canonical_rowspace(candidate).tobytes()
            if key in seen:
                counters["duplicate"] += 1
                continue
            seen.add(key)
            if not tanner_connected(candidate):
                counters["disconnected"] += 1
                continue
            counters["connected"] += 1
            analysis = analyze_candidate(candidate)
            analysis["source_n28_stabilizer_masks"] = saved["stabilizer_masks"]
            analysis["omitted_source_orbit"] = omitted
            distance = int(analysis["certified_distance"])
            counters[f"distance_{distance}"] += 1
            if analysis["accepted"]:
                counters["accepted"] += 1
                survivors.append(analysis)
            best.append(analysis)
            best.sort(key=score, reverse=True)
            del best[maximum_saved:]
        return {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "counters": dict(sorted(counters.items())),
            "best": best,
            "survivors": survivors,
            "seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--starts", type=int, default=75)
    parser.add_argument("--trials-per-start", type=int, default=300)
    parser.add_argument("--seed", type=int, default=320401)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--attempts-per-orbit", type=int, default=200)
    parser.add_argument("--saved-per-start", type=int, default=5)
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
    saved = saved[: args.starts]
    if not saved:
        raise SystemExit("input contains no distance-five starts")
    records_by_index: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    def checkpoint(status: str) -> None:
        counters: Counter[str] = Counter()
        for record in records_by_index.values():
            counters.update(record.get("counters", {}))
        progress = {
            "status": status,
            "completed_starts": len(records_by_index),
            "total_starts": len(saved),
            "trials_per_start": args.trials_per_start,
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
                item,
                args.trials_per_start,
                args.seed + 104729 * index,
                args.maximum_check_weight,
                args.attempts_per_orbit,
                args.saved_per_start,
            ): index
            for index, item in enumerate(saved)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["index"])
            records_by_index[index] = record
            atomic_json(parts / f"start-{index:03d}.json", record)
            checkpoint("searching")
    records = [records_by_index[index] for index in range(len(saved))]
    write_jsonl(output / "starts.jsonl", records)
    survivors = [
        candidate
        for record in records
        for candidate in record.get("survivors", [])
    ]
    best = [
        candidate for record in records for candidate in record.get("best", [])
    ]
    best.sort(key=score, reverse=True)
    write_jsonl(output / "survivors.jsonl", survivors)
    write_jsonl(output / "best-candidates.jsonl", best[:200])
    counters: Counter[str] = Counter()
    for record in records:
        counters.update(record.get("counters", {}))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "coupled [[28,4,5]] to [[32,4,>=6]] extensions",
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
