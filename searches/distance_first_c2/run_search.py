#!/usr/bin/env python3
"""Run checkpointed distance-first `[[16,2,6]]` CSS discovery."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from searches.distance_first_c2.search import TARGET_DISTANCE, run_restart


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def worker(index: int, args: argparse.Namespace) -> dict[str, Any]:
    result = run_restart(
        n=args.length,
        rank=(args.length - 2) // 2,
        seed=args.seed + 104729 * index,
        iterations=args.iterations,
        proposals_per_iteration=args.proposals,
        downhill_probability=args.downhill_probability,
    )
    return {"restart": index, **result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--length", type=int, choices=(14, 16, 18, 20), default=16)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--proposals", type=int, default=8)
    parser.add_argument("--downhill-probability", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=160206)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--stop-on-hit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length % 2:
        raise SystemExit("ZX-balanced k=2 searches require even n")
    output = args.output_root / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    parts = output / "restarts"
    parts.mkdir()
    atomic_json(output / "arguments.json", {**vars(args), "output_root": str(args.output_root)})
    started = time.perf_counter()
    records: dict[int, dict[str, Any]] = {}
    survivors: dict[str, dict[str, Any]] = {}

    def checkpoint(status: str, active: int | None = None) -> None:
        distances = [int(record["best"]["distance"]) for record in records.values()]
        counters: Counter[str] = Counter()
        for record in records.values():
            counters.update(record.get("counters", {}))
        progress = {
            "status": status,
            "n": args.length,
            "k": 2,
            "rank_x": (args.length - 2) // 2,
            "rank_z": (args.length - 2) // 2,
            "completed_restarts": len(records),
            "total_restarts": args.restarts,
            "active_result": active,
            "best_distance": max(distances + [0]),
            "best_distance_counts": dict(sorted(Counter(distances).items())),
            "survivors": len(survivors),
            "counters": dict(sorted(counters.items())),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(output / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    checkpoint("starting")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(worker, index, args): index for index in range(args.restarts)
    }
    stopped_early = False
    try:
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["restart"])
            records[index] = record
            atomic_json(parts / f"restart-{index:04d}.json", record)
            append_jsonl(output / "completed-restarts.jsonl", record)
            best = record["best"]
            if int(best["distance"]) >= TARGET_DISTANCE:
                key = str(best["state_key"])
                if key not in survivors:
                    survivors[key] = best
                    append_jsonl(output / "survivors.jsonl", best)
            checkpoint("searching", active=index)
            if args.stop_on_hit and survivors:
                stopped_early = True
                for pending in futures:
                    pending.cancel()
                break
    finally:
        executor.shutdown(wait=not stopped_early, cancel_futures=stopped_early)

    ordered = [records[index] for index in sorted(records)]
    distances = [int(record["best"]["distance"]) for record in ordered]
    summary = {
        "target": f"distance-first [[{args.length},2,>=6]] CSS discovery",
        "status": "hit" if survivors else "complete-no-hit",
        "arguments": {**vars(args), "output_root": str(args.output_root)},
        "completed_restarts": len(ordered),
        "best_distance": max(distances + [0]),
        "best_distance_counts": dict(sorted(Counter(distances).items())),
        "survivors": len(survivors),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint(summary["status"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
