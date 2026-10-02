#!/usr/bin/env python3
"""Solve canonical fixed-C sectors of a balanced `[[n,2,6]]` CSS problem."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import z3

from searches.distance_first_c2.search import analyze_css
from searches.distance_first_c2.systematic import CSector, SystematicCSSSolver, canonical_c_sectors


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


def solve_sector(
    index: int,
    sector: CSector,
    n: int,
    timeout_d5_seconds: float,
    timeout_d6_seconds: float,
    direct_d6: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        solver = SystematicCSSSolver(
            n,
            fixed_c=sector.matrix,
            timeout_ms=max(
                1,
                int(1000 * (timeout_d6_seconds if direct_d6 else timeout_d5_seconds)),
            ),
        )
        if direct_d6:
            build_d6_started = time.perf_counter()
            for weight in range(1, 6):
                solver.add_weight(weight)
            build_d6_seconds = time.perf_counter() - build_d6_started
            solve_d6_started = time.perf_counter()
            status_d6 = solver.check()
            record: dict[str, Any] = {
                "index": index,
                "sector": sector.name,
                "counts": list(sector.counts),
                "c_matrix": [list(row) for row in sector.matrix],
                "raw_multiplicity": sector.raw_multiplicity,
                "direct_d6": True,
                "d5_status": "not-run",
                "d6_status": str(status_d6),
                "d6_reason_unknown": (
                    solver.solver.reason_unknown()
                    if status_d6 == z3.unknown
                    else None
                ),
                "build_d6_seconds": round(build_d6_seconds, 6),
                "solve_d6_seconds": round(
                    time.perf_counter() - solve_d6_started, 6
                ),
            }
            if status_d6 == z3.sat:
                record["survivor"] = analyze_css(solver.state())
            record["seconds"] = round(time.perf_counter() - started, 6)
            return record
        build_d5_started = time.perf_counter()
        for weight in range(1, 5):
            solver.add_weight(weight)
        build_d5_seconds = time.perf_counter() - build_d5_started
        solve_d5_started = time.perf_counter()
        status_d5 = solver.check()
        solve_d5_seconds = time.perf_counter() - solve_d5_started
        record: dict[str, Any] = {
            "index": index,
            "sector": sector.name,
            "counts": list(sector.counts),
            "c_matrix": [list(row) for row in sector.matrix],
            "raw_multiplicity": sector.raw_multiplicity,
            "d5_status": str(status_d5),
            "d5_reason_unknown": (
                solver.solver.reason_unknown() if status_d5 == z3.unknown else None
            ),
            "build_d5_seconds": round(build_d5_seconds, 6),
            "solve_d5_seconds": round(solve_d5_seconds, 6),
        }
        if status_d5 != z3.sat:
            record["seconds"] = round(time.perf_counter() - started, 6)
            return record

        model_d5 = analyze_css(solver.state())
        record["model_d5"] = model_d5
        build_d6_started = time.perf_counter()
        solver.add_weight(5)
        record["build_d6_seconds"] = round(
            time.perf_counter() - build_d6_started, 6
        )
        solver.set_timeout(max(1, int(1000 * timeout_d6_seconds)))
        solve_d6_started = time.perf_counter()
        status_d6 = solver.check()
        record["solve_d6_seconds"] = round(
            time.perf_counter() - solve_d6_started, 6
        )
        record["d6_status"] = str(status_d6)
        record["d6_reason_unknown"] = (
            solver.solver.reason_unknown() if status_d6 == z3.unknown else None
        )
        if status_d6 == z3.sat:
            record["survivor"] = analyze_css(solver.state())
        record["seconds"] = round(time.perf_counter() - started, 6)
        return record
    except Exception as error:
        return {
            "index": index,
            "sector": sector.name,
            "counts": list(sector.counts),
            "raw_multiplicity": sector.raw_multiplicity,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.perf_counter() - started, 6),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--length", type=int, choices=(14, 16, 18, 20), default=16)
    parser.add_argument("--sector-start", type=int, default=0)
    parser.add_argument("--sector-stop", type=int)
    parser.add_argument("--sector-index", type=int, action="append")
    parser.add_argument("--maximum-sectors", type=int)
    parser.add_argument("--timeout-d5-seconds", type=float, default=60)
    parser.add_argument("--timeout-d6-seconds", type=float, default=120)
    parser.add_argument(
        "--direct-d6",
        action="store_true",
        help="add weights one through five before the first solver call",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--stop-on-hit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    parts = output / "sectors"
    parts.mkdir()
    all_sectors = canonical_c_sectors((args.length - 2) // 2)
    stop = len(all_sectors) if args.sector_stop is None else args.sector_stop
    selected = list(enumerate(all_sectors))[args.sector_start : stop]
    if args.sector_index is not None:
        requested = set(args.sector_index)
        invalid = requested - set(range(len(all_sectors)))
        if invalid:
            raise SystemExit(f"invalid C-sector indices: {sorted(invalid)}")
        selected = [item for item in selected if item[0] in requested]
    if args.maximum_sectors is not None:
        selected = selected[: args.maximum_sectors]
    if not selected:
        raise SystemExit("no C sectors selected")
    atomic_json(
        output / "arguments.json",
        {
            **vars(args),
            "output_root": str(args.output_root),
            "canonical_sectors_total": len(all_sectors),
            "raw_c_matrices_total": sum(s.raw_multiplicity for s in all_sectors),
            "selected_indices": [index for index, _sector in selected],
        },
    )
    started = time.perf_counter()
    records: dict[int, dict[str, Any]] = {}
    survivors: dict[str, dict[str, Any]] = {}

    def checkpoint(status: str, active: int | None = None) -> None:
        d5_counts = Counter(record.get("d5_status", "error") for record in records.values())
        d6_counts = Counter(
            record.get("d6_status", "not-run") for record in records.values()
        )
        progress = {
            "status": status,
            "n": args.length,
            "completed_sectors": len(records),
            "selected_sectors": len(selected),
            "canonical_sectors_total": len(all_sectors),
            "raw_c_matrices_total": sum(s.raw_multiplicity for s in all_sectors),
            "active_result": active,
            "d5_status_counts": dict(sorted(d5_counts.items())),
            "d6_status_counts": dict(sorted(d6_counts.items())),
            "survivors": len(survivors),
            "worker_errors": sum("error" in record for record in records.values()),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(output / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)

    checkpoint("starting")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            solve_sector,
            index,
            sector,
            args.length,
            args.timeout_d5_seconds,
            args.timeout_d6_seconds,
            args.direct_d6,
        ): index
        for index, sector in selected
    }
    stopped_early = False
    try:
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = int(record["index"])
            records[index] = record
            atomic_json(parts / f"sector-{index:03d}.json", record)
            append_jsonl(output / "completed-sectors.jsonl", record)
            if "survivor" in record:
                key = str(record["survivor"]["state_key"])
                if key not in survivors:
                    survivors[key] = record["survivor"]
                    append_jsonl(output / "survivors.jsonl", record)
            checkpoint("searching", active=index)
            if args.stop_on_hit and survivors:
                stopped_early = True
                for pending in futures:
                    pending.cancel()
                break
    finally:
        executor.shutdown(wait=not stopped_early, cancel_futures=stopped_early)

    d5_counts = Counter(record.get("d5_status", "error") for record in records.values())
    d6_counts = Counter(record.get("d6_status", "not-run") for record in records.values())
    unresolved = sum(
        record.get("d5_status") == "unknown" or record.get("d6_status") == "unknown"
        for record in records.values()
    )
    complete = len(records) == len(selected) and not unresolved and not any(
        "error" in record for record in records.values()
    )
    summary = {
        "target": f"fixed-C systematic [[{args.length},2,>=6]] CSS synthesis",
        "status": "hit" if survivors else ("complete" if complete else "incomplete"),
        "completed_sectors": len(records),
        "selected_sectors": len(selected),
        "canonical_sectors_total": len(all_sectors),
        "raw_c_matrices_total": sum(s.raw_multiplicity for s in all_sectors),
        "d5_status_counts": dict(sorted(d5_counts.items())),
        "d6_status_counts": dict(sorted(d6_counts.items())),
        "survivors": len(survivors),
        "unresolved_sectors": unresolved,
        "worker_errors": sum("error" in record for record in records.values()),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint(summary["status"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
