#!/usr/bin/env python3
"""Checkpointed exact guarded-bulk screening of the 32 clean BB64 schedules."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from codes.n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_through_four,
    estimate_four_fault_table,
    heuristic_screen,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    atomic_json,
    utc_now,
)

from searches.bb64_simultaneous_basis_search.stim_fault_distance.circuit import (
    NoiseModel,
    build_bulk_circuit,
)
from searches.bb64_simultaneous_basis_search.stim_fault_distance.model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-survivor", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compact_heuristic(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "upper_bound": result["upper_bound"],
        "elapsed_seconds": result["elapsed_seconds"],
        "explained_errors": result["explained_errors"],
    }


def main() -> None:
    args = parse_args()
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive")
    schedules = sorted(args.schedule_dir.glob("schedule-*.json"))[: args.max_candidates]
    if not schedules:
        raise SystemExit(f"no complete schedules found under {args.schedule_dir}")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    configuration = {
        "schema_version": 1,
        "schedule_dir": str(args.schedule_dir.resolve()),
        "schedule_count": len(schedules),
        "schedule_files": [path.name for path in schedules],
        "bases": list(bases),
        "probability": args.probability,
        "heartbeat_seconds": args.heartbeat_seconds,
        "screen": (
            "Stim heuristic upper bound followed by exact one-through-four "
            "guarded-bulk CNOT-fault certification"
        ),
    }
    if args.output_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = args.output_dir / "configuration.json"
    records_path = args.output_dir / "candidates.jsonl"
    progress_path = args.output_dir / "progress.json"
    pending_path = args.output_dir / "pending.json"
    if args.resume:
        if not configuration_path.exists():
            raise SystemExit("resume configuration is absent")
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise SystemExit("resume configuration mismatch")
    else:
        atomic_json(configuration_path, configuration)

    records = load_jsonl(records_path)
    completed_ids = {str(record["schedule_id"]) for record in records}
    status_counts = Counter(str(record["status"]) for record in records)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started": utc_now(),
        "updated": utc_now(),
        "configuration": configuration,
        "completed": len(records),
        "status_histogram": dict(sorted(status_counts.items())),
        "active": None,
    }
    if args.resume and progress_path.exists():
        old = json.loads(progress_path.read_text(encoding="utf-8"))
        state["started"] = old.get("started", state["started"])
    atomic_json(progress_path, state)
    total_started = time.monotonic()

    for schedule_number, schedule_path in enumerate(schedules, start=1):
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        identifier = str(schedule["schedule_id"])
        if identifier in completed_ids:
            continue
        record: dict[str, Any] = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "schedule_number": schedule_number,
            "schedule_id": identifier,
            "schedule": str(schedule_path.resolve()),
            "edge_orbit_colors_zero_based": schedule["edge_orbit_colors_zero_based"],
            "status": "screening",
            "bulk_cnot": {},
        }
        atomic_json(pending_path, record)
        code = load_code_data(schedule_path)
        noise = NoiseModel.cnot_only(args.probability)
        passed = True
        candidate_started = time.monotonic()
        for basis in bases:
            circuit = build_bulk_circuit(basis, noise=noise, code=code)
            estimate = estimate_four_fault_table(circuit)

            def progress(event: dict[str, Any]) -> None:
                state["updated"] = utc_now()
                state["active"] = {
                    "schedule_number": schedule_number,
                    "schedule_id": identifier,
                    "basis": basis,
                    **event,
                }
                atomic_json(progress_path, state)
                fields = " ".join(
                    f"{key}={value}" for key, value in event.items() if key != "timestamp"
                )
                print(
                    f"checkpoint {event.get('timestamp', utc_now())} "
                    f"schedule={schedule_number}/{len(schedules)} id={identifier} "
                    f"basis={basis} {fields}",
                    flush=True,
                )

            print(
                f"checkpoint {utc_now()} schedule={schedule_number}/{len(schedules)} "
                f"id={identifier} basis={basis} phase=heuristic-start "
                f"effects={estimate['distinct_fault_effects']} "
                f"pair_table_bytes={estimate['table_bytes']}",
                flush=True,
            )
            heuristic = heuristic_screen(
                circuit,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=progress,
            )
            basis_result: dict[str, Any] = {
                "table_estimate": estimate,
                "heuristic": compact_heuristic(heuristic),
            }
            upper_bound = heuristic["upper_bound"]
            if upper_bound is not None and int(upper_bound) <= 4:
                basis_result["outcome"] = f"concrete-{upper_bound}-fault-witness"
                record["bulk_cnot"][basis] = basis_result
                record["status"] = f"rejected-bulk-{basis}-heuristic-d{upper_bound}"
                passed = False
                break
            print(
                f"checkpoint {utc_now()} schedule={schedule_number}/{len(schedules)} "
                f"id={identifier} basis={basis} phase=exact-through-four-start",
                flush=True,
            )
            exact = certify_through_four(
                circuit,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=progress,
            )
            basis_result["exact_through_four"] = exact
            record["bulk_cnot"][basis] = basis_result
            if int(exact["lower_bound"]) < 5:
                basis_result["outcome"] = "exact-four-fault-witness"
                record["status"] = f"rejected-bulk-{basis}-exact-d4"
                passed = False
                break
            basis_result["outcome"] = "exact-lower-bound-five"

        if passed:
            record["status"] = "bulk-cnot-survivor-lower-bound-five"
            survivor_path = args.output_dir / "survivors" / f"{identifier}.schedule.json"
            atomic_json(survivor_path, schedule)
            record["survivor_schedule"] = str(survivor_path.resolve())
        record["elapsed_seconds"] = round(time.monotonic() - candidate_started, 3)
        append_jsonl(records_path, record)
        pending_path.unlink(missing_ok=True)
        records.append(record)
        completed_ids.add(identifier)
        status_counts[record["status"]] += 1
        state.update(
            {
                "updated": utc_now(),
                "completed": len(records),
                "status_histogram": dict(sorted(status_counts.items())),
                "active": None,
                "elapsed_seconds_this_invocation": round(
                    time.monotonic() - total_started, 3
                ),
            }
        )
        atomic_json(progress_path, state)
        print(
            f"checkpoint {utc_now()} schedule={schedule_number}/{len(schedules)} "
            f"id={identifier} status={record['status']} "
            f"histogram={json.dumps(dict(sorted(status_counts.items())), sort_keys=True)}",
            flush=True,
        )
        if args.stop_on_survivor and passed:
            break

    state.update(
        {
            "status": "complete" if len(records) >= len(schedules) else "stopped-on-survivor",
            "completed_at": utc_now(),
            "updated": utc_now(),
            "elapsed_seconds_this_invocation": round(time.monotonic() - total_started, 3),
            "active": None,
        }
    )
    atomic_json(progress_path, state)
    atomic_json(args.output_dir / "summary.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
