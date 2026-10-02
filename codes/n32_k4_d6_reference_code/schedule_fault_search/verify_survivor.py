#!/usr/bin/env python3
"""Run full-noise bulk and memory certification for a saved schedule survivor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from codes.n32_k4_d6_reference_code.schedule_fault_search.data import code_for_schedule
from codes.n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_through_three,
    certify_through_four,
    estimate_four_fault_table,
    heuristic_screen,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.circuit import (
    NoiseModel,
    build_bulk_circuit,
    build_memory_circuit,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import atomic_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-rounds", default="3,18")
    parser.add_argument("--skip-bulk", action="store_true")
    parser.add_argument("--max-faults", type=int, default=6)
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--maximum-pair-table-mib",
        type=float,
        default=1024,
        help="defer exact four-fault exclusion when its compact table would exceed this size",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.output.with_suffix(".checkpoint.json")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    code = code_for_schedule(schedule)
    rounds = sorted({int(value) for value in args.memory_rounds.split(",") if value})
    experiments = ([] if args.skip_bulk else [("bulk", 3)]) + [
        ("memory", value) for value in rounds
    ]
    configuration = {
        "schedule": str(args.schedule.resolve()),
        "schedule_id": schedule["schedule_id"],
        "experiments": experiments,
        "bases": ["X", "Z"],
        "fault_model": "full",
        "probability": args.probability,
        "max_faults": args.max_faults,
        "heartbeat_seconds": args.heartbeat_seconds,
        "maximum_pair_table_mib": args.maximum_pair_table_mib,
        "skip_bulk": args.skip_bulk,
    }
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["configuration"] != configuration:
            raise SystemExit("resume configuration mismatch")
        state["status"] = "running"
    else:
        state = {
            "schema_version": 1,
            "started": utc_now(),
            "updated": utc_now(),
            "status": "running",
            "configuration": configuration,
            "results": {},
            "active": None,
        }
    atomic_json(checkpoint, state)
    noise = NoiseModel(args.probability)
    for experiment, round_count in experiments:
        for basis in ("X", "Z"):
            key = f"{experiment}-r{round_count}-{basis}"
            if key in state["results"]:
                continue
            circuit = (
                build_bulk_circuit(basis, noise=noise, code=code)
                if experiment == "bulk"
                else build_memory_circuit(basis, round_count, noise=noise, code=code)
            )

            def progress(event: dict[str, Any]) -> None:
                state["updated"] = utc_now()
                state["active"] = {"key": key, **event}
                atomic_json(checkpoint, state)
                fields = " ".join(
                    f"{name}={value}" for name, value in event.items() if name != "timestamp"
                )
                print(f"checkpoint {event.get('timestamp', utc_now())} key={key} {fields}", flush=True)

            heuristic = heuristic_screen(
                circuit,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=progress,
            )
            upper_bound = heuristic["upper_bound"]
            estimate = estimate_four_fault_table(circuit)
            limit_bytes = round(args.maximum_pair_table_mib * 1024 * 1024)
            result: dict[str, Any] = {
                "heuristic": heuristic,
                "four_fault_table_estimate": estimate,
                "maximum_pair_table_bytes": limit_bytes,
                "exact_through_four": None,
                "distance": None,
                "lower_bound": None,
                "status": "screened",
            }
            if upper_bound is not None and int(upper_bound) <= 4:
                exact_three = certify_through_three(
                    circuit,
                    heartbeat_seconds=args.heartbeat_seconds,
                    progress=progress,
                )
                result["exact_through_three"] = exact_three
                if exact_three["distance"] is not None:
                    result["distance"] = exact_three["distance"]
                    result["lower_bound"] = exact_three["distance"]
                else:
                    result["distance"] = int(upper_bound)
                    result["lower_bound"] = 4
                result["status"] = "exact-distance-at-most-four"
            elif estimate["table_bytes"] <= limit_bytes:
                exact = certify_through_four(
                    circuit,
                    heartbeat_seconds=args.heartbeat_seconds,
                    progress=progress,
                )
                result["exact_through_four"] = exact
                result["lower_bound"] = exact["lower_bound"]
                if exact["distance"] is not None:
                    result["distance"] = exact["distance"]
                    result["status"] = "exact-distance-at-most-four"
                elif upper_bound is not None and int(upper_bound) == 5:
                    result["distance"] = 5
                    result["status"] = "exact-distance-five"
                else:
                    result["status"] = "exact-lower-bound-five"
            else:
                result["lower_bound"] = 1
                result["status"] = "exact-four-deferred-memory-limit"
            state["results"][key] = result
            state["active"] = None
            state["updated"] = utc_now()
            atomic_json(checkpoint, state)
    state["status"] = "complete"
    state["completed"] = utc_now()
    state["updated"] = utc_now()
    state["active"] = None
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
