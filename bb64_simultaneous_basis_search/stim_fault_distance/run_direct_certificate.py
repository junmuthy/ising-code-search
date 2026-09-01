#!/usr/bin/env python3
"""Run low-memory exact one-through-three BB64 fault certification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_through_four,
    certify_through_three,
    estimate_four_fault_table,
)
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    atomic_json,
    utc_now,
)

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--experiment", choices=("bulk", "memory"), default="memory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--fault-model", choices=("cnot", "no-idle", "full"), default="full")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-faults", type=int, choices=(3, 4), default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def noise_from_args(name: str, probability: float) -> NoiseModel:
    if name == "cnot":
        return NoiseModel.cnot_only(probability)
    if name == "no-idle":
        return NoiseModel.without_idles(probability)
    return NoiseModel(probability)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output}")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    configuration = {
        "schema_version": 1,
        "schedule": str(args.schedule.resolve()),
        "experiment": args.experiment,
        "rounds": args.rounds,
        "bases": list(bases),
        "fault_model": args.fault_model,
        "probability": args.probability,
        "maximum_fault_cardinality": args.max_faults,
    }
    checkpoint = args.output.with_suffix(".checkpoint.json")
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["configuration"] != configuration:
            raise SystemExit("resume configuration mismatch")
        state["status"] = "running"
    else:
        state: dict[str, Any] = {
            "schema_version": 1,
            "started": utc_now(),
            "updated": utc_now(),
            "status": "running",
            "configuration": configuration,
            "results": {},
            "active": None,
        }
    atomic_json(checkpoint, state)
    code = load_code_data(args.schedule)
    noise = noise_from_args(args.fault_model, args.probability)
    started = time.monotonic()
    for basis in bases:
        if basis in state["results"]:
            continue
        circuit = (
            build_bulk_circuit(basis, noise=noise, code=code)
            if args.experiment == "bulk"
            else build_memory_circuit(basis, args.rounds, noise=noise, code=code)
        )

        def progress(event: dict[str, Any]) -> None:
            state["updated"] = utc_now()
            state["active"] = {"basis": basis, **event}
            atomic_json(checkpoint, state)
            fields = " ".join(
                f"{key}={value}" for key, value in event.items() if key != "timestamp"
            )
            print(
                f"checkpoint {event.get('timestamp', utc_now())} basis={basis} {fields}",
                flush=True,
            )

        estimate = estimate_four_fault_table(circuit) if args.max_faults == 4 else None
        print(
            f"checkpoint {utc_now()} basis={basis} phase=exact-through-{args.max_faults}-start "
            f"detectors={circuit.num_detectors} "
            f"pair_table_bytes={None if estimate is None else estimate['table_bytes']}",
            flush=True,
        )
        certify = certify_through_four if args.max_faults == 4 else certify_through_three
        result = certify(
            circuit,
            heartbeat_seconds=args.heartbeat_seconds,
            progress=progress,
        )
        if estimate is not None:
            result["table_estimate"] = estimate
        state["results"][basis] = result
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=exact-through-{args.max_faults}-finish "
            f"lower_bound={result['lower_bound']} distance={result['distance']} "
            f"effects={result['distinct_fault_effect_count']} "
            f"elapsed_seconds={result['elapsed_seconds']}",
            flush=True,
        )
    state.update(
        {
            "status": "complete",
            "completed": utc_now(),
            "updated": utc_now(),
            "active": None,
            "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3),
            f"all_bases_exclude_{args.max_faults}_fault_logicals": all(
                int(state["results"][basis]["lower_bound"]) > args.max_faults
                for basis in bases
            ),
        }
    )
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
