#!/usr/bin/env python3
"""Run resumable exact BB64 Clifford circuit fault-distance searches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    atomic_json,
    certify_fault_distance,
    utc_now,
)

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--experiment", choices=("bulk", "memory"), default="bulk")
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--fault-model", choices=("cnot", "no-idle", "full"), default="full")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--max-faults", type=int, default=6)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
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
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    checkpoint_path = args.checkpoint or args.output.with_suffix(".checkpoint.json")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    noise = noise_from_args(args.fault_model, args.probability)
    code = load_code_data(args.schedule)
    configuration = {
        "experiment": args.experiment,
        "bases": list(bases),
        "rounds": args.rounds,
        "fault_model": args.fault_model,
        "noise": noise.to_json(),
        "max_faults": args.max_faults,
        "heartbeat_seconds": args.heartbeat_seconds,
        "basis_npz": str(code.basis_path),
        "schedule": str(code.schedule_path),
    }
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit("resume checkpoint does not exist")
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("configuration") != configuration:
            raise SystemExit("checkpoint configuration does not match this command")
        state["status"] = "running"
        state["updated"] = utc_now()
        state["active"] = None
    else:
        state = {
            "schema_version": 1,
            "started": utc_now(),
            "updated": utc_now(),
            "status": "running",
            "configuration": configuration,
            "active": None,
            "results": {},
        }
    atomic_json(checkpoint_path, state)
    total_started = time.monotonic()
    for basis in bases:
        if basis in state["results"]:
            print(f"checkpoint {utc_now()} basis={basis} status=resume-skip", flush=True)
            continue
        circuit = (
            build_bulk_circuit(basis, noise=noise, code=code)
            if args.experiment == "bulk"
            else build_memory_circuit(basis, args.rounds, noise=noise, code=code)
        )

        def progress(event: dict[str, Any]) -> None:
            state["updated"] = utc_now()
            state["active"] = {"basis": basis, **event}
            atomic_json(checkpoint_path, state)
            fields = " ".join(
                f"{key}={value}" for key, value in event.items() if key != "timestamp"
            )
            print(f"checkpoint {event.get('timestamp', utc_now())} basis={basis} {fields}", flush=True)

        print(f"checkpoint {utc_now()} basis={basis} status=start", flush=True)
        result = certify_fault_distance(
            circuit,
            max_faults=args.max_faults,
            heartbeat_seconds=args.heartbeat_seconds,
            progress=progress,
        )
        state["results"][basis] = result
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint_path, state)
        print(
            f"checkpoint {utc_now()} basis={basis} status=done "
            f"distance={result['distance']} elapsed_seconds={result['elapsed_seconds']}",
            flush=True,
        )
    state["status"] = "complete"
    state["completed"] = utc_now()
    state["elapsed_seconds"] = round(time.monotonic() - total_started, 3)
    atomic_json(args.output, state)
    atomic_json(checkpoint_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
