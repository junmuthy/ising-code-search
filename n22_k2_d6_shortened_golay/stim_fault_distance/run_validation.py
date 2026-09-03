#!/usr/bin/env python3
"""Validate one schedule in CNOT-only bulk and full-noise memory circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .fault_distance import atomic_json, utc_now
from .model import DEFAULT_SCHEDULE_PATH, load_code_data
from .screening import certify_through_four, heuristic_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = load_code_data(args.schedule)
    output: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "code": "[[22,2,6]]",
        "schedule": str(args.schedule),
        "experiments": {},
    }

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)

    experiments = (
        ("bulk-cnot-X", build_bulk_circuit("X", noise=NoiseModel.cnot_only(), code=code)),
        ("bulk-cnot-Z", build_bulk_circuit("Z", noise=NoiseModel.cnot_only(), code=code)),
        ("memory-r3-full-X", build_memory_circuit("X", 3, noise=NoiseModel(), code=code)),
        ("memory-r3-full-Z", build_memory_circuit("Z", 3, noise=NoiseModel(), code=code)),
    )
    for name, circuit in experiments:
        print(json.dumps({"event": "experiment-start", "name": name}), flush=True)
        heuristic = heuristic_screen(
            circuit,
            heartbeat_seconds=args.heartbeat_seconds,
            progress=lambda event, experiment=name: progress({"experiment": experiment, **event}),
        )
        exact = certify_through_four(
            circuit,
            heartbeat_seconds=args.heartbeat_seconds,
            progress=lambda event, experiment=name: progress({"experiment": experiment, **event}),
        )
        upper = heuristic["upper_bound"]
        lower = exact["lower_bound"]
        distance = int(upper) if upper is not None and int(upper) == int(lower) else None
        output["experiments"][name] = {
            "num_qubits": circuit.num_qubits,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "heuristic": heuristic,
            "exact_through_four": exact,
            "certified_fault_distance": distance,
        }
        atomic_json(args.output, output)
        print(
            json.dumps(
                {
                    "event": "experiment-finish",
                    "name": name,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "certified_fault_distance": distance,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

