#!/usr/bin/env python3
"""Validate one schedule in CNOT-only bulk and full-noise memory circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .fault_distance import atomic_json, utc_now
from .model import DEFAULT_SCHEDULE_PATH, STATIC_DISTANCE, load_code_data
from .screening import certify_through_four, heuristic_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-rounds", type=int, default=5)
    parser.add_argument(
        "--memory-heuristic",
        action="store_true",
        help="run Stim's expensive hypergraph heuristic on full memory circuits",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = load_code_data(args.schedule)
    output: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "code": "[[28,4,5]]",
        "schedule": str(args.schedule),
        "experiments": {},
    }

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)

    if args.memory_rounds <= 0:
        raise ValueError("memory rounds must be positive")
    experiments = (
        ("bulk-cnot-X", build_bulk_circuit("X", noise=NoiseModel.cnot_only(), code=code)),
        ("bulk-cnot-Z", build_bulk_circuit("Z", noise=NoiseModel.cnot_only(), code=code)),
        (
            f"memory-r{args.memory_rounds}-full-X",
            build_memory_circuit("X", args.memory_rounds, noise=NoiseModel(), code=code),
        ),
        (
            f"memory-r{args.memory_rounds}-full-Z",
            build_memory_circuit("Z", args.memory_rounds, noise=NoiseModel(), code=code),
        ),
    )
    for name, circuit in experiments:
        print(json.dumps({"event": "experiment-start", "name": name}), flush=True)
        if name.startswith("memory-") and not args.memory_heuristic:
            heuristic = {
                "upper_bound": STATIC_DISTANCE,
                "elapsed_seconds": 0.0,
                "explained_errors": [],
                "reason": (
                    "five final-data measurement faults on a saved weight-five logical "
                    "give an explicit static-distance upper bound"
                ),
            }
        else:
            heuristic = heuristic_screen(
                circuit,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=lambda event, experiment=name: progress(
                    {"experiment": experiment, **event}
                ),
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
