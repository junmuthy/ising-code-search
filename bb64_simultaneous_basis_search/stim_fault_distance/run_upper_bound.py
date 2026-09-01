#!/usr/bin/env python3
"""Find and save a concrete BB64 circuit-fault upper-bound witness."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--max-symptoms", type=int, default=6)
    parser.add_argument("--max-edge-degree", type=int, default=17)
    parser.add_argument("--allow-symptom-increase", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
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
        raise SystemExit(f"refusing to overwrite {args.output}")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    code = load_code_data(args.schedule)
    noise = noise_from_args(args.fault_model, args.probability)
    state: dict[str, Any] = {
        "schema_version": 1,
        "started": utc_now(),
        "status": "running",
        "configuration": {
            "schedule": str(args.schedule.resolve()),
            "experiment": args.experiment,
            "rounds": args.rounds,
            "bases": list(bases),
            "fault_model": args.fault_model,
            "probability": args.probability,
            "max_symptoms": args.max_symptoms,
            "max_edge_degree": args.max_edge_degree,
            "allow_symptom_increase": args.allow_symptom_increase,
            "method": "Stim heuristic; concrete witness but not a lower-bound proof",
        },
        "results": {},
        "active": None,
    }
    checkpoint = args.output.with_suffix(".checkpoint.json")
    atomic_json(checkpoint, state)
    total_started = time.monotonic()
    for basis in bases:
        circuit = (
            build_bulk_circuit(basis, noise=noise, code=code)
            if args.experiment == "bulk"
            else build_memory_circuit(basis, args.rounds, noise=noise, code=code)
        )

        def search() -> list[Any]:
            return circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=args.max_symptoms,
                dont_explore_edges_with_degree_above=args.max_edge_degree,
                dont_explore_edges_increasing_symptom_degree=(
                    not args.allow_symptom_increase
                ),
                canonicalize_circuit_errors=True,
            )

        started = time.monotonic()
        print(f"checkpoint {utc_now()} basis={basis} phase=upper-bound-start", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search)
            while True:
                try:
                    errors = future.result(timeout=args.heartbeat_seconds)
                    break
                except concurrent.futures.TimeoutError:
                    elapsed = round(time.monotonic() - started, 3)
                    state["active"] = {
                        "basis": basis,
                        "phase": "upper-bound-search",
                        "elapsed_seconds": elapsed,
                    }
                    atomic_json(checkpoint, state)
                    print(
                        f"checkpoint {utc_now()} basis={basis} "
                        f"phase=upper-bound-heartbeat elapsed_seconds={elapsed}",
                        flush=True,
                    )
        result = {
            "upper_bound": len(errors) if errors else None,
            "explained_errors": [str(error) for error in errors],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        state["results"][basis] = result
        state["active"] = None
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=upper-bound-finish "
            f"upper_bound={result['upper_bound']} "
            f"elapsed_seconds={result['elapsed_seconds']}",
            flush=True,
        )
    state.update(
        {
            "status": "complete",
            "completed": utc_now(),
            "elapsed_seconds_this_invocation": round(time.monotonic() - total_started, 3),
            "active": None,
        }
    )
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
