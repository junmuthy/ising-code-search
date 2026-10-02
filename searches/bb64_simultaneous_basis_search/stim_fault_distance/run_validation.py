#!/usr/bin/env python3
"""Validate noiseless and noisy BB64 Stim circuits in both memory bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    atomic_json,
    detector_error_model,
    utc_now,
)

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-circuits", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    code = load_code_data(args.schedule)
    results = []
    for basis in ("X", "Z"):
        print(f"checkpoint {utc_now()} phase=validation basis={basis} status=start", flush=True)
        ideal = build_memory_circuit(basis, args.rounds, code=code)
        detector_samples, observable_samples = ideal.compile_detector_sampler().sample(
            shots=args.shots, separate_observables=True
        )
        noisy = build_memory_circuit(
            basis, args.rounds, noise=NoiseModel(args.probability), code=code
        )
        bulk = build_bulk_circuit(basis, noise=NoiseModel(args.probability), code=code)
        noisy_dem = detector_error_model(noisy)
        bulk_dem = detector_error_model(bulk)
        result = {
            "basis": basis,
            "ideal": {
                "num_qubits": ideal.num_qubits,
                "num_detectors": ideal.num_detectors,
                "num_observables": ideal.num_observables,
                "num_measurements": ideal.num_measurements,
                "all_detector_samples_zero": not bool(np.any(detector_samples)),
                "all_observable_samples_zero": not bool(np.any(observable_samples)),
            },
            "noisy_memory": {
                "dem_errors": noisy_dem.num_errors,
                "num_detectors": noisy.num_detectors,
            },
            "noisy_bulk": {
                "dem_errors": bulk_dem.num_errors,
                "num_detectors": bulk.num_detectors,
            },
        }
        results.append(result)
        print(
            f"checkpoint {utc_now()} phase=validation basis={basis} status=done "
            f"detectors={ideal.num_detectors} memory_dem_errors={noisy_dem.num_errors}",
            flush=True,
        )
        if args.write_circuits:
            args.write_circuits.mkdir(parents=True, exist_ok=True)
            (args.write_circuits / f"memory_{basis.lower()}_r{args.rounds}.stim").write_text(
                str(noisy) + "\n", encoding="utf-8"
            )
            (args.write_circuits / f"bulk_{basis.lower()}.stim").write_text(
                str(bulk) + "\n", encoding="utf-8"
            )
    output = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "basis_npz": str(code.basis_path),
        "schedule": str(code.schedule_path),
        "rounds": args.rounds,
        "probability": args.probability,
        "results": results,
        "all_required_checks_pass": all(
            result["ideal"]["all_detector_samples_zero"]
            and result["ideal"]["all_observable_samples_zero"]
            and result["ideal"]["num_qubits"] == 128
            and result["ideal"]["num_observables"] == 8
            for result in results
        ),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["all_required_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
