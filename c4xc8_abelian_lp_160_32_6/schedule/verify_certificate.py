#!/usr/bin/env python3
"""Verify the saved LP160 schedule and circuit-fault certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import build_and_verify
from stim_circuit import build_guarded_bulk_circuit, last_cnot_injections


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def injected_witness(basis: str, record: dict) -> dict:
    support = [int(value) for value in record["data_support"]]
    pauli = str(record["data_error"])
    injections = last_cnot_injections(support, pauli)
    circuit = build_guarded_bulk_circuit(
        basis, injected_middle_errors=injections
    )
    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
        allow_gauge_detectors=False,
    )
    detectors: set[int] = set()
    observables: set[int] = set()
    faults = 0
    for instruction in dem:
        if instruction.type != "error":
            continue
        faults += 1
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update((int(target.val),))
            elif target.is_logical_observable_id():
                observables.symmetric_difference_update((int(target.val),))
            elif not target.is_separator():
                raise AssertionError(f"unexpected DEM target: {target}")
    if faults != 6 or detectors or not observables:
        raise AssertionError("invalid six-fault circuit witness")
    if sorted(observables) != record["combined_logical_observables"]:
        raise AssertionError("logical-observable set differs from certificate")
    return {
        "faults": faults,
        "detectors": sorted(detectors),
        "observables": sorted(observables),
        "injections_by_zero_based_layer": {
            str(layer): [[pauli, data] for pauli, data in entries]
            for layer, entries in sorted(injections.items())
        },
    }


def full_pair_check() -> dict:
    with tempfile.TemporaryDirectory(prefix="lp160-pair-") as temporary:
        temporary = Path(temporary)
        executable = temporary / "pair_certify"
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++20",
                "-march=native",
                str(ROOT / "pair_certify.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        result = {}
        for basis in ("X", "Z"):
            effects = temporary / f"effects_{basis.lower()}.bin"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "export_fault_effects.py"),
                    "--basis",
                    basis,
                    "--output",
                    str(effects),
                ],
                check=True,
            )
            completed = subprocess.run(
                [str(executable), str(effects)],
                check=True,
                capture_output=True,
                text=True,
            )
            result[basis] = json.loads(completed.stdout.strip().splitlines()[-1])
            if not result[basis].get("no_faults_at_most_4"):
                raise AssertionError(f"pair certificate failed in {basis} basis")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pair", action="store_true")
    args = parser.parse_args()
    schedule_path = ROOT / "optimal_depth12_schedule.json"
    certificate = json.loads((ROOT / "circuit_fault_certificate.json").read_text())
    schedule = json.loads(schedule_path.read_text())
    expected = build_and_verify.materialize()
    if schedule != expected:
        raise AssertionError("schedule does not match canonical reconstruction")
    if schedule["schedule_id"] != certificate["schedule_id"]:
        raise AssertionError("schedule identifier mismatch")
    if sha256(schedule_path) != certificate["schedule_sha256"]:
        raise AssertionError("schedule SHA-256 mismatch")
    if certificate["conclusion"]["requested_lower_bound_passes"] is not True:
        raise AssertionError("certificate does not assert the requested lower bound")
    upper = certificate["upper_bound"]
    witnesses = {
        "X": injected_witness("X", upper["x_memory_basis"]),
        "Z": injected_witness("Z", upper["z_memory_basis"]),
    }
    result = {
        "verified": True,
        "schedule_id": schedule["schedule_id"],
        "depth": schedule["cnot_depth"],
        "gates_per_layer": schedule["gates_per_layer"],
        "fault_distance_interval_x": [5, 6],
        "fault_distance_interval_z": [5, 6],
        "six_fault_witnesses": witnesses,
    }
    if args.full_pair:
        result["full_pair_recheck"] = full_pair_check()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
