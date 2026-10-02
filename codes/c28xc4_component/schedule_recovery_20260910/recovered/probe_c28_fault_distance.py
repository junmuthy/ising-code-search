#!/usr/bin/env python3
"""Build guarded-bulk Stim circuits and probe the depth-24 schedule's fault distance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stim

CODE_ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
SEARCH_ROOT = Path("/home/judah_unmuth/gala-code-search")
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SEARCH_ROOT))
from component_code import logical_fibres  # noqa: E402
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (  # noqa: E402
    _find_low_cardinality_witness,
    detector_error_model,
    explain_effect,
    extract_fault_effects,
)

NUM_DATA = 112
NUM_CHECKS = 48
X_START = NUM_DATA
Z_START = NUM_DATA + NUM_CHECKS
DATA = tuple(range(NUM_DATA))
X_ANCILLAS = tuple(range(X_START, Z_START))
Z_ANCILLAS = tuple(range(Z_START, Z_START + NUM_CHECKS))


def records(circuit, measurements):
    now = circuit.num_measurements
    return [stim.target_rec(int(index) - now) for index in measurements]


def syndrome_round(circuit, layers, noisy, probability):
    circuit.append("RX", X_ANCILLAS)
    circuit.append("R", Z_ANCILLAS)
    circuit.append("TICK")
    for layer in layers:
        pairs = []
        for gate in layer:
            if gate["type"] == "X":
                pairs.extend((X_START + int(gate["check"]), int(gate["data"])))
            else:
                pairs.extend((int(gate["data"]), Z_START + int(gate["check"])))
        circuit.append("CX", pairs)
        if noisy:
            circuit.append("DEPOLARIZE2", pairs, probability)
        circuit.append("TICK")
    start_x = circuit.num_measurements
    circuit.append("MX", X_ANCILLAS)
    start_z = circuit.num_measurements
    circuit.append("M", Z_ANCILLAS)
    circuit.append("TICK")
    return (
        tuple(range(start_x, start_x + NUM_CHECKS)),
        tuple(range(start_z, start_z + NUM_CHECKS)),
    )


def build_from_layers(basis, layers, noisy=True):
    check = np.zeros((NUM_CHECKS, NUM_DATA), dtype=np.uint8)
    # Reconstruct directly from the scheduled Tanner edges.
    for gate in (gate for layer in layers for gate in layer):
        check[int(gate["check"]), int(gate["data"])] = 1
    logicals = logical_fibres()
    circuit = stim.Circuit()
    circuit.append("RX" if basis == "X" else "R", DATA)
    circuit.append("TICK")
    previous = None
    for round_index in range(3):
        current = syndrome_round(circuit, layers, noisy and round_index == 1, 1e-3)
        if round_index == 0:
            selected = current[0 if basis == "X" else 1]
            for measurement in selected:
                circuit.append("DETECTOR", records(circuit, (measurement,)))
        else:
            for kind in range(2):
                for row in range(NUM_CHECKS):
                    circuit.append("DETECTOR", records(circuit, (current[kind][row], previous[kind][row])))
        previous = current
    start = circuit.num_measurements
    circuit.append("MX" if basis == "X" else "M", DATA)
    measured = tuple(range(start, start + NUM_DATA))
    kind = 0 if basis == "X" else 1
    for row in range(NUM_CHECKS):
        support = np.flatnonzero(check[row])
        circuit.append("DETECTOR", records(circuit, (previous[kind][row], *(measured[int(data)] for data in support))))
    for logical, row in enumerate(logicals):
        support = np.flatnonzero(row)
        circuit.append("OBSERVABLE_INCLUDE", records(circuit, (measured[int(data)] for data in support)), logical)
    return circuit


def build(basis, schedule_path, noisy=True):
    schedule = json.loads(schedule_path.read_text())
    return build_from_layers(basis, schedule["layers"], noisy=noisy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--schedule", type=Path, default=Path("/tmp/c28_clean_d24.json"))
    parser.add_argument("--max-direct", type=int, default=2)
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()
    ideal = build(args.basis, args.schedule, noisy=False)
    ideal.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    circuit = build(args.basis, args.schedule, noisy=True)
    dem = detector_error_model(circuit)
    effects = extract_fault_effects(dem)
    result = {
        "basis": args.basis,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "dem_errors": dem.num_errors,
        "distinct_fault_effects": len(effects),
        "direct": [],
    }
    print(json.dumps({**result, "event": "dem-ready"}), flush=True)
    for count in range(1, args.max_direct + 1):
        selected = _find_low_cardinality_witness(
            effects,
            count,
            heartbeat_seconds=30,
            callback=lambda event: print(json.dumps(event), flush=True),
        )
        entry = {"fault_count": count, "status": "sat" if selected else "unsat"}
        if selected:
            entry["effects"] = [
                {
                    "detector_weight": effects[index].detector_weight,
                    "observable_weight": effects[index].observable_weight,
                    "representative": explain_effect(circuit, effects[index]),
                }
                for index in selected
            ]
        result["direct"].append(entry)
        print(json.dumps({"event": "direct", **entry}), flush=True)
        if selected:
            break
    if args.heuristic:
        errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=8,
            dont_explore_edges_with_degree_above=12,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )
        result["heuristic_upper_bound"] = len(errors)
        result["heuristic_errors"] = [str(error) for error in errors]
        print(json.dumps({"event": "heuristic", "upper_bound": len(errors)}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
