#!/usr/bin/env python3
"""Print or save one generated surface-code TMR circuit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit import build_circuit
from surface_code import (
    ANCILLA_QUBITS,
    DATA_QUBITS,
    LOGICAL_X,
    LOGICAL_Y_AXES,
    LOGICAL_Z,
    QUBIT_COORDS,
    X_CHECKS,
    Z_CHECKS,
    validate_code,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("projection", "scheduled"), default="scheduled")
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--basis", choices=tuple("XYZ"), default="X")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bundle = build_circuit(args.mode, args.theta, args.basis)
    metadata = {
        "validation": validate_code(),
        "data_qubits": DATA_QUBITS,
        "ancilla_qubits": ANCILLA_QUBITS,
        "coordinates": QUBIT_COORDS,
        "x_checks": X_CHECKS,
        "z_checks": Z_CHECKS,
        "logical_x": LOGICAL_X,
        "logical_y_axes": LOGICAL_Y_AXES,
        "logical_z": LOGICAL_Z,
        "mode": bundle.mode,
        "basis": bundle.basis,
        "theta": bundle.theta,
        "theta_star": bundle.theta_star,
        "detector_groups": bundle.detector_groups,
        "num_qubits": bundle.circuit.num_qubits,
        "num_detectors": bundle.circuit.num_detectors,
    }
    circuit_text = str(bundle.circuit) + "\n"
    print(json.dumps(metadata, indent=2))
    print("\n--- circuit ---")
    print(circuit_text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(circuit_text, encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
