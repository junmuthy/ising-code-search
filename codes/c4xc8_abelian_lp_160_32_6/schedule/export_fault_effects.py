#!/usr/bin/env python3
"""Export distinct CNOT-fault signatures for the exact pair certificate."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from stim_circuit import build_guarded_bulk_circuit


def distinct_effects(circuit):
    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
        allow_gauge_detectors=False,
        approximate_disjoint_errors=True,
    )
    effects = {}
    for instruction in dem:
        if instruction.type != "error":
            continue
        detector = 0
        observable = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detector ^= 1 << int(target.val)
            elif target.is_logical_observable_id():
                observable ^= 1 << int(target.val)
            elif not target.is_separator():
                raise ValueError(f"unexpected detector-error target: {target}")
        if detector or observable:
            effects[(detector, observable)] = None
    return sorted(effects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=("X", "Z"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    circuit = build_guarded_bulk_circuit(args.basis)
    effects = distinct_effects(circuit)
    words = (circuit.num_detectors + 63) // 64
    with args.output.open("wb") as handle:
        handle.write(
            struct.pack(
                "<IIII",
                len(effects),
                circuit.num_detectors,
                circuit.num_observables,
                words,
            )
        )
        for detector, observable in effects:
            for word in range(words):
                handle.write(
                    struct.pack(
                        "<Q", (detector >> (64 * word)) & ((1 << 64) - 1)
                    )
                )
            handle.write(struct.pack("<Q", observable))
    print(json.dumps({
        "basis": args.basis,
        "effects": len(effects),
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "detector_words": words,
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
