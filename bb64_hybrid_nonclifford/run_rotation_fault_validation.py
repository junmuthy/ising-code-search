#!/usr/bin/env python3
"""Validate forced Pauli faults immediately before/after BB64 rotations in ClifT."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import clifft
import numpy as np

from bb64_tmr_postselection.circuit import NoiseModel

from .circuit_decoder.ledger import FaultMechanism
from .circuit_decoder.propagate import parse_circuit, propagate_faults
from .decoder import NoiselessTableDecoder
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit, extract_syndrome_histories


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "rotation_fault_validation_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--max-cases", type=int, default=12, help="use 144 for all 24x3x2 cases")
    parser.add_argument("--checkpoint-cases", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _forced_text(lines: list[str], line_index: int, gate: str, qubit: int, boundary: str) -> str:
    inserted = f"{gate} {qubit}"
    output = list(lines)
    output.insert(line_index if boundary == "before" else line_index + 1, inserted)
    return "\n".join(output) + "\n"


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.shots <= 0 or not 1 <= args.max_cases <= 144:
        raise SystemExit("shots must be positive and max-cases must lie in 1..144")
    model = load_hybrid_model()
    history = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(),
        syndrome_rounds=3,
    )
    parsed = parse_circuit(history.text)
    rotation_instructions = [item for item in parsed.instructions if item.name == "R_Z"]
    targets = [
        (instruction, target)
        for instruction in rotation_instructions
        for target in instruction.targets
    ]
    axes = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    cases = [
        (rotation, instruction, int(target), axis, boundary)
        for rotation, (instruction, target) in enumerate(targets)
        for axis in axes
        for boundary in ("before", "after")
    ][: args.max_cases]
    lines = history.text.rstrip().splitlines()
    invalid_corrected_histories = 0
    compile_seconds = 0.0
    sample_seconds = 0.0
    records = []
    table = NoiselessTableDecoder(model)
    for case_index, (rotation, instruction, qubit, axis, boundary) in enumerate(cases):
        x, z = axes[axis]
        injection_instruction = instruction.index - 1 if boundary == "before" else instruction.index
        fault = FaultMechanism(
            -1,
            -1,
            injection_instruction,
            "FORCED",
            (qubit,),
            axis,
            x << qubit,
            z << qubit,
            1.0,
            1.0,
        )
        signature = propagate_faults(parsed, (fault,))
        forced = _forced_text(lines, instruction.index, axis, qubit, boundary)
        started = time.monotonic()
        program = clifft.compile(forced)
        compile_seconds += time.monotonic() - started
        started = time.monotonic()
        sample = clifft.sample(
            program,
            shots=args.shots,
            seed=args.seed + case_index,
            threads=args.threads,
            batch_size=1,
        )
        sample_seconds += time.monotonic() - started
        detectors = np.asarray(sample.detectors, dtype=np.uint8).copy()
        predicted = np.fromiter(
            ((signature.detector_mask >> index) & 1 for index in range(parsed.num_detectors)),
            dtype=np.uint8,
        )
        detectors ^= predicted[None, :]
        x_history, z_history = extract_syndrome_histories(detectors, history)
        used = {
            index
            for group in (*history.x_detector_groups, *history.z_detector_groups)
            for index in group
        }
        initialization = [index for index in range(parsed.num_detectors) if index not in used]
        local_invalid = 0
        for shot in range(args.shots):
            valid = not np.any(detectors[shot, initialization])
            valid &= not np.any(z_history[shot])
            valid &= not np.any(x_history[shot] ^ x_history[shot, :1, :])
            if valid:
                try:
                    table.decode(x_history[shot, 0])
                except ValueError:
                    valid = False
            local_invalid += int(not valid)
        invalid_corrected_histories += local_invalid
        records.append(
            {
                "rotation": rotation,
                "axis": axis,
                "boundary": boundary,
                "qubit": qubit,
                "rotation_sign_mask": signature.rotation_sign_mask,
                "detector_weight": signature.detector_mask.bit_count(),
                "final_x_weight": signature.final_x_mask.bit_count(),
                "final_z_weight": signature.final_z_mask.bit_count(),
                "invalid_corrected_histories": local_invalid,
            }
        )
        completed = case_index + 1
        if completed % args.checkpoint_cases == 0 or completed == len(cases):
            print(
                f"checkpoint phase=rotation_fault cases={completed}/{len(cases)} "
                f"invalid={invalid_corrected_histories}",
                flush=True,
            )
            _atomic_json(
                args.output,
                {
                    "schema": "bb64-rotation-fault-validation-v1",
                    "completed": completed == len(cases),
                    "cases_completed": completed,
                    "cases_requested": len(cases),
                },
            )
    payload = {
        "schema": "bb64-rotation-fault-validation-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "shots_per_case": args.shots,
        "cases": len(cases),
        "total_shots": len(cases) * args.shots,
        "invalid_corrected_histories": invalid_corrected_histories,
        "compile_seconds": compile_seconds,
        "sampling_seconds": sample_seconds,
        "records": records,
        "interpretation": (
            "ClifT detector histories after a forced Pauli are XOR-corrected by the exact "
            "symbolic signature and must return to the ideal persistent TMR syndrome image"
        ),
    }
    _atomic_json(args.output, payload)
    print(f"checkpoint phase=complete output={args.output}", flush=True)
    if invalid_corrected_histories:
        raise SystemExit("forced rotation-fault validation failed")


if __name__ == "__main__":
    main()
