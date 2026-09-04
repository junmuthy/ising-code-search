#!/usr/bin/env python3
"""ClifT cross-check of all signed local TMR branch-angle corrections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import clifft
import numpy as np

from .toy_tmr import build_signed_toy_branch_circuit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "signed_toy_clifft.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    records = []
    total_observable_errors = 0
    total_final_detector_errors = 0
    maximum_acceptance_z_score = 0.0
    for sign_mask in range(8):
        for label in range(4):
            case = 4 * sign_mask + label
            bundle = build_signed_toy_branch_circuit(
                theta=args.theta, branch_label=label, sign_mask=sign_mask
            )
            program = clifft.compile(
                bundle.text, postselection_mask=list(bundle.postselection_mask)
            )
            sample = clifft.sample_survivors(
                program,
                shots=args.shots,
                seed=args.seed + case,
                keep_records=True,
            )
            observable_errors = int(np.count_nonzero(sample.observables))
            detector_errors = int(np.count_nonzero(sample.detectors[:, 2:]))
            total_observable_errors += observable_errors
            total_final_detector_errors += detector_errors
            acceptance = int(sample.passed_shots) / args.shots
            standard_error = math.sqrt(
                bundle.expected_probability * (1.0 - bundle.expected_probability) / args.shots
            )
            z_score = abs(acceptance - bundle.expected_probability) / standard_error
            maximum_acceptance_z_score = max(maximum_acceptance_z_score, z_score)
            records.append(
                {
                    "sign_mask": sign_mask,
                    "branch_label": label,
                    "shots": args.shots,
                    "passed": int(sample.passed_shots),
                    "acceptance": acceptance,
                    "acceptance_standard_error": standard_error,
                    "acceptance_z_score": z_score,
                    "expected_acceptance": bundle.expected_probability,
                    "observable_errors": observable_errors,
                    "final_detector_errors": detector_errors,
                }
            )
        print(f"checkpoint phase=signed_toy sign_masks={sign_mask + 1}/8", flush=True)
    payload = {
        "schema": "bb64-signed-local-tmr-crosscheck-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "shots_per_case": args.shots,
        "cases": 32,
        "observable_errors": total_observable_errors,
        "final_detector_errors": total_final_detector_errors,
        "maximum_acceptance_z_score": maximum_acceptance_z_score,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"checkpoint phase=complete output={args.output}", flush=True)
    if total_observable_errors or total_final_detector_errors:
        raise SystemExit("signed local TMR cross-check failed")


if __name__ == "__main__":
    main()
