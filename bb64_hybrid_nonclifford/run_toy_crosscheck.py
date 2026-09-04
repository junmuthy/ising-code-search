#!/usr/bin/env python3
"""Cross-check the local hybrid TMR primitive with ClifT or tsim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .toy_tmr import build_toy_branch_circuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("clifft", "tsim"), required=True)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for label in range(4):
        bundle = build_toy_branch_circuit(theta=args.theta, branch_label=label)
        mask = np.asarray(bundle.postselection_mask, dtype=np.bool_)
        if args.backend == "clifft":
            import clifft

            program = clifft.compile(bundle.text, postselection_mask=mask.tolist())
            sample = clifft.sample_survivors(
                program, shots=args.shots, seed=args.seed + label, keep_records=True
            )
            passed = int(sample.passed_shots)
            observable_errors = int(np.count_nonzero(sample.observables))
            final_detector_errors = int(np.count_nonzero(sample.detectors[:, 2:]))
        else:
            import tsim

            circuit = tsim.Circuit(bundle.text)
            sampler = circuit.compile_detector_sampler(seed=args.seed + label)
            detectors, observables = sampler.sample(
                shots=args.shots,
                separate_observables=True,
                postselection_mask=mask,
            )
            kept = ~np.any(detectors[:, mask], axis=1)
            passed = int(np.count_nonzero(kept))
            observable_errors = int(np.count_nonzero(observables[kept]))
            final_detector_errors = int(np.count_nonzero(detectors[kept, 2:]))
        record = {
            "branch_label": label,
            "shots": args.shots,
            "passed": passed,
            "acceptance": passed / args.shots,
            "expected_acceptance": bundle.expected_probability,
            "observable_errors": observable_errors,
            "final_detector_errors": final_detector_errors,
        }
        records.append(record)
        print(
            f"checkpoint backend={args.backend} branch={label} passed={passed}/{args.shots} "
            f"logical_errors={observable_errors} final_detector_errors={final_detector_errors}",
            flush=True,
        )
    result = {
        "schema": "bb64-local-tmr-crosscheck-v1",
        "backend": args.backend,
        "theta": args.theta,
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(record["observable_errors"] or record["final_detector_errors"] for record in records):
        raise SystemExit("local TMR cross-check failed")


if __name__ == "__main__":
    main()
