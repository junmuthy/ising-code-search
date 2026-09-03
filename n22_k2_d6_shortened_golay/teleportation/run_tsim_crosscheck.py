#!/usr/bin/env python3
"""Independent zero-noise tsim cross-check of the Fig. 10 estimator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tsim

from n22_k2_d6_shortened_golay.teleportation.circuit import build_estimator_circuit
from n22_k2_d6_shortened_golay.tmr_postselection.model import load_code_artifact
from n22_k2_d6_shortened_golay.tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = load_code_artifact()
    certificate = certify_partitions(
        code,
        load_partitions(PACKAGE_DIR.parent / "tmr_postselection" / "partitions_m3.json"),
    )
    bundle = build_estimator_circuit(
        code=code,
        partition_certificate=certificate,
        theta=args.theta,
        logical_count=2,
    )
    circuit = tsim.Circuit(bundle.text)
    sampler = circuit.compile_detector_sampler()
    mask = np.asarray(bundle.postselection_mask, dtype=np.bool_)
    detectors, observables = sampler.sample(
        shots=args.shots,
        separate_observables=True,
        postselection_mask=mask,
    )
    accepted = ~np.any(detectors[:, mask], axis=1)
    signs = observables[:, :2]
    outputs = observables[:, 2:]
    pattern_counts = {
        "".join(map(str, map(int, pattern))): int(count)
        for pattern, count in zip(
            *np.unique(signs[accepted], axis=0, return_counts=True), strict=True
        )
    }
    cancel_errors = []
    for logical in range(2):
        cancel = accepted & (signs[:, logical] == 0)
        cancel_errors.append(
            {
                "logical": logical,
                "cancel": int(np.sum(cancel)),
                "errors": int(np.sum(outputs[cancel, logical])),
            }
        )
    result = {
        "schema": "n22-tsim-teleportation-crosscheck-v1",
        "theta": args.theta,
        "shots": args.shots,
        "accepted": int(np.sum(accepted)),
        "acceptance": float(np.mean(accepted)),
        "sign_pattern_counts": pattern_counts,
        "cancel_branch_errors": cancel_errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
