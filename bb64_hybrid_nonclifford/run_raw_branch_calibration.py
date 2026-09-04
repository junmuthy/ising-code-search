#!/usr/bin/env python3
"""Run a checkpointed full BB64 M=3 -> raw-branch -> M=1 calibration."""

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

from .circuit import class_id_for_labels
from .model import load_hybrid_model
from .raw_branch_calibration import build_raw_branch_calibration_circuit


PACKAGE_DIR = Path(__file__).resolve().parent


def _labels(value: str) -> tuple[int, ...]:
    if len(value) != 8 or any(character not in "0123" for character in value):
        raise argparse.ArgumentTypeError("branch labels must contain eight base-four digits")
    return tuple(map(int, value))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.0)
    parser.add_argument("--branches", type=_labels, nargs="+", default=((0,) * 8, (1, 0, 0, 0, 0, 0, 0, 0)))
    parser.add_argument("--final-syndrome-rounds", type=int, default=1)
    parser.add_argument("--max-shots", type=int, default=20_000)
    parser.add_argument("--checkpoint-shots", type=int, default=1_000)
    parser.add_argument("--target-survivors", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "results" / "raw_branch_calibration.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    model = load_hybrid_model()
    records: list[dict[str, object]] = []
    noise = NoiseModel(probability=args.probability)
    for branch_index, labels in enumerate(args.branches):
        class_id = class_id_for_labels(model, labels)
        bundle = build_raw_branch_calibration_circuit(
            model,
            theta=args.theta,
            syndrome_class_id=class_id,
            noise=noise,
            final_syndrome_rounds=args.final_syndrome_rounds,
        )
        print(f"checkpoint phase=compile branch={''.join(map(str, labels))} status=start", flush=True)
        started = time.monotonic()
        program = clifft.compile(
            bundle.text,
            postselection_mask=list(bundle.postselection_mask),
            expected_detectors=list(bundle.expected_detectors),
        )
        compile_seconds = time.monotonic() - started
        print(
            f"checkpoint phase=compile branch={''.join(map(str, labels))} status=done "
            f"seconds={compile_seconds:.3f} peak_active_width={program.peak_active_width}",
            flush=True,
        )
        attempted = passed = batches = any_errors = 0
        per_logical = np.zeros(8, dtype=np.int64)
        cooccurrence = np.zeros((8, 8), dtype=np.int64)
        sampling_seconds = 0.0
        while attempted < args.max_shots and passed < args.target_survivors:
            local = min(args.checkpoint_shots, args.max_shots - attempted)
            started = time.monotonic()
            sample = clifft.sample_survivors(
                program,
                shots=local,
                seed=args.seed + branch_index * 1_000_000 + batches,
                keep_records=True,
                threads=args.threads,
                batch_size=1,
            )
            sampling_seconds += time.monotonic() - started
            errors = np.asarray(sample.observables, dtype=np.uint8)
            attempted += local
            passed += int(sample.passed_shots)
            batches += 1
            if len(errors):
                per_logical += np.sum(errors, axis=0, dtype=np.int64)
                cooccurrence += errors.T.astype(np.int64) @ errors.astype(np.int64)
                any_errors += int(np.count_nonzero(np.any(errors, axis=1)))
            print(
                f"checkpoint phase=sample branch={''.join(map(str, labels))} batch={batches} "
                f"attempted={attempted}/{args.max_shots} passed={passed}/{args.target_survivors} "
                f"any_errors={any_errors} rate={attempted / sampling_seconds:.3f}/s",
                flush=True,
            )
        acceptance = passed / attempted
        any_rate = any_errors / passed if passed else math.nan
        covariance = np.full((8, 8), np.nan)
        if passed:
            covariance = cooccurrence / passed - np.outer(per_logical / passed, per_logical / passed)
        records.append(
            {
                "branch_labels": "".join(map(str, labels)),
                "syndrome_class_id": class_id,
                "alternative_count": int(sum(bundle.alternative_mask)),
                "attempted": attempted,
                "passed": passed,
                "acceptance": acceptance,
                "acceptance_standard_error": math.sqrt(acceptance * (1 - acceptance) / attempted),
                "expected_ideal_branch_probability": bundle.expected_ideal_branch_probability,
                "per_logical_infidelity": (per_logical / passed).tolist() if passed else [math.nan] * 8,
                "any_logical_infidelity": any_rate,
                "any_logical_infidelity_standard_error": (
                    math.sqrt(any_rate * (1 - any_rate) / passed) if passed else math.nan
                ),
                "error_covariance": covariance.tolist(),
                "compile_seconds": compile_seconds,
                "sampling_seconds": sampling_seconds,
                "attempts_per_second": attempted / sampling_seconds,
                "program": {
                    "num_qubits": program.num_qubits,
                    "num_detectors": program.num_detectors,
                    "peak_active_width": program.peak_active_width,
                },
                "operation_counts": bundle.operation_counts,
            }
        )
        _atomic_json(
            args.output,
            {
                "schema": "bb64-full-raw-branch-hybrid-calibration-v1",
                "completed": branch_index + 1 == len(args.branches),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "theta": args.theta,
                "probability": args.probability,
                "records": records,
            },
        )
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
