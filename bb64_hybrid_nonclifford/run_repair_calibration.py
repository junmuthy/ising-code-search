#!/usr/bin/env python3
"""Run checkpointed noisy ClifT calibration of scheduled BB64 M=1 repairs."""

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

from .model import load_hybrid_model
from .repair_calibration import build_repair_calibration_circuit


PACKAGE_DIR = Path(__file__).resolve().parent


def _mask(value: str) -> tuple[int, ...]:
    if len(value) != 8 or any(character not in "01" for character in value):
        raise argparse.ArgumentTypeError("repair masks must contain eight binary digits")
    return tuple(map(int, value))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument(
        "--masks",
        type=_mask,
        nargs="+",
        default=((1, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 0, 0)),
    )
    parser.add_argument("--final-syndrome-rounds", type=int, default=1)
    parser.add_argument("--max-shots", type=int, default=100_000)
    parser.add_argument("--checkpoint-shots", type=int, default=1_000)
    parser.add_argument("--target-survivors", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "results" / "repair_calibration.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    model = load_hybrid_model()
    noise = NoiseModel(probability=args.probability)
    records: list[dict[str, object]] = []
    for mask_index, mask in enumerate(args.masks):
        bundle = build_repair_calibration_circuit(
            model,
            theta=args.theta,
            alternative_mask=mask,
            noise=noise,
            final_syndrome_rounds=args.final_syndrome_rounds,
        )
        print(
            f"checkpoint phase=compile mask={''.join(map(str, mask))} status=start",
            flush=True,
        )
        started = time.monotonic()
        program = clifft.compile(bundle.text, postselection_mask=list(bundle.postselection_mask))
        compile_seconds = time.monotonic() - started
        print(
            f"checkpoint phase=compile mask={''.join(map(str, mask))} status=done "
            f"seconds={compile_seconds:.3f} peak_active_width={program.peak_active_width}",
            flush=True,
        )
        attempted = passed = batches = 0
        sampling_seconds = 0.0
        per_logical = np.zeros(8, dtype=np.int64)
        cooccurrence = np.zeros((8, 8), dtype=np.int64)
        any_errors = 0
        while attempted < args.max_shots and passed < args.target_survivors:
            local = min(args.checkpoint_shots, args.max_shots - attempted)
            started = time.monotonic()
            sample = clifft.sample_survivors(
                program,
                shots=local,
                seed=args.seed + mask_index * 1_000_000 + batches,
                keep_records=True,
                threads=args.threads,
                batch_size=1,
            )
            sampling_seconds += time.monotonic() - started
            errors = np.asarray(sample.observables, dtype=np.uint8)
            passed += int(sample.passed_shots)
            attempted += local
            batches += 1
            if len(errors):
                per_logical += np.sum(errors, axis=0, dtype=np.int64)
                cooccurrence += errors.T.astype(np.int64) @ errors.astype(np.int64)
                any_errors += int(np.count_nonzero(np.any(errors, axis=1)))
            print(
                f"checkpoint phase=sample mask={''.join(map(str, mask))} batch={batches} "
                f"attempted={attempted}/{args.max_shots} passed={passed}/{args.target_survivors} "
                f"any_errors={any_errors} rate={attempted / sampling_seconds:.3f}/s",
                flush=True,
            )
        acceptance = passed / attempted
        per_rates = (per_logical / passed).tolist() if passed else [math.nan] * 8
        any_rate = any_errors / passed if passed else math.nan
        covariance = np.full((8, 8), np.nan)
        if passed:
            covariance = cooccurrence / passed - np.outer(per_logical / passed, per_logical / passed)
        record = {
            "alternative_mask": "".join(map(str, mask)),
            "alternative_count": int(sum(mask)),
            "attempted": attempted,
            "passed": passed,
            "acceptance": acceptance,
            "acceptance_standard_error": math.sqrt(acceptance * (1 - acceptance) / attempted),
            "per_logical_infidelity": per_rates,
            "any_logical_infidelity": any_rate,
            "any_logical_infidelity_standard_error": (
                math.sqrt(any_rate * (1 - any_rate) / passed) if passed else math.nan
            ),
            "error_cooccurrence_counts": cooccurrence.tolist(),
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
        records.append(record)
        _atomic_json(
            args.output,
            {
                "schema": "bb64-noisy-m1-repair-calibration-v1",
                "completed": mask_index + 1 == len(args.masks),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "theta": args.theta,
                "probability": args.probability,
                "final_syndrome_rounds": args.final_syndrome_rounds,
                "records": records,
            },
        )
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
