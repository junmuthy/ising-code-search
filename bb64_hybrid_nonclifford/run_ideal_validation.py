#!/usr/bin/env python3
"""Checkpointed ClifT or tsim validation of exact BB64 repair branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .circuit import build_ideal_branch_repair_circuit, class_id_for_labels
from .model import load_hybrid_model


PACKAGE_DIR = Path(__file__).resolve().parent


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _labels(value: str) -> tuple[int, ...]:
    if len(value) != 8 or any(character not in "0123" for character in value):
        raise argparse.ArgumentTypeError("branch labels must be eight base-four digits")
    return tuple(map(int, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("clifft", "tsim"), required=True)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--branches", type=_labels, nargs="+", default=((0,) * 8, (1, 0, 0, 0, 0, 0, 0, 0)))
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--checkpoint-shots", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample_clifft(bundle, shots: int, seed: int, threads: int):
    import clifft

    program = clifft.compile(bundle.text, postselection_mask=list(bundle.postselection_mask))
    sample = clifft.sample_survivors(
        program,
        shots=shots,
        seed=seed,
        keep_records=True,
        threads=threads,
        batch_size=1,
    )
    return (
        int(sample.passed_shots),
        int(np.count_nonzero(sample.observables)),
        int(np.count_nonzero(sample.detectors[:, bundle.initial_detector_count :])),
        {"num_qubits": program.num_qubits, "peak_active_width": program.peak_active_width},
    )


def _sample_tsim(bundle, shots: int, seed: int, _threads: int):
    import tsim

    circuit = tsim.Circuit(bundle.text)
    sampler = circuit.compile_detector_sampler(seed=seed)
    mask = np.asarray(bundle.postselection_mask, dtype=np.bool_)
    detectors, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
        postselection_mask=mask,
    )
    kept = ~np.any(detectors[:, mask], axis=1)
    return (
        int(np.count_nonzero(kept)),
        int(np.count_nonzero(observables[kept])),
        int(np.count_nonzero(detectors[kept, bundle.initial_detector_count :])),
        {"num_qubits": circuit.num_qubits},
    )


def main() -> None:
    args = parse_args()
    output = args.output or PACKAGE_DIR / "results" / f"ideal_validation_{args.backend}.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {output}")
    if args.shots <= 0 or args.checkpoint_shots <= 0:
        raise SystemExit("shot counts must be positive")
    model = load_hybrid_model()
    sampler = _sample_clifft if args.backend == "clifft" else _sample_tsim
    records: list[dict[str, object]] = []
    for branch_index, labels in enumerate(args.branches):
        class_id = class_id_for_labels(model, labels)
        bundle = build_ideal_branch_repair_circuit(
            model,
            theta=args.theta,
            syndrome_class_id=class_id,
        )
        attempted = passed = observable_errors = final_detector_errors = 0
        seconds = 0.0
        metadata: dict[str, object] = {}
        while attempted < args.shots:
            local = min(args.checkpoint_shots, args.shots - attempted)
            started = time.monotonic()
            local_passed, local_observable_errors, local_detector_errors, metadata = sampler(
                bundle, local, args.seed + branch_index * 100000 + attempted, args.threads
            )
            seconds += time.monotonic() - started
            attempted += local
            passed += local_passed
            observable_errors += local_observable_errors
            final_detector_errors += local_detector_errors
            print(
                f"checkpoint backend={args.backend} branch={''.join(map(str, labels))} "
                f"attempted={attempted}/{args.shots} passed={passed} "
                f"logical_errors={observable_errors} final_detector_errors={final_detector_errors}",
                flush=True,
            )
        acceptance = passed / attempted
        record = {
            "branch_labels": "".join(map(str, labels)),
            "syndrome_class_id": class_id,
            "alternative_count": int(np.count_nonzero(bundle.alternative_mask)),
            "attempted": attempted,
            "passed": passed,
            "acceptance": acceptance,
            "acceptance_standard_error": math.sqrt(acceptance * (1 - acceptance) / attempted),
            "expected_acceptance": bundle.expected_branch_probability,
            "observable_errors": observable_errors,
            "final_detector_errors": final_detector_errors,
            "sampling_seconds": seconds,
            "attempts_per_second": attempted / seconds,
            "simulator": metadata,
            "operation_counts": bundle.operation_counts,
        }
        records.append(record)
        _atomic_json(
            output,
            {
                "schema": "bb64-ideal-branch-repair-validation-v1",
                "completed": branch_index + 1 == len(args.branches),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "backend": args.backend,
                "theta": args.theta,
                "records": records,
            },
        )
    if any(record["observable_errors"] or record["final_detector_errors"] for record in records):
        raise SystemExit("ideal repair validation found an error")
    print(f"checkpoint phase=complete output={output}", flush=True)


if __name__ == "__main__":
    main()
