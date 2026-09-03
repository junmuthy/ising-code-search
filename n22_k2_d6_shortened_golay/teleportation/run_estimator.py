#!/usr/bin/env python3
"""Run a resumable Fig. 10 teleportation-fidelity estimator experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import clifft
import numpy as np

from n22_k2_d6_shortened_golay.teleportation.circuit import (
    TeleportationNoise,
    build_estimator_circuit,
)
from n22_k2_d6_shortened_golay.teleportation.decoder import JointPauliDecoder
from n22_k2_d6_shortened_golay.tmr_postselection.circuit import (
    NoiseModel,
    ideal_tmr_acceptance,
)
from n22_k2_d6_shortened_golay.tmr_postselection.model import load_code_artifact
from n22_k2_d6_shortened_golay.tmr_postselection.partitions import (
    certify_partitions,
    load_partitions,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARTITIONS = PACKAGE_DIR.parent / "tmr_postselection" / "partitions_m3.json"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _parse_batch_size(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--partition-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--logical-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--preparation-probability", type=float, default=0.0)
    parser.add_argument("--teleportation-probability", type=float, default=0.0)
    parser.add_argument("--max-shots", type=int, default=100_000)
    parser.add_argument("--target-all-cancel", type=int, default=0)
    parser.add_argument("--checkpoint-shots", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--clifft-batch-size", type=_parse_batch_size, default="auto")
    parser.add_argument("--max-decoder-weight", type=int, default=5)
    parser.add_argument("--seed", type=int, default=220260)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-circuit", action="store_true")
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.theta < 0:
        parser.error("theta must be nonnegative")
    for name in ("preparation_probability", "teleportation_probability"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name.replace('_', '-')} must lie in [0,1]")
    if args.max_shots <= 0 or args.checkpoint_shots <= 0:
        parser.error("shot counts must be positive")
    if args.target_all_cancel < 0 or args.threads <= 0:
        parser.error("targets and threads must be nonnegative/positive")
    if args.max_decoder_weight < 0:
        parser.error("maximum decoder weight must be nonnegative")
    return args


def _initial_counts(logical_count: int) -> dict[str, Any]:
    return {
        "attempted": 0,
        "accepted": 0,
        "batches_completed": 0,
        "sign_patterns": {},
        "cancel_counts": [0] * logical_count,
        "logical_error_counts": [0] * logical_count,
        "all_cancel": 0,
        "all_cancel_any_error": 0,
        "all_cancel_error_patterns": {},
        "decoder_weight_histogram": {},
        "decoder_ambiguous": 0,
        "sampling_seconds": 0.0,
        "decoding_seconds": 0.0,
    }


def _merge_counter(target: dict[str, int], source: Counter[Any]) -> None:
    for key, count in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(count)


def _bit_patterns(bits: np.ndarray) -> Counter[str]:
    return Counter("".join(str(int(bit)) for bit in row) for row in bits)


def _standard_error(successes: int, trials: int) -> float | None:
    if not trials:
        return None
    rate = successes / trials
    return math.sqrt(rate * (1 - rate) / trials)


def _rate(successes: int, trials: int) -> float | None:
    return successes / trials if trials else None


def _estimates(counts: dict[str, Any], logical_count: int) -> dict[str, Any]:
    attempted = int(counts["attempted"])
    accepted = int(counts["accepted"])
    all_cancel = int(counts["all_cancel"])
    per_logical = []
    for logical in range(logical_count):
        cancel = int(counts["cancel_counts"][logical])
        errors = int(counts["logical_error_counts"][logical])
        per_logical.append(
            {
                "logical": logical,
                "cancel_count": cancel,
                "error_count": errors,
                "conditional_infidelity": _rate(errors, cancel),
                "standard_error": _standard_error(errors, cancel),
            }
        )

    pattern_counts = {
        str(key): int(value) for key, value in counts["all_cancel_error_patterns"].items()
    }
    marginals: list[float | None] = []
    if all_cancel:
        for logical in range(logical_count):
            marginals.append(
                sum(
                    count
                    for pattern, count in pattern_counts.items()
                    if pattern[logical] == "1"
                )
                / all_cancel
            )
    else:
        marginals = [None] * logical_count
    covariance = None
    correlation = None
    if logical_count == 2 and all_cancel:
        joint = pattern_counts.get("11", 0) / all_cancel
        covariance = joint - float(marginals[0]) * float(marginals[1])
        variance_product = (
            float(marginals[0])
            * (1 - float(marginals[0]))
            * float(marginals[1])
            * (1 - float(marginals[1]))
        )
        correlation = covariance / math.sqrt(variance_product) if variance_product else None

    any_errors = int(counts["all_cancel_any_error"])
    return {
        "preparation_acceptance": _rate(accepted, attempted),
        "preparation_acceptance_standard_error": _standard_error(accepted, attempted),
        "accepted_sign_pattern_frequencies": {
            pattern: count / accepted if accepted else None
            for pattern, count in sorted(counts["sign_patterns"].items())
        },
        "accepted_and_all_cancel_per_raw_attempt": _rate(all_cancel, attempted),
        "all_cancel_given_accepted": _rate(all_cancel, accepted),
        "per_logical": per_logical,
        "all_cancel_any_logical_infidelity": _rate(any_errors, all_cancel),
        "all_cancel_any_logical_infidelity_standard_error": _standard_error(
            any_errors, all_cancel
        ),
        "all_cancel_error_marginals": marginals,
        "all_cancel_error_covariance": covariance,
        "all_cancel_error_correlation": correlation,
    }


def _configuration(args: argparse.Namespace, circuit_hash: str) -> dict[str, Any]:
    return {
        "theta": args.theta,
        "partition_count": args.partition_count,
        "logical_count": args.logical_count,
        "preparation_probability": args.preparation_probability,
        "teleportation_probability": args.teleportation_probability,
        "threads": args.threads,
        "clifft_batch_size": args.clifft_batch_size,
        "max_decoder_weight": args.max_decoder_weight,
        "seed": args.seed,
        "circuit_sha256": circuit_hash,
    }


def _result_payload(
    *,
    args: argparse.Namespace,
    configuration: dict[str, Any],
    counts: dict[str, Any],
    circuit: Any,
    program: Any,
    compile_seconds: float,
    decoder: JointPauliDecoder,
    completed: bool,
) -> dict[str, Any]:
    return {
        "schema": "n22-teleportation-estimator-v1",
        "updated_utc": utc_now(),
        "completed": bool(completed),
        "configuration": configuration,
        "counts": counts,
        "estimates": _estimates(counts, args.logical_count),
        "theory": {
            "ideal_preparation_acceptance": ideal_tmr_acceptance(
                args.theta, args.partition_count
            )
            ** args.logical_count,
            "per_logical_cancel_probability_given_accepted": 0.5,
            "all_cancel_probability_given_accepted": 0.5**args.logical_count,
            "ideal_mean_rus_levels_for_two_logicals": 8 / 3
            if args.logical_count == 2
            else 2.0,
        },
        "circuit": {
            "num_qubits": program.num_qubits,
            "num_measurements": program.num_measurements,
            "num_detectors": program.num_detectors,
            "num_observables": program.num_observables,
            "peak_active_width": program.peak_active_width,
            "compile_seconds": compile_seconds,
            "operation_counts": circuit.operation_counts,
            "artifact": circuit.artifact,
        },
        "decoder": {
            "kind": "lazy exact joint minimum-symplectic-weight",
            "cached_syndromes": decoder.cached_syndromes,
            "max_search_weight": args.max_decoder_weight,
            "independent_z_checks": list(decoder.z_check_rows),
            "independent_x_checks": list(decoder.x_check_rows),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "clifft": package_version("clifft"),
        },
    }


def main() -> None:
    args = parse_args()
    code = load_code_artifact()
    certificate = certify_partitions(code, load_partitions(args.partitions))
    circuit = build_estimator_circuit(
        code=code,
        partition_certificate=certificate,
        theta=args.theta,
        partition_count=args.partition_count,
        logical_count=args.logical_count,
        preparation_noise=NoiseModel(probability=args.preparation_probability),
        teleportation_noise=TeleportationNoise(probability=args.teleportation_probability),
    )
    circuit_hash = hashlib.sha256(circuit.text.encode()).hexdigest()
    configuration = _configuration(args, circuit_hash)
    if args.resume and args.output.exists():
        record = json.loads(args.output.read_text(encoding="utf-8"))
        if record.get("configuration") != configuration:
            raise ValueError("resume configuration does not match the existing checkpoint")
        counts = record["counts"]
    else:
        counts = _initial_counts(args.logical_count)

    if args.save_circuit:
        circuit_path = args.output.with_suffix(".stim")
        if circuit_path.exists() and not args.resume:
            raise SystemExit(f"refusing to overwrite {circuit_path}")
        circuit_path.parent.mkdir(parents=True, exist_ok=True)
        circuit_path.write_text(circuit.text, encoding="utf-8")

    print(
        f"checkpoint {utc_now()} phase=compile status=start theta={args.theta:.12g} "
        f"M={args.partition_count} N={args.logical_count} "
        f"p_prep={args.preparation_probability:.6g} "
        f"p_teleport={args.teleportation_probability:.6g}",
        flush=True,
    )
    compile_started = time.monotonic()
    program = clifft.compile(circuit.text, postselection_mask=circuit.postselection_mask)
    compile_seconds = time.monotonic() - compile_started
    decoder = JointPauliDecoder(code, max_search_weight=args.max_decoder_weight)
    print(
        f"checkpoint {utc_now()} phase=compile status=done seconds={compile_seconds:.3f} "
        f"qubits={program.num_qubits} detectors={program.num_detectors} "
        f"observables={program.num_observables} peak_active_width={program.peak_active_width}",
        flush=True,
    )

    while counts["attempted"] < args.max_shots:
        if args.target_all_cancel and counts["all_cancel"] >= args.target_all_cancel:
            break
        shots = min(args.checkpoint_shots, args.max_shots - counts["attempted"])
        batch = int(counts["batches_completed"])
        sample_started = time.monotonic()
        sample = clifft.sample_survivors(
            program,
            shots=shots,
            seed=args.seed + batch,
            threads=args.threads,
            batch_size=args.clifft_batch_size,
            keep_records=True,
        )
        sample_seconds = time.monotonic() - sample_started
        accepted = int(sample.passed_shots)

        decode_started = time.monotonic()
        if accepted:
            terminal_z = sample.measurements[:, list(circuit.terminal_z_measurements)]
            terminal_x = sample.measurements[:, list(circuit.terminal_x_measurements)]
            decoded = decoder.decode(terminal_z, terminal_x)
            selected = list(circuit.selected_logicals)
            logical_z = decoded.logical_z[:, selected]
            logical_x = decoded.logical_x[:, selected]
            _merge_counter(counts["sign_patterns"], _bit_patterns(logical_z))
            for local in range(args.logical_count):
                cancel = logical_z[:, local] == 0
                counts["cancel_counts"][local] += int(np.sum(cancel))
                counts["logical_error_counts"][local] += int(
                    np.sum(logical_x[cancel, local])
                )
            all_cancel = ~np.any(logical_z, axis=1)
            error_bits = logical_x[all_cancel]
            counts["all_cancel"] += int(np.sum(all_cancel))
            counts["all_cancel_any_error"] += int(np.sum(np.any(error_bits, axis=1)))
            _merge_counter(
                counts["all_cancel_error_patterns"], _bit_patterns(error_bits)
            )
            _merge_counter(
                counts["decoder_weight_histogram"],
                Counter(map(int, decoded.correction_weight)),
            )
            counts["decoder_ambiguous"] += int(np.sum(decoded.ambiguous))
        decode_seconds = time.monotonic() - decode_started

        counts["attempted"] += int(sample.total_shots)
        counts["accepted"] += accepted
        counts["batches_completed"] += 1
        counts["sampling_seconds"] += sample_seconds
        counts["decoding_seconds"] += decode_seconds
        completed = counts["attempted"] >= args.max_shots or (
            args.target_all_cancel > 0
            and counts["all_cancel"] >= args.target_all_cancel
        )
        payload = _result_payload(
            args=args,
            configuration=configuration,
            counts=counts,
            circuit=circuit,
            program=program,
            compile_seconds=compile_seconds,
            decoder=decoder,
            completed=completed,
        )
        atomic_json(args.output, payload)
        estimates = payload["estimates"]
        sampling_rate = counts["attempted"] / counts["sampling_seconds"]
        print(
            f"checkpoint {utc_now()} phase=sample batch={counts['batches_completed']} "
            f"attempted={counts['attempted']}/{args.max_shots} accepted={counts['accepted']} "
            f"acceptance={estimates['preparation_acceptance']:.8g} "
            f"all_cancel={counts['all_cancel']} "
            f"any_error={counts['all_cancel_any_error']} "
            f"rate={sampling_rate:.1f}_attempts_per_second",
            flush=True,
        )

    final = _result_payload(
        args=args,
        configuration=configuration,
        counts=counts,
        circuit=circuit,
        program=program,
        compile_seconds=compile_seconds,
        decoder=decoder,
        completed=True,
    )
    atomic_json(args.output, final)
    print(json.dumps(final["estimates"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
