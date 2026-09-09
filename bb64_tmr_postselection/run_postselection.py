#!/usr/bin/env python3
"""Run one resumable BB64 TMR postselection experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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

from .circuit import NoiseModel, build_circuit
from .model import load_code_artifact
from .partitions import certify_partitions, load_partitions


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARTITIONS = PACKAGE_DIR / "partitions_m3.json"


def utc_now() -> str:
    import datetime

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


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    estimate = successes / trials
    denominator = 1 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        estimate * (1 - estimate) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _parse_batch_order(value: str) -> tuple[int, int]:
    normalized = value.replace("B", "").replace("b", "").replace("-", "")
    if normalized == "01":
        return (0, 1)
    if normalized == "10":
        return (1, 0)
    raise argparse.ArgumentTypeError("batch order must be 0-1 or 1-0")


def _parse_clifft_batch_size(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("ClifT batch size must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ideal-projection", "scheduled-tmr-only", "full"),
        default="ideal-projection",
    )
    parser.add_argument(
        "--protocol",
        choices=("two-projection", "single-final-check"),
        default="two-projection",
    )
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--partition-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--logical-count", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--batch-order", type=_parse_batch_order, default=(0, 1))
    parser.add_argument(
        "--postselection-policy",
        choices=("tmr-x", "strict-xz"),
        default="strict-xz",
        help=(
            "tmr-x accepts on the TMR-sensitive X projection; strict-xz "
            "additionally rejects every nonzero raw Z syndrome"
        ),
    )
    parser.add_argument("--probability", type=float, default=0.0)
    parser.add_argument("--disable-preparation-noise", action="store_true")
    parser.add_argument("--disable-measurement-noise", action="store_true")
    parser.add_argument("--disable-cnot-noise", action="store_true")
    parser.add_argument("--disable-idle-noise", action="store_true")
    parser.add_argument("--disable-rotation-noise", action="store_true")
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--max-shots", type=int, default=10_000)
    parser.add_argument("--target-survivors", type=int, default=0)
    parser.add_argument("--checkpoint-shots", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--clifft-batch-size", type=_parse_clifft_batch_size, default="auto"
    )
    parser.add_argument("--seed", type=int, default=640808)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-circuit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_shots <= 0 or args.checkpoint_shots <= 0:
        parser.error("shot counts must be positive")
    if args.target_survivors < 0 or args.threads <= 0:
        parser.error("target survivors must be nonnegative and threads positive")
    if not 0 <= args.probability <= 1:
        parser.error("probability must lie in [0,1]")
    if args.mode == "ideal-projection" and args.probability:
        parser.error("ideal projection mode requires probability zero")
    return args


def _configuration(args: argparse.Namespace, circuit_hash: str) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "protocol": args.protocol,
        "theta": args.theta,
        "partition_count": args.partition_count,
        "logical_count": args.logical_count,
        "batch_order": list(args.batch_order),
        "postselection_policy": args.postselection_policy,
        "probability": args.probability,
        "noise": {
            "preparation": not args.disable_preparation_noise,
            "measurement": not args.disable_measurement_noise,
            "cnot": not args.disable_cnot_noise,
            "idle": not args.disable_idle_noise,
            "rotation": not args.disable_rotation_noise,
        },
        "diagnostic": args.diagnostic,
        "threads": args.threads,
        "clifft_batch_size": args.clifft_batch_size,
        "seed": args.seed,
        "circuit_sha256": circuit_hash,
    }


def _initial_counts() -> dict[str, Any]:
    return {
        "attempted": 0,
        "accepted": 0,
        "batches_completed": 0,
        "group_pass_counts": {},
        "syndrome_weight_histogram": {},
        "top_postselection_syndromes": {},
        "sampling_seconds": 0.0,
    }


def _load_resume(path: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        record = json.load(source)
    if record.get("configuration") != configuration:
        raise ValueError("resume configuration does not match the existing checkpoint")
    return record["counts"]


def _merge_counter(target: dict[str, int], source: Counter[Any]) -> None:
    for key, count in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(count)


def _batch_labels(detector_groups: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    labels = sorted(
        {
            group.rsplit("_", 1)[0]
            for group in detector_groups
            if group.startswith("post_batch_") and group.endswith(("_X", "_Z"))
        }
    )
    return {
        label: tuple(
            [*detector_groups.get(f"{label}_X", ()), *detector_groups.get(f"{label}_Z", ())]
        )
        for label in labels
    }


def _diagnostic_update(
    counts: dict[str, Any],
    detectors: np.ndarray,
    detector_groups: dict[str, tuple[int, ...]],
    postselection: tuple[int, ...],
) -> int:
    detectors = np.asarray(detectors, dtype=np.bool_)
    survivor_mask = ~np.any(detectors[:, list(postselection)], axis=1)
    accepted = int(np.sum(survivor_mask))
    for group, indices in {**detector_groups, **_batch_labels(detector_groups)}.items():
        if not indices:
            continue
        passed = int(np.sum(~np.any(detectors[:, list(indices)], axis=1)))
        counts["group_pass_counts"][group] = int(
            counts["group_pass_counts"].get(group, 0)
        ) + passed

    post = detectors[:, list(postselection)]
    weights = np.count_nonzero(post, axis=1)
    _merge_counter(counts["syndrome_weight_histogram"], Counter(map(int, weights)))
    packed = np.packbits(post, axis=1, bitorder="little")
    unique, multiplicities = np.unique(packed, axis=0, return_counts=True)
    syndrome_counter = Counter(
        {
            bytes(row).hex(): int(count)
            for row, count in zip(unique, multiplicities)
        }
    )
    _merge_counter(counts["top_postselection_syndromes"], syndrome_counter)
    return accepted


def _selected_postselection_detectors(bundle: Any, policy: str) -> tuple[int, ...]:
    if policy == "strict-xz":
        return tuple(bundle.postselection_detectors)
    if policy == "tmr-x":
        return tuple(
            detector
            for group, detectors in bundle.detector_groups.items()
            if group.startswith("post_") and group.endswith("_X")
            for detector in detectors
        )
    raise ValueError(f"unknown postselection policy: {policy}")


def _postselection_mask(bundle: Any, selected: tuple[int, ...]) -> list[int]:
    mask = [0] * sum(len(group) for group in bundle.detector_groups.values())
    for detector in selected:
        mask[detector] = 1
    return mask


def _result_payload(
    *,
    configuration: dict[str, Any],
    counts: dict[str, Any],
    bundle: Any,
    program: Any,
    selected_postselection: tuple[int, ...],
    compile_seconds: float,
    completed: bool,
) -> dict[str, Any]:
    attempted = int(counts["attempted"])
    accepted = int(counts["accepted"])
    acceptance = accepted / attempted if attempted else 0.0
    standard_error = (
        math.sqrt(acceptance * (1 - acceptance) / attempted) if attempted else None
    )
    interval = wilson_interval(accepted, attempted)
    ideal = bundle.ideal_acceptance
    top = sorted(
        counts["top_postselection_syndromes"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    summarized_counts = dict(counts)
    summarized_counts["top_postselection_syndromes"] = dict(top)
    group_acceptance = {
        group: passed / attempted
        for group, passed in counts["group_pass_counts"].items()
    } if attempted else {}
    ordered_batch_labels = [
        f"post_batch_{batch}"
        for batch in configuration["batch_order"]
        if f"post_batch_{batch}" in group_acceptance
    ]
    conditional_second = None
    if len(ordered_batch_labels) == 2:
        first_passed = counts["group_pass_counts"][ordered_batch_labels[0]]
        conditional_second = accepted / first_passed if first_passed else None
    return {
        "schema_version": 1,
        "timestamp": utc_now(),
        "completed": completed,
        "configuration": configuration,
        "circuit": {
            "num_qubits": program.num_qubits,
            "num_measurements": program.num_measurements,
            "num_detectors": program.num_detectors,
            "peak_active_width": program.peak_active_width,
            "operation_counts": bundle.operation_counts,
            "artifact": bundle.artifact,
            "partition_certificate": bundle.partition_certificate,
            "theta_star": bundle.theta_star,
            "theta_star_over_pi": bundle.theta_star / math.pi,
            "postselection_detector_count": len(selected_postselection),
            "postselection_detectors": list(selected_postselection),
            "detector_groups": {
                name: list(indices) for name, indices in bundle.detector_groups.items()
            },
        },
        "environment": {
            "python": platform.python_version(),
            "clifft": package_version("clifft"),
            "numpy": np.__version__,
        },
        "counts": summarized_counts,
        "estimates": {
            "block_acceptance": acceptance,
            "block_acceptance_standard_error": standard_error,
            "block_acceptance_wilson_95": list(interval),
            "ideal_tmr_acceptance": ideal,
            "noise_survival_ratio": acceptance / ideal if ideal else None,
            "accepted_candidate_resources_per_attempt": len(bundle.logicals) * acceptance,
            "detector_group_acceptance": group_acceptance,
            "second_projection_acceptance_given_first": conditional_second,
        },
        "timing": {
            "compile_seconds": compile_seconds,
            "sampling_seconds": counts["sampling_seconds"],
            "attempted_shots_per_second": (
                attempted / counts["sampling_seconds"] if counts["sampling_seconds"] else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --resume to continue")

    code = load_code_artifact()
    partitions = load_partitions(args.partitions)
    certificate = certify_partitions(code, partitions)
    noise = NoiseModel(
        probability=args.probability,
        preparation=not args.disable_preparation_noise,
        measurement=not args.disable_measurement_noise,
        cnot=not args.disable_cnot_noise,
        idle=not args.disable_idle_noise,
        rotation=not args.disable_rotation_noise,
    )
    bundle = build_circuit(
        code=code,
        partition_certificate=certificate,
        mode=args.mode,
        protocol=args.protocol,
        theta=args.theta,
        partition_count=args.partition_count,
        logical_count=args.logical_count,
        batch_order=args.batch_order,
        noise=noise,
    )
    circuit_hash = hashlib.sha256(bundle.text.encode()).hexdigest()
    configuration = _configuration(args, circuit_hash)
    counts = _load_resume(args.output, configuration) if args.resume else _initial_counts()
    selected_postselection = _selected_postselection_detectors(
        bundle, args.postselection_policy
    )

    if args.save_circuit:
        circuit_path = args.output.with_suffix(".stim")
        if circuit_path.exists() and not args.resume:
            raise SystemExit(f"refusing to overwrite {circuit_path}")
        circuit_path.parent.mkdir(parents=True, exist_ok=True)
        circuit_path.write_text(bundle.text, encoding="utf-8")

    postselection_mask = (
        None
        if args.diagnostic
        else _postselection_mask(bundle, selected_postselection)
    )
    print(
        f"checkpoint {utc_now()} phase=compile status=start mode={args.mode} "
        f"protocol={args.protocol} N={args.logical_count} M={args.partition_count} "
        f"theta={args.theta:.12g} p={args.probability:.6g}",
        flush=True,
    )
    compile_started = time.monotonic()
    program = clifft.compile(bundle.text, postselection_mask=postselection_mask)
    compile_seconds = time.monotonic() - compile_started
    print(
        f"checkpoint {utc_now()} phase=compile status=done seconds={compile_seconds:.3f} "
        f"qubits={program.num_qubits} detectors={program.num_detectors} "
        f"peak_active_width={program.peak_active_width}",
        flush=True,
    )

    while counts["attempted"] < args.max_shots:
        if args.target_survivors and counts["accepted"] >= args.target_survivors:
            break
        local_shots = min(args.checkpoint_shots, args.max_shots - counts["attempted"])
        started = time.monotonic()
        seed = args.seed + int(counts["batches_completed"])
        if args.diagnostic:
            sample = clifft.sample(
                program,
                shots=local_shots,
                seed=seed,
                threads=args.threads,
                batch_size=args.clifft_batch_size,
            )
            local_accepted = _diagnostic_update(
                counts,
                sample.detectors,
                bundle.detector_groups,
                selected_postselection,
            )
        else:
            sample = clifft.sample_survivors(
                program,
                shots=local_shots,
                seed=seed,
                threads=args.threads,
                batch_size=args.clifft_batch_size,
            )
            local_accepted = int(sample.passed_shots)
        elapsed = time.monotonic() - started
        counts["attempted"] += local_shots
        counts["accepted"] += local_accepted
        counts["batches_completed"] += 1
        counts["sampling_seconds"] += elapsed
        payload = _result_payload(
            configuration=configuration,
            counts=counts,
            bundle=bundle,
            program=program,
            selected_postselection=selected_postselection,
            compile_seconds=compile_seconds,
            completed=False,
        )
        atomic_json(args.output, payload)
        acceptance = counts["accepted"] / counts["attempted"]
        rate = counts["attempted"] / counts["sampling_seconds"]
        print(
            f"checkpoint {utc_now()} phase=sample batch={counts['batches_completed']} "
            f"attempted={counts['attempted']}/{args.max_shots} "
            f"accepted={counts['accepted']} acceptance={acceptance:.8g} "
            f"candidate_yield={args.logical_count * acceptance:.8g} "
            f"rate={rate:.3f}_attempts_per_second",
            flush=True,
        )

    completed = counts["attempted"] >= args.max_shots or (
        args.target_survivors > 0 and counts["accepted"] >= args.target_survivors
    )
    payload = _result_payload(
        configuration=configuration,
        counts=counts,
        bundle=bundle,
        program=program,
        selected_postselection=selected_postselection,
        compile_seconds=compile_seconds,
        completed=completed,
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload["estimates"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
