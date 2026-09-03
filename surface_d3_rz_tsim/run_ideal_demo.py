#!/usr/bin/env python3
"""Run a checkpointed ideal TMR preparation experiment with tsim."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

import jax
import numpy as np

from circuit import build_circuit, ideal_tmr_success
from surface_code import validate_code


EXPECTED_OBSERVABLE = {
    "X": math.cos,
    "Y": math.sin,
    "Z": lambda _theta: 0.0,
}


def _syndrome_key(row: np.ndarray) -> str:
    return "".join("1" if bit else "0" for bit in row)


def _angle_slug(theta: float) -> str:
    return (
        f"{theta:.12g}"
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def sample_observable(
    *,
    mode: str,
    theta: float,
    basis: str,
    shots: int,
    batch_size: int,
    seed: int,
    save_circuit: Path | None,
) -> dict[str, object]:
    bundle = build_circuit(mode, theta, basis)
    if save_circuit is not None:
        save_circuit.parent.mkdir(parents=True, exist_ok=True)
        save_circuit.write_text(str(bundle.circuit) + "\n", encoding="utf-8")

    print(
        f"[{mode} theta={theta:.12g} basis={basis}] compiling "
        f"({bundle.circuit.num_qubits} qubits, "
        f"{bundle.circuit.num_detectors} detectors)",
        flush=True,
    )
    compile_started = time.monotonic()
    sampler = bundle.circuit.compile_detector_sampler(seed=seed)
    compile_seconds = time.monotonic() - compile_started
    print(
        f"[{mode} theta={theta:.12g} basis={basis}] compiled in "
        f"{compile_seconds:.3f} s",
        flush=True,
    )

    accepted = 0
    accepted_sign_sum = 0
    sampled = 0
    syndrome_counts: Counter[str] = Counter()
    group_survivors = {name: 0 for name in bundle.detector_groups}
    sample_started = time.monotonic()
    batch_number = 0
    total_batches = math.ceil(shots / batch_size)
    while sampled < shots:
        local_shots = min(batch_size, shots - sampled)
        detectors, observables = sampler.sample(
            local_shots,
            batch_size=local_shots,
            separate_observables=True,
        )
        detectors = np.asarray(detectors, dtype=np.bool_)
        observables = np.asarray(observables, dtype=np.bool_)
        survivor_mask = ~np.any(detectors, axis=1)
        local_accepted = int(np.sum(survivor_mask))
        signs = 1 - 2 * observables[survivor_mask, 0].astype(np.int8)
        accepted += local_accepted
        accepted_sign_sum += int(np.sum(signs))
        sampled += local_shots
        batch_number += 1

        syndrome_counts.update(_syndrome_key(row) for row in detectors)
        for name, indices in bundle.detector_groups.items():
            local_group = detectors[:, list(indices)]
            group_survivors[name] += int(np.sum(~np.any(local_group, axis=1)))

        print(
            f"[{mode} theta={theta:.12g} basis={basis}] batch "
            f"{batch_number}/{total_batches}: {sampled}/{shots} shots, "
            f"accepted={accepted} ({accepted / sampled:.6f})",
            flush=True,
        )

    sample_seconds = time.monotonic() - sample_started
    if accepted == 0:
        raise RuntimeError("No shots survived the trivial-syndrome projection")
    expectation = accepted_sign_sum / accepted
    standard_error = math.sqrt(max(0.0, 1 - expectation**2) / accepted)
    expected = EXPECTED_OBSERVABLE[basis](theta)
    top_syndromes = [
        {"bits": bits, "count": count, "fraction": count / shots}
        for bits, count in syndrome_counts.most_common(10)
    ]
    return {
        "mode": mode,
        "basis": basis,
        "theta": theta,
        "theta_star": bundle.theta_star,
        "theta_star_over_pi": bundle.theta_star / math.pi,
        "partitions": bundle.partitions,
        "num_qubits": bundle.circuit.num_qubits,
        "num_detectors": bundle.circuit.num_detectors,
        "shots": shots,
        "accepted": accepted,
        "acceptance": accepted / shots,
        "ideal_acceptance": ideal_tmr_success(theta, bundle.partitions),
        "logical_expectation": expectation,
        "standard_error": standard_error,
        "ideal_logical_expectation": expected,
        "expectation_residual": expectation - expected,
        "group_acceptance": {
            name: count / shots for name, count in group_survivors.items()
        },
        "top_syndromes": top_syndromes,
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "seed": seed,
        "batch_size": batch_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("projection", "scheduled", "both"),
        default="projection",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=(0.0, 0.01, math.pi / 8),
        help="Target logical RZ angles in radians.",
    )
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=240902)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ideal_demo.json"),
    )
    parser.add_argument(
        "--save-circuits",
        action="store_true",
        help="Save each generated tsim circuit below results/circuits.",
    )
    args = parser.parse_args()
    if args.shots <= 0 or args.batch_size <= 0:
        parser.error("--shots and --batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    modes = ("projection", "scheduled") if args.mode == "both" else (args.mode,)
    validation = validate_code()
    print(f"code validation: {validation}", flush=True)
    print(f"JAX devices: {[str(device) for device in jax.devices()]}", flush=True)

    records: list[dict[str, object]] = []
    task_count = len(modes) * len(args.angles) * 3
    task = 0
    for mode in modes:
        for theta in args.angles:
            for basis_index, basis in enumerate("XYZ"):
                task += 1
                print(f"task {task}/{task_count}", flush=True)
                circuit_path = None
                if args.save_circuits:
                    circuit_path = Path("results/circuits") / (
                        f"{mode}-theta-{_angle_slug(theta)}-{basis}.tsim"
                    )
                records.append(
                    sample_observable(
                        mode=mode,
                        theta=theta,
                        basis=basis,
                        shots=args.shots,
                        batch_size=args.batch_size,
                        seed=args.seed + 100 * task + basis_index,
                        save_circuit=circuit_path,
                    )
                )

    payload = {
        "schema_version": 1,
        "description": "Ideal d=3 rotated-surface-code TMR preparation",
        "environment": {
            "python": platform.python_version(),
            "bloqade_tsim": version("bloqade-tsim"),
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "code_validation": validation,
        "configuration": {
            "modes": modes,
            "angles": args.angles,
            "shots_per_observable": args.shots,
            "batch_size": args.batch_size,
        },
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)

    print("\nsummary", flush=True)
    for record in records:
        print(
            f"{record['mode']:10s} theta={record['theta']:.8g} "
            f"{record['basis']}: acceptance={record['acceptance']:.6f} "
            f"(ideal {record['ideal_acceptance']:.6f}), "
            f"expectation={record['logical_expectation']:+.6f} "
            f"(ideal {record['ideal_logical_expectation']:+.6f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
