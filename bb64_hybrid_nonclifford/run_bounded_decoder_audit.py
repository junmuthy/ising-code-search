#!/usr/bin/env python3
"""Audit the bounded latent-boundary decoder on labeled synthetic histories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table

from .bounded_decoder import BoundedLatentBoundaryDecoder
from .model import load_hybrid_model


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "manifests" / "bounded_decoder_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--shots-per-configuration", type=int, default=25)
    parser.add_argument("--checkpoint-shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample_class_ids(model, theta: float, shots: int, rng: np.random.Generator) -> np.ndarray:
    table = angle_table(theta)
    probabilities = np.array(
        [
            float(table["probability_target"]),
            *([float(table["alternative"]["probability_per_pattern"])] * 3),
        ]
    )
    labels = np.stack(
        [rng.choice(4, size=shots, p=probabilities) for _ in range(8)], axis=1
    ).astype(np.uint8)
    base4 = labels.astype(np.uint32) @ (4 ** np.arange(8, dtype=np.uint32))
    inverse = np.empty(65536, dtype=np.uint32)
    inverse[model.recovery.base4_branch_id] = np.arange(65536, dtype=np.uint32)
    return inverse[base4]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.shots_per_configuration <= 0 or args.checkpoint_shots <= 0:
        raise SystemExit("shot and checkpoint counts must be positive")
    model = load_hybrid_model()
    print("checkpoint phase=catalog status=start", flush=True)
    started = time.monotonic()
    decoder = BoundedLatentBoundaryDecoder(
        model,
        theta=args.theta,
        rounds=args.rounds,
        data_probability=args.probability,
        measurement_probability=args.probability,
        max_faults=2,
    )
    catalog_seconds = time.monotonic() - started
    print(
        f"checkpoint phase=catalog status=done seconds={catalog_seconds:.3f} "
        f"x_patterns={len(decoder.x_catalog)} z_patterns={len(decoder.z_catalog)}",
        flush=True,
    )

    configurations = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    rng = np.random.default_rng(args.seed)
    records = []
    for configuration_index, (x_weight, z_weight) in enumerate(configurations):
        class_ids = _sample_class_ids(
            model, args.theta, args.shots_per_configuration, rng
        )
        in_radius = ambiguous = class_correct = mask_correct = unambiguous_correct = 0
        seconds = 0.0
        for shot, class_id in enumerate(class_ids):
            x_history = np.repeat(
                model.displayed_syndromes[int(class_id)][None, :], args.rounds, axis=0
            )
            z_history = np.zeros((args.rounds, 32), dtype=np.uint8)
            x_indices = rng.choice(len(decoder.x_events), size=x_weight, replace=False)
            z_indices = rng.choice(len(decoder.z_events), size=z_weight, replace=False)
            x_history = decoder.apply_events(
                x_history, channel="x", event_indices=x_indices
            )
            z_history = decoder.apply_events(
                z_history, channel="z", event_indices=z_indices
            )
            started = time.monotonic()
            decoded = decoder.decode(x_history, z_history)
            seconds += time.monotonic() - started
            in_radius += int(decoded.in_radius)
            ambiguous += int(decoded.ambiguous)
            correct = (
                decoded.branch is not None
                and decoded.branch.syndrome_class_id == int(class_id)
            )
            class_correct += int(correct)
            if decoded.branch is not None:
                expected_mask = model.recovery.alternative_mask[int(class_id)]
                mask_correct += int(
                    np.array_equal(decoded.branch.alternative_mask, expected_mask)
                )
            unambiguous_correct += int(correct and not decoded.ambiguous)
            if (shot + 1) % args.checkpoint_shots == 0 or shot + 1 == len(class_ids):
                print(
                    f"checkpoint phase=audit configuration={configuration_index + 1}/"
                    f"{len(configurations)} faults=({x_weight},{z_weight}) "
                    f"shots={shot + 1}/{len(class_ids)}",
                    flush=True,
                )
        shots = len(class_ids)
        records.append(
            {
                "x_faults": x_weight,
                "z_faults": z_weight,
                "shots": shots,
                "in_radius_fraction": in_radius / shots,
                "ambiguous_fraction": ambiguous / shots,
                "class_accuracy": class_correct / shots,
                "alternative_mask_accuracy": mask_correct / shots,
                "unambiguous_correct_fraction": unambiguous_correct / shots,
                "decode_seconds": seconds,
                "histories_per_second": shots / seconds,
            }
        )
    payload = {
        "schema": "bb64-bounded-latent-boundary-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "rounds": args.rounds,
        "phenomenological_data_probability": args.probability,
        "phenomenological_measurement_probability": args.probability,
        "maximum_faults_per_channel": 2,
        "catalog_seconds": catalog_seconds,
        "x_event_count": len(decoder.x_events),
        "z_event_count": len(decoder.z_events),
        "x_unique_pattern_count": len(decoder.x_catalog),
        "z_unique_pattern_count": len(decoder.z_catalog),
        "records": records,
        "interpretation": (
            "exact maximum-score oracle for the bounded phenomenological event catalog; "
            "not a scheduled circuit-location decoder"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
