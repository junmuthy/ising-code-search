#!/usr/bin/env python3
"""Run checkpointed exact-MAP repeated-measurement decoder pilots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table

from .decoder import MeasurementMapDecoder, sample_measurement_histories
from .model import load_hybrid_model


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "measurement_decoder_pilot.json"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_class_ids(model, theta: float, shots: int, seed: int) -> np.ndarray:
    table = angle_table(theta)
    probabilities = np.array(
        [
            float(table["probability_target"]),
            *([float(table["alternative"]["probability_per_pattern"])] * 3),
        ]
    )
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    labels = np.stack(
        [rng.choice(4, size=shots, p=probabilities) for _ in range(8)], axis=1
    ).astype(np.uint8)
    base4 = labels.astype(np.uint32) @ (4 ** np.arange(8, dtype=np.uint32))
    inverse = np.empty(65536, dtype=np.uint32)
    inverse[model.recovery.base4_branch_id] = np.arange(65536, dtype=np.uint32)
    return inverse[base4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--rounds", type=int, nargs="+", default=(1, 3, 5, 7))
    parser.add_argument(
        "--measurement-probabilities", type=float, nargs="+", default=(0.001, 0.01, 0.03)
    )
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--ambiguity-gap", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.shots <= 0:
        raise SystemExit("shots must be positive")
    model = load_hybrid_model()
    records: list[dict[str, object]] = []
    configurations = [
        (rounds, probability)
        for probability in args.measurement_probabilities
        for rounds in args.rounds
    ]
    print(
        f"checkpoint phase=setup configurations={len(configurations)} shots_per_configuration={args.shots}",
        flush=True,
    )
    for index, (rounds, probability) in enumerate(configurations):
        seed = args.seed + 1000 * index
        true_ids = _sample_class_ids(model, args.theta, args.shots, seed)
        histories = sample_measurement_histories(
            model,
            true_ids,
            rounds=rounds,
            measurement_probability=probability,
            seed=seed + 1,
        )
        decoder = MeasurementMapDecoder(
            model,
            theta=args.theta,
            measurement_probability=probability,
            ambiguity_gap=args.ambiguity_gap,
        )
        print(
            f"checkpoint phase=decode configuration={index + 1}/{len(configurations)} "
            f"rounds={rounds} measurement_probability={probability:.6g} status=start",
            flush=True,
        )
        started = time.monotonic()
        result = decoder.decode_batch(histories, batch_size=args.batch_size)
        elapsed = time.monotonic() - started
        true_mask = model.recovery.alternative_mask[true_ids]
        decoded_mask = model.recovery.alternative_mask[result.syndrome_class_id]
        class_correct = result.syndrome_class_id == true_ids
        mask_correct = np.all(decoded_mask == true_mask, axis=1)
        record = {
            "rounds": int(rounds),
            "measurement_probability": float(probability),
            "shots": int(args.shots),
            "class_accuracy": float(np.mean(class_correct)),
            "alternative_mask_accuracy": float(np.mean(mask_correct)),
            "ambiguous_fraction": float(np.mean(result.ambiguous)),
            "mean_posterior_probability": float(np.mean(result.posterior_probability)),
            "mean_likelihood_gap": float(np.mean(result.likelihood_gap)),
            "minimum_likelihood_gap": float(np.min(result.likelihood_gap)),
            "decode_seconds": float(elapsed),
            "histories_per_second": float(args.shots / elapsed),
            "seed": int(seed),
        }
        records.append(record)
        _atomic_json(
            args.output,
            {
                "schema": "bb64-measurement-decoder-pilot-v1",
                "completed": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "theta": args.theta,
                "records": records,
            },
        )
        print(
            f"checkpoint phase=decode configuration={index + 1}/{len(configurations)} "
            f"class_accuracy={record['class_accuracy']:.8g} "
            f"mask_accuracy={record['alternative_mask_accuracy']:.8g} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )
    payload = {
        "schema": "bb64-measurement-decoder-pilot-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "shots_per_configuration": args.shots,
        "ambiguity_gap": args.ambiguity_gap,
        "records": records,
    }
    _atomic_json(args.output, payload)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.with_name(csv_path.name + ".tmp").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    csv_path.with_name(csv_path.name + ".tmp").replace(csv_path)
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
