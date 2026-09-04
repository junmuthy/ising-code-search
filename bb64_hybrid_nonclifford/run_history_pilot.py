#!/usr/bin/env python3
"""Sample full BB64 M=3 syndrome histories and run the MAP reference decoder."""

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

from .decoder import MeasurementMapDecoder
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit, extract_syndrome_histories


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "syndrome_history_pilot.json"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--syndrome-rounds", type=int, default=3)
    parser.add_argument("--decoder-measurement-probability", type=float)
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--checkpoint-shots", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.shots <= 0 or args.checkpoint_shots <= 0:
        raise SystemExit("shot counts must be positive")
    model = load_hybrid_model()
    bundle = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(probability=args.probability),
        syndrome_rounds=args.syndrome_rounds,
    )
    decoder_probability = args.decoder_measurement_probability
    if decoder_probability is None:
        decoder_probability = max(args.probability, 1e-9)
    decoder = MeasurementMapDecoder(
        model,
        theta=args.theta,
        measurement_probability=decoder_probability,
    )

    print("checkpoint phase=compile status=start", flush=True)
    started = time.monotonic()
    program = clifft.compile(bundle.text)
    compile_seconds = time.monotonic() - started
    print(
        f"checkpoint phase=compile status=done seconds={compile_seconds:.3f} "
        f"peak_active_width={program.peak_active_width}",
        flush=True,
    )

    sampled = 0
    batches = 0
    sampling_seconds = 0.0
    decoding_seconds = 0.0
    image_matches = np.zeros(args.syndrome_rounds, dtype=np.int64)
    hamming_residual = np.zeros(args.syndrome_rounds, dtype=np.int64)
    z_weights = np.zeros(args.syndrome_rounds, dtype=np.int64)
    zero_z = np.zeros(args.syndrome_rounds, dtype=np.int64)
    bad_histogram = np.zeros(9, dtype=np.int64)
    ambiguous = 0
    posterior_sum = 0.0
    gap_sum = 0.0
    inconsistent = 0

    rows = np.asarray(model.recovery.independent_syndrome_rows, dtype=int)
    while sampled < args.shots:
        local = min(args.checkpoint_shots, args.shots - sampled)
        started = time.monotonic()
        sample = clifft.sample(
            program,
            shots=local,
            seed=args.seed + batches,
            threads=args.threads,
            batch_size=1,
        )
        sampling_seconds += time.monotonic() - started
        x_history, z_history = extract_syndrome_histories(sample.detectors, bundle)
        started = time.monotonic()
        decoded = decoder.decode_batch(x_history, batch_size=16)
        decoding_seconds += time.monotonic() - started

        decoded_patterns = model.displayed_syndromes[decoded.syndrome_class_id]
        hamming_residual += np.sum(
            x_history ^ decoded_patterns[:, None, :], axis=(0, 2), dtype=np.int64
        )
        for round_index in range(args.syndrome_rounds):
            class_ids = model.class_ids_from_coordinates(x_history[:, round_index, rows])
            reconstructed = model.displayed_syndromes[class_ids]
            image_matches[round_index] += int(
                np.count_nonzero(np.all(reconstructed == x_history[:, round_index], axis=1))
            )
        round_z_weights = np.sum(z_history, axis=2, dtype=np.int64)
        z_weights += np.sum(round_z_weights, axis=0, dtype=np.int64)
        zero_z += np.sum(round_z_weights == 0, axis=0, dtype=np.int64)
        bad_counts = model.recovery.alternative_count[decoded.syndrome_class_id]
        bad_histogram += np.bincount(bad_counts, minlength=9)
        ambiguous += int(np.count_nonzero(decoded.ambiguous))
        posterior_sum += float(np.sum(decoded.posterior_probability))
        gap_sum += float(np.sum(decoded.likelihood_gap))
        inconsistent += int(
            np.count_nonzero(np.any(x_history != x_history[:, :1, :], axis=(1, 2)))
        )
        sampled += local
        batches += 1
        partial = {
            "schema": "bb64-syndrome-history-pilot-v1",
            "completed": sampled == args.shots,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sampled": sampled,
            "requested_shots": args.shots,
            "physical_probability": args.probability,
            "syndrome_rounds": args.syndrome_rounds,
        }
        _atomic_json(args.output, partial)
        print(
            f"checkpoint phase=sample batch={batches} sampled={sampled}/{args.shots} "
            f"rate={sampled / sampling_seconds:.3f}/s ambiguous={ambiguous}",
            flush=True,
        )

    payload = {
        "schema": "bb64-syndrome-history-pilot-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "physical_probability": args.probability,
        "decoder_measurement_probability": decoder_probability,
        "syndrome_rounds": args.syndrome_rounds,
        "shots": args.shots,
        "raw_x_image_fraction_by_round": (image_matches / args.shots).tolist(),
        "mean_x_hamming_residual_to_decoded_class_by_round": (
            hamming_residual / args.shots
        ).tolist(),
        "mean_z_syndrome_weight_by_round": (z_weights / args.shots).tolist(),
        "zero_z_syndrome_fraction_by_round": (zero_z / args.shots).tolist(),
        "decoded_alternative_count_histogram": bad_histogram.tolist(),
        "ambiguous_fraction": ambiguous / args.shots,
        "mean_map_posterior": posterior_sum / args.shots,
        "mean_likelihood_gap": gap_sum / args.shots,
        "x_history_changed_fraction": inconsistent / args.shots,
        "compile_seconds": compile_seconds,
        "sampling_seconds": sampling_seconds,
        "decoding_seconds": decoding_seconds,
        "attempts_per_sampling_second": args.shots / sampling_seconds,
        "program": {
            "num_qubits": program.num_qubits,
            "num_detectors": program.num_detectors,
            "peak_active_width": program.peak_active_width,
        },
        "operation_counts": bundle.operation_counts,
        "interpretation": (
            "diagnostic retained histories; the persistent-syndrome MAP model is not a "
            "circuit-fault decoder and has no labeled true class when physical noise is active"
        ),
    }
    _atomic_json(args.output, payload)
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
