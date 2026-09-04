#!/usr/bin/env python3
"""Audit BB64 action-posterior decoding on labeled catalog histories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from bb64_tmr_postselection.circuit import NoiseModel

from .circuit_decoder.archive import load_fault_catalog
from .circuit_decoder.decoder import ScheduledActionDecoder
from .circuit_decoder.propagate import propagate_faults
from .model import load_hybrid_model
from .syndrome_history import build_syndrome_history_circuit


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = PACKAGE_DIR / "results" / "scheduled_fault_catalog_p1e3_pairs64.npz"
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "scheduled_decoder_audit_p1e3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--syndrome-rounds", type=int, default=3)
    parser.add_argument("--ideal-classes", type=int, default=32)
    parser.add_argument("--single-cases", type=int, default=64)
    parser.add_argument("--pair-linearity-cases", type=int, default=100)
    parser.add_argument("--minimum-posterior", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ideal_mask(history, syndrome: np.ndarray) -> int:
    value = 0
    for group in history.x_detector_groups:
        for check, detector in enumerate(group):
            value |= int(syndrome[check]) << detector
    return value


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    rng = np.random.default_rng(args.seed)
    model = load_hybrid_model()
    history = build_syndrome_history_circuit(
        model,
        theta=args.theta,
        noise=NoiseModel(probability=args.probability),
        syndrome_rounds=args.syndrome_rounds,
    )
    print("checkpoint phase=load_catalog status=start", flush=True)
    started = time.monotonic()
    catalog = load_fault_catalog(args.catalog, history)
    load_seconds = time.monotonic() - started
    print(f"checkpoint phase=load_catalog status=done seconds={load_seconds:.3f}", flush=True)
    modeled_decoder = ScheduledActionDecoder(
        model, catalog, theta=args.theta, minimum_posterior=0.0
    )
    conservative_decoder = ScheduledActionDecoder(
        model,
        catalog,
        theta=args.theta,
        minimum_posterior=args.minimum_posterior,
    )

    class_ids = rng.choice(len(model.displayed_syndromes), size=args.ideal_classes, replace=False)
    ideal_action_matches = 0
    ideal_conservative_accepts = 0
    modeled_posteriors: list[float] = []
    lower_bounds: list[float] = []
    for count, class_id in enumerate(class_ids, start=1):
        observed = _ideal_mask(history, model.displayed_syndromes[int(class_id)])
        result = modeled_decoder.decode_mask(observed)
        conservative = conservative_decoder.decode_mask(observed)
        expected, _probability = modeled_decoder._action(  # audited internal oracle
            int(class_id), catalog.no_fault_group.frame, 0
        )
        ideal_action_matches += int(result.action == expected)
        ideal_conservative_accepts += int(not conservative.reset)
        modeled_posteriors.append(result.modeled_posterior)
        lower_bounds.append(result.conservative_posterior_lower_bound)
        if count % 8 == 0 or count == len(class_ids):
            print(f"checkpoint phase=ideal_decodes completed={count}/{len(class_ids)}", flush=True)

    group_indices = rng.choice(
        len(catalog.single_groups), size=min(args.single_cases, len(catalog.single_groups)), replace=False
    )
    single_action_matches = 0
    single_in_catalog = 0
    for count, group_index in enumerate(group_indices, start=1):
        group = catalog.single_groups[int(group_index)]
        class_id = int(rng.integers(len(model.displayed_syndromes)))
        ideal = _ideal_mask(history, model.displayed_syndromes[class_id])
        observed = ideal ^ group.signature.detector_mask
        result = modeled_decoder.decode_mask(observed)
        expected, _probability = modeled_decoder._action(
            class_id, group.frame, group.signature.rotation_sign_mask
        )
        single_in_catalog += int(result.action is not None)
        single_action_matches += int(result.action == expected)
        if count % 16 == 0 or count == len(group_indices):
            print(f"checkpoint phase=single_decodes completed={count}/{len(group_indices)}", flush=True)

    linearity_failures = 0
    for count in range(args.pair_linearity_cases):
        left = catalog.ledger.mechanisms[int(rng.integers(len(catalog.ledger.mechanisms)))]
        while True:
            right = catalog.ledger.mechanisms[int(rng.integers(len(catalog.ledger.mechanisms)))]
            if right.location_id != left.location_id:
                break
        first = propagate_faults(catalog.circuit, (left,))
        second = propagate_faults(catalog.circuit, (right,))
        direct = propagate_faults(catalog.circuit, (left, right))
        expected = (
            first.detector_mask ^ second.detector_mask,
            first.final_x_mask ^ second.final_x_mask,
            first.final_z_mask ^ second.final_z_mask,
            first.rotation_sign_mask ^ second.rotation_sign_mask,
        )
        actual = (
            direct.detector_mask,
            direct.final_x_mask,
            direct.final_z_mask,
            direct.rotation_sign_mask,
        )
        linearity_failures += int(actual != expected)
        if (count + 1) % 25 == 0 or count + 1 == args.pair_linearity_cases:
            print(
                f"checkpoint phase=pair_linearity completed={count + 1}/{args.pair_linearity_cases}",
                flush=True,
            )

    payload = {
        "schema": "bb64-scheduled-action-decoder-audit-v1",
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "physical_probability": args.probability,
        "catalog": str(args.catalog),
        "catalog_counts": {
            "locations": len(catalog.ledger.locations),
            "mechanisms": len(catalog.ledger.mechanisms),
            "single_groups": len(catalog.single_groups),
            "selected_pair_groups": len(catalog.pair_groups),
        },
        "ideal_histories": {
            "tested": len(class_ids),
            "best_action_matches_no_fault_truth": ideal_action_matches,
            "mean_modeled_posterior": float(np.mean(modeled_posteriors)),
            "minimum_modeled_posterior": float(np.min(modeled_posteriors)),
            "mean_conservative_lower_bound": float(np.mean(lower_bounds)),
            "conservative_accepts": ideal_conservative_accepts,
        },
        "single_signature_histories": {
            "tested": len(group_indices),
            "returned_modeled_action": single_in_catalog,
            "best_action_matches_injected_truth": single_action_matches,
        },
        "pair_composition": {
            "tested": args.pair_linearity_cases,
            "linearity_failures": linearity_failures,
        },
        "posterior_policy": {
            "minimum_conservative_posterior": args.minimum_posterior,
            "omitted_probability_upper_bound": catalog.conservative_tail_probability,
            "meaning": "all omitted fault mass is adversarially assigned against the leading action",
        },
        "load_seconds": load_seconds,
    }
    _atomic_json(args.output, payload)
    print(f"checkpoint phase=complete output={args.output}", flush=True)
    if linearity_failures:
        raise SystemExit("pair-composition audit failed")


if __name__ == "__main__":
    main()
