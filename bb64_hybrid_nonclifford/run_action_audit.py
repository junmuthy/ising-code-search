#!/usr/bin/env python3
"""Exhaustively audit decoder-to-repair actions for all ideal BB64 branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table

from .actions import repair_action, verify_action_angles
from .decoder import NoiselessTableDecoder
from .model import load_hybrid_model


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_DIR / "manifests" / "action_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=math.pi / 32)
    parser.add_argument("--thresholds", type=int, nargs="+", default=(0, 1, 2, 3, 8))
    parser.add_argument("--checkpoint-classes", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.checkpoint_classes <= 0:
        raise SystemExit("checkpoint class count must be positive")
    thresholds = tuple(dict.fromkeys(args.thresholds))
    if not thresholds or any(threshold not in range(9) for threshold in thresholds):
        raise SystemExit("thresholds must lie in 0..8")

    model = load_hybrid_model()
    decoder = NoiselessTableDecoder(model)
    table = angle_table(args.theta)
    p_target = float(table["probability_target"])
    p_alternative = float(table["alternative"]["probability_per_pattern"])
    acceptance = {threshold: 0.0 for threshold in thresholds}
    repair_cost = {threshold: 0.0 for threshold in thresholds}
    accepted_classes = {threshold: 0 for threshold in thresholds}
    max_angle_error = {threshold: 0.0 for threshold in thresholds}

    classes = len(model.displayed_syndromes)
    for class_id, syndrome in enumerate(model.displayed_syndromes):
        decoded = decoder.decode(syndrome)
        if decoded.syndrome_class_id != class_id:
            raise RuntimeError(f"class round trip failed at {class_id}")
        bad = decoded.bad_count
        branch_probability = p_target ** (8 - bad) * p_alternative**bad
        # Cycling through all 256 logical-X frames across the class table tests
        # both command signs at every logical position without changing weights.
        frame_value = class_id & 0xFF
        frame_x = tuple((frame_value >> logical) & 1 for logical in range(8))
        for threshold in thresholds:
            action = repair_action(
                decoded,
                theta=args.theta,
                threshold=threshold,
                logical_x_frame=frame_x,
            )
            should_reset = bad > threshold
            if action.reset != should_reset:
                raise RuntimeError(
                    f"threshold action mismatch at class={class_id} threshold={threshold}"
                )
            if should_reset:
                if action.reset_reason != "repair_threshold" or action.repaired_logicals:
                    raise RuntimeError("invalid reset action")
                continue
            error = verify_action_angles(decoded, action, theta=args.theta)
            if not np.isfinite(error) or error > 1e-12:
                raise RuntimeError(
                    f"angle repair mismatch at class={class_id} threshold={threshold}: {error}"
                )
            acceptance[threshold] += branch_probability
            repair_cost[threshold] += branch_probability * bad
            accepted_classes[threshold] += 1
            max_angle_error[threshold] = max(max_angle_error[threshold], float(error))
        if (class_id + 1) % args.checkpoint_classes == 0 or class_id + 1 == classes:
            print(
                f"checkpoint phase=action-audit classes={class_id + 1}/{classes}",
                flush=True,
            )

    class_probabilities = (
        p_target ** (8 - model.recovery.alternative_count)
        * p_alternative ** model.recovery.alternative_count
    )
    prior_sum = math.fsum(map(float, class_probabilities))
    if abs(prior_sum - 1.0) > 1e-12:
        raise RuntimeError(f"branch prior does not normalize: {prior_sum}")
    policies = []
    for threshold in thresholds:
        retained = model.recovery.alternative_count <= threshold
        accepted = math.fsum(map(float, class_probabilities[retained]))
        stable_repair_cost = math.fsum(
            map(
                float,
                class_probabilities[retained]
                * model.recovery.alternative_count[retained],
            )
        )
        if not math.isclose(accepted, acceptance[threshold], rel_tol=0.0, abs_tol=2e-12):
            raise RuntimeError("stable policy sum disagrees with audited action sum")
        if not math.isclose(
            stable_repair_cost,
            repair_cost[threshold],
            rel_tol=0.0,
            abs_tol=2e-12,
        ):
            raise RuntimeError("stable repair-cost sum disagrees with audited action sum")
        policies.append(
            {
                "threshold": threshold,
                "accepted_classes": accepted_classes[threshold],
                "acceptance_probability": accepted,
                "reset_probability": max(0.0, 1.0 - accepted),
                "logical_resources_per_attempt": 8.0 * accepted,
                "repair_rotations_per_attempt": stable_repair_cost,
                "repair_rotations_per_accepted_block": stable_repair_cost / accepted,
                "maximum_angle_error": max_angle_error[threshold],
            }
        )
    payload = {
        "schema": "bb64-hybrid-action-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "theta": args.theta,
        "syndrome_classes": classes,
        "logical_x_frames_exercised": 256,
        "branch_prior_sum": prior_sum,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"checkpoint phase=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
