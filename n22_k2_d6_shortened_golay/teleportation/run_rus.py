#!/usr/bin/env python3
"""Run a checkpointed classical RUS simulation from calibrated injection data."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Any

from n22_k2_d6_shortened_golay.teleportation.rus import (
    InjectionTable,
    expected_parallel_levels,
    simulate_rus,
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--logical-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-shots", type=int, default=100_000)
    parser.add_argument("--maximum-levels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=220261)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.theta <= 0 or args.shots <= 0 or args.checkpoint_shots <= 0:
        parser.error("theta and shot counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    table = InjectionTable.from_json(args.model)
    aggregate: dict[str, Any] | None = None
    completed = 0
    started = time.monotonic()
    batch = 0
    while completed < args.shots:
        local_shots = min(args.checkpoint_shots, args.shots - completed)
        local = simulate_rus(
            theta=args.theta,
            logical_count=args.logical_count,
            shots=local_shots,
            table=table,
            seed=args.seed + batch,
            maximum_levels=args.maximum_levels,
        )
        if aggregate is None:
            aggregate = local
        else:
            old_shots = int(aggregate["shots"])
            new_shots = old_shots + local_shots
            for key in (
                "mean_parallel_levels",
                "mean_raw_preparation_attempts",
                "mean_syndrome_cycles",
                "mean_teleportations",
                "any_logical_error_rate",
            ):
                aggregate[key] = (
                    float(aggregate[key]) * old_shots + float(local[key]) * local_shots
                ) / new_shots
            aggregate["per_logical_error_rates"] = [
                (old * old_shots + new * local_shots) / new_shots
                for old, new in zip(
                    aggregate["per_logical_error_rates"],
                    local["per_logical_error_rates"],
                    strict=True,
                )
            ]
            aggregate["final_error_pattern_counts"] = [
                old + new
                for old, new in zip(
                    aggregate["final_error_pattern_counts"],
                    local["final_error_pattern_counts"],
                    strict=True,
                )
            ]
            for histogram in ("round_histogram", "raw_attempt_histogram"):
                for key, value in local[histogram].items():
                    aggregate[histogram][key] = aggregate[histogram].get(key, 0) + value
            aggregate["truncated"] += local["truncated"]
            aggregate["shots"] = new_shots
        completed += local_shots
        batch += 1
        payload = {
            "schema": "n22-rus-v1",
            "updated_utc": utc_now(),
            "completed": completed == args.shots,
            "model": str(args.model),
            "result": aggregate,
            "theory": {
                "uncapped_ideal_mean_parallel_levels": expected_parallel_levels(
                    args.logical_count
                )
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.output, payload)
        print(
            f"checkpoint {utc_now()} batch={batch} shots={completed}/{args.shots} "
            f"mean_levels={aggregate['mean_parallel_levels']:.6g} "
            f"mean_raw_attempts={aggregate['mean_raw_preparation_attempts']:.6g} "
            f"any_error={aggregate['any_logical_error_rate']:.6g}",
            flush=True,
        )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
