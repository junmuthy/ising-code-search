#!/usr/bin/env python3
"""Collect estimator JSON files into a calibrated RUS injection table."""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pattern_index(pattern: str) -> int:
    return sum(int(bit) << logical for logical, bit in enumerate(pattern))


def _point(record: dict[str, Any], source: Path) -> dict[str, Any]:
    configuration = record["configuration"]
    counts = record["counts"]
    estimates = record["estimates"]
    logical_count = int(configuration["logical_count"])
    all_cancel = int(counts["all_cancel"])
    if not all_cancel:
        raise ValueError(f"{source} has no all-cancel samples")
    pattern_counts = [0] * (1 << logical_count)
    for pattern, count in counts["all_cancel_error_patterns"].items():
        pattern_counts[_pattern_index(pattern)] += int(count)
    return {
        "theta": float(configuration["theta"]),
        "logical_count": logical_count,
        "partition_count": int(configuration["partition_count"]),
        "acceptance": float(estimates["preparation_acceptance"]),
        "error_pattern_probabilities": [count / all_cancel for count in pattern_counts],
        "syndrome_cycles_per_attempt": 2.0,
        "source": str(source),
        "all_cancel_samples": all_cancel,
        "preparation_probability": float(configuration["preparation_probability"]),
        "teleportation_probability": float(configuration["teleportation_probability"]),
    }


def _preferred_partition(theta: float) -> int:
    # Adaptive choice used in the STAR analysis: M=3 for small angles and M=1
    # once the RUS ladder reaches the large-angle regime.
    return 3 if theta <= math.pi / 8 + 1e-12 else 1


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}")
    candidates = []
    for path in args.inputs:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "n22-teleportation-estimator-v1":
            raise ValueError(f"unexpected schema in {path}")
        candidates.append(_point(record, path))

    selected: dict[tuple[float, int], dict[str, Any]] = {}
    for point in candidates:
        key = (round(point["theta"], 14), point["logical_count"])
        preferred = _preferred_partition(point["theta"])
        current = selected.get(key)
        if current is None:
            selected[key] = point
        elif point["partition_count"] == preferred and current["partition_count"] != preferred:
            selected[key] = point
        elif point["partition_count"] == current["partition_count"]:
            if point["all_cancel_samples"] > current["all_cancel_samples"]:
                selected[key] = point

    missing_preferred = [
        point
        for point in selected.values()
        if point["partition_count"] != _preferred_partition(point["theta"])
    ]
    if missing_preferred:
        details = ", ".join(
            f"theta={point['theta']:.12g},N={point['logical_count']}"
            for point in missing_preferred
        )
        raise ValueError(f"missing the preferred M calibration for: {details}")

    payload = {
        "schema": "n22-rus-injection-table-v1",
        "created_utc": utc_now(),
        "angle_tolerance": 1e-9,
        "partition_policy": "M=3 for theta<=pi/8; M=1 above pi/8",
        "points": sorted(
            selected.values(), key=lambda point: (point["theta"], point["logical_count"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(payload['points'])} calibrated points")


if __name__ == "__main__":
    main()
