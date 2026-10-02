#!/usr/bin/env python3
"""Validate the saved BB56 circuit-fault certificate and six-fault witness."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    schedule = json.loads(
        (ROOT / "optimal_depth8_schedule.json").read_text(encoding="utf-8")
    )
    certificate = json.loads(
        (ROOT / "circuit_fault_certificate.json").read_text(encoding="utf-8")
    )
    if certificate["schedule_id"] != schedule["schedule_id"]:
        raise AssertionError("schedule identifier mismatch")
    if certificate["conclusion"] != {
        "d_fault_x": 6,
        "d_fault_z": 6,
        "exact": True,
    }:
        raise AssertionError("unexpected certificate conclusion")
    for fault_count in ("fault_counts_1_to_3", "fault_count_4", "fault_count_5"):
        if certificate["lower_bound"][fault_count]["status"] != "unsat":
            raise AssertionError(f"lower-bound stage did not pass: {fault_count}")

    witness = certificate["upper_bound"]["x_basis_witness"]
    faults = witness["faults"]
    if len(faults) != certificate["upper_bound"]["fault_count"]:
        raise AssertionError("wrong witness cardinality")
    detectors: set[int] = set()
    logicals: set[int] = set()
    for fault in faults:
        for token in str(fault["effect"]).split():
            match = re.fullmatch(r"([DL])(\d+)", token)
            if match is None:
                raise AssertionError(f"invalid DEM token: {token}")
            target = detectors if match.group(1) == "D" else logicals
            value = int(match.group(2))
            target.symmetric_difference_update((value,))
    if detectors:
        raise AssertionError(f"witness has nonzero detector set: {sorted(detectors)}")
    if not logicals:
        raise AssertionError("witness has zero logical observable")
    if sorted(logicals) != witness["combined_logical_observables"]:
        raise AssertionError("saved combined logical set is inconsistent")
    print(
        json.dumps(
            {
                "verified": True,
                "schedule_id": schedule["schedule_id"],
                "fault_distance": 6,
                "witness_faults": len(faults),
                "witness_logicals": sorted(logicals),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
