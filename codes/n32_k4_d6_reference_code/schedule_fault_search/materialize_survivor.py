#!/usr/bin/env python3
"""Regenerate the complete gate-level schedule from the compact survivor record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codes.n32_k4_d6_reference_code.schedule_fault_search.data import code_for_schedule
from codes.n32_k4_d6_reference_code.schedule_fault_search.scheduler import (
    build_edge_orbits,
    build_schedule_record,
    matrix_from_supports,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import atomic_json


MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR.parent
DEFAULT_RECORD = MODULE_DIR / "survivor_a381a067d3e82750.json"
PRESENTATION = (
    REFERENCE_DIR / "schedule" / "all_weight8_translation_symmetric_v1" / "presentation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    compact = json.loads(args.record.read_text(encoding="utf-8"))
    presentation = json.loads(PRESENTATION.read_text(encoding="utf-8"))
    checks_x = matrix_from_supports(presentation["checks_x"])
    _orbits, edge_orbit = build_edge_orbits(
        checks_x, presentation["translation_action_on_x_checks"]
    )
    fold = dict(compact["fold"])
    fold["pairing"] = fold.pop("logical_pairing")
    schedule = build_schedule_record(
        checks_x,
        fold["permutation"],
        edge_orbit,
        compact["edge_orbit_colors_zero_based"],
        depth=int(compact["cnot_depth"]),
        fold_index=int(compact["fold_index"]),
        fold=fold,
    )
    if schedule["schedule_id"] != compact["schedule_id"]:
        raise RuntimeError("materialized schedule identifier does not match the compact record")
    code_for_schedule(schedule)
    atomic_json(args.output, schedule)
    print(json.dumps(schedule, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
