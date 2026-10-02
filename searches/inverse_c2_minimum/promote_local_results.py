#!/usr/bin/env python3
"""Promote local-refinement JSONL candidates into reusable seed files."""

from __future__ import annotations

import argparse
import json
import pathlib

from searches.inverse_c2_minimum.local_refinement import (
    analyze_generators,
    canonical_basis,
    recover_low_weight_generators,
    seed_from_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-distance", type=int, default=4)
    parser.add_argument("--maximum-check-weight", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        record for record in records if int(record.get("distance", 0)) >= args.minimum_distance
    ]
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    promoted = 0
    for record in selected:
        module_type = tuple(int(value) for value in record["module_type"])
        signature = (
            tuple(int(value) for value in record["fold"]["permutation"]),
            canonical_basis(tuple(int(value) for value in record["basis_x_masks"]), 16),
        )
        if signature in seen:
            continue
        seen.add(signature)
        generators = recover_low_weight_generators(
            record, module_type, args.maximum_check_weight
        )
        fold_index = int(record["fold"]["name"].rsplit("-", 1)[-1])
        wrapper = {
            "analysis": record,
            "fold_index": fold_index,
            "module_type": list(module_type),
            "promoted_generators_x_masks": list(generators),
        }
        seed = seed_from_record(wrapper)
        # Replace the arbitrary saved basis with the recovered orbit presentation.
        promoted_analysis = analyze_generators(generators, module_type, seed["fold"])
        if int(promoted_analysis["distance"]) != int(record["distance"]):
            raise AssertionError("promotion changed the code distance")
        wrapper["analysis"] = promoted_analysis
        path = args.output / f"seed-{promoted:03d}-fold-{fold_index:04d}-best.json"
        path.write_text(json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        promoted += 1
    summary = {
        "input_records": len(records),
        "eligible_records": len(selected),
        "promoted_unique_seeds": promoted,
        "minimum_distance": args.minimum_distance,
        "maximum_check_weight": args.maximum_check_weight,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
