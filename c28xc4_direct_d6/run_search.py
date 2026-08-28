#!/usr/bin/env python3
"""Run the direct ``C28 x C4`` ``[[112,16,d>=6]]`` ideal search."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

import numpy as np

from algebra import build_stabilizer_basis, fibre_logicals, quotient_annihilator_basis
from search import analyze_candidate, build_sparse_catalog, enumerate_ideal_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    print(json.dumps({"stage": "building_sparse_catalog"}), flush=True)
    catalog = build_sparse_catalog()
    candidates, raw_counts = enumerate_ideal_candidates(catalog)
    if args.maximum is not None:
        candidates = candidates[: args.maximum]
    catalog_summary = {
        "physical_group": "C28 x C4",
        "quotient_ring": "GF(8)[C4 x C4]",
        "raw_counts": raw_counts,
        "normalized_sparse_catalog": len(catalog),
        "unique_ideal_candidates": len(candidates),
        "maximum": args.maximum,
        "check_weight_ceiling": 12,
    }
    atomic_json(args.output / "catalog-summary.json", catalog_summary)
    print(json.dumps({"stage": "search", **catalog_summary}, sort_keys=True), flush=True)

    distance_histogram: Counter[int] = Counter()
    subgroup_histogram: Counter[int] = Counter()
    high_distance = 0
    weight12_generated = 0
    connected_weight12 = 0
    usable = 0
    saved = 0
    for index, candidate in enumerate(candidates, start=1):
        record = analyze_candidate(candidate, catalog)
        append_jsonl(args.output / "candidates.jsonl", record)
        distance_histogram[int(record["distance"])] += 1
        presentation = record["presentation"]
        weight12_generated += int(presentation["complete_weight_12_span"])
        connected_weight12 += int(presentation["connected_weight_12_presentation_exists"])
        if record["target_distance_six"]:
            high_distance += 1
            subgroup_histogram[int(presentation["all_sparse_displacement_subgroup_size"])] += 1
            append_jsonl(args.output / "distance-survivors.jsonl", record)
        if record["fully_usable"]:
            usable += 1
            append_jsonl(args.output / "usable-survivors.jsonl", record)
        if record["target_distance_six"] and saved < 100:
            annihilator = quotient_annihilator_basis(list(candidate.ideal_basis))
            stabilizer = build_stabilizer_basis(candidate.ideal_basis, annihilator)
            np.savez_compressed(
                args.output / f"{candidate.candidate_id}.npz",
                ideal_basis=candidate.ideal_basis,
                annihilator_basis=annihilator,
                stabilizer_basis=stabilizer,
                logical_x=fibre_logicals(),
                logical_z=fibre_logicals(),
            )
            saved += 1
        progress = {
            "completed": index,
            "total": len(candidates),
            "distance_histogram": dict(sorted(distance_histogram.items())),
            "distance_at_least_six": high_distance,
            "weight_12_generated": weight12_generated,
            "connected_weight_12": connected_weight12,
            "fully_usable": usable,
            "last_candidate_id": candidate.candidate_id,
            "last_distance": record["distance"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if index % args.checkpoint_every == 0 or index == len(candidates):
            atomic_json(args.output / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    summary = {
        "complete": True,
        **catalog_summary,
        "distance_histogram": dict(sorted(distance_histogram.items())),
        "distance_at_least_six": high_distance,
        "weight_12_generated": weight12_generated,
        "connected_weight_12": connected_weight12,
        "high_distance_displacement_subgroup_histogram": dict(sorted(subgroup_histogram.items())),
        "fully_usable": usable,
        "saved_high_distance_matrices": saved,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
