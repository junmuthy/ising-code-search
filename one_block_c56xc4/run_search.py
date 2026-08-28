#!/usr/bin/env python3
"""Run the first exact-self-dual one-block ``C56 x C4`` ideal search."""

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

from algebra import build_stabilizer_basis, quotient_annihilator_basis
from search import (
    analyze_candidate,
    build_sparse_catalog,
    enumerate_ideal_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", type=int)
    parser.add_argument(
        "--family",
        choices=("all", "monomial_ideal", "binomial", "trinomial"),
        default="all",
    )
    parser.add_argument("--no-exact-distance", action="store_true")
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
    if args.family != "all":
        candidates = [
            candidate for candidate in candidates if args.family in candidate.families
        ]
    if args.maximum is not None:
        candidates = candidates[: args.maximum]
    atomic_json(
        args.output / "catalog-summary.json",
        {
            "raw_counts": raw_counts,
            "normalized_sparse_catalog": len(catalog),
            "unique_ideal_candidates_selected": len(candidates),
            "family_filter": args.family,
            "maximum": args.maximum,
        },
    )
    print(
        json.dumps(
            {
                "stage": "search",
                "candidates": len(candidates),
                "raw_counts": raw_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    output_jsonl = args.output / "candidates.jsonl"
    survivors_jsonl = args.output / "distance-survivors.jsonl"
    usable_survivors_jsonl = args.output / "usable-survivors.jsonl"
    structural_count = 0
    distance_histogram: Counter[int] = Counter()
    unresolved_count = 0
    usable_survivor_count = 0
    saved_matrix_count = 0
    for index, candidate in enumerate(candidates, start=1):
        record = analyze_candidate(
            candidate,
            catalog,
            exact_distance=not args.no_exact_distance,
        )
        append_jsonl(output_jsonl, record)
        structural_count += int(record["structurally_accepted"])
        if record["distance"] is None:
            unresolved_count += 1
        else:
            distance_histogram[int(record["distance"])] += 1
        if record["target_distance_six"]:
            append_jsonl(survivors_jsonl, record)
            if record["structurally_accepted"]:
                usable_survivor_count += 1
                append_jsonl(usable_survivors_jsonl, record)
            if saved_matrix_count < 20:
                annihilator = quotient_annihilator_basis(list(candidate.ideal_basis))
                stabilizer = build_stabilizer_basis(candidate.ideal_basis, annihilator)
                np.savez_compressed(
                    args.output / f"{candidate.candidate_id}.npz",
                    ideal_basis=candidate.ideal_basis,
                    annihilator_basis=annihilator,
                    stabilizer_basis=stabilizer,
                )
                saved_matrix_count += 1

        progress = {
            "completed": index,
            "total": len(candidates),
            "structurally_accepted": structural_count,
            "distance_histogram": dict(sorted(distance_histogram.items())),
            "unresolved": unresolved_count,
            "usable_distance_survivors": usable_survivor_count,
            "last_candidate_id": candidate.candidate_id,
            "last_distance": record["distance"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if index % args.checkpoint_every == 0 or index == len(candidates):
            atomic_json(args.output / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    summary = {
        "complete": True,
        "raw_counts": raw_counts,
        "normalized_sparse_catalog": len(catalog),
        "unique_ideal_candidates": len(candidates),
        "structurally_accepted": structural_count,
        "distance_histogram": dict(sorted(distance_histogram.items())),
        "unresolved": unresolved_count,
        "distance_at_least_six": sum(
            count for distance, count in distance_histogram.items() if distance >= 6
        ),
        "distance_seven": distance_histogram[7],
        "usable_distance_survivors": usable_survivor_count,
        "saved_high_distance_matrices": saved_matrix_count,
        "exact_distance_enabled": not args.no_exact_distance,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
