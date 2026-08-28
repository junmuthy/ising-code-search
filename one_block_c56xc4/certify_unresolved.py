#!/usr/bin/env python3
"""Exactly certify records not resolved through weight five."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

from algebra import (
    build_stabilizer_basis,
    fibre_logicals,
    quotient_annihilator_basis,
)
from search import (
    build_sparse_catalog,
    certify_distance_ilp,
    enumerate_ideal_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    unresolved_ids = [
        record["candidate_id"] for record in records if record["distance"] is None
    ]
    if args.maximum is not None:
        unresolved_ids = unresolved_ids[: args.maximum]

    catalog = build_sparse_catalog()
    candidates, _raw = enumerate_ideal_candidates(catalog)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    started = time.perf_counter()
    histogram: Counter[int] = Counter()
    with args.output.open("x") as stream:
        for index, candidate_id in enumerate(unresolved_ids, start=1):
            candidate = candidate_map[candidate_id]
            annihilator = quotient_annihilator_basis(list(candidate.ideal_basis))
            stabilizer = build_stabilizer_basis(candidate.ideal_basis, annihilator)
            weight, support, status, elapsed = certify_distance_ilp(
                stabilizer, fibre_logicals()[0]
            )
            result = {
                "candidate_id": candidate_id,
                "certified_distance": weight,
                "support": support,
                "solver_status": status,
                "seconds": round(elapsed, 6),
            }
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            histogram[weight] += 1
            if index % args.checkpoint_every == 0 or index == len(unresolved_ids):
                progress = {
                    "completed": index,
                    "total": len(unresolved_ids),
                    "distance_histogram": dict(sorted(histogram.items())),
                    "last_candidate_id": candidate_id,
                    "last_distance": weight,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                atomic_json(args.output.with_suffix(".progress.json"), progress)
                print(json.dumps(progress, sort_keys=True), flush=True)

    atomic_json(
        args.output.with_suffix(".summary.json"),
        {
            "complete": True,
            "certified": len(unresolved_ids),
            "distance_histogram": dict(sorted(histogram.items())),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    )


if __name__ == "__main__":
    main()
