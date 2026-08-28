#!/usr/bin/env python3
"""Reproduce the complete certificate for the best S3 single-grid code."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from gala_search.s3_ising import find_zero_syndrome_by_tanner_search
from gala_search.s3_single_grid import (
    analyze_certified_single_grid_code,
    build_certified_single_grid_code,
    certified_single_grid_candidate,
    certified_single_grid_support,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_DIR / "results" / "s3-single-grid-best.json",
    )
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = certified_single_grid_candidate()
    code = build_certified_single_grid_code()
    analysis = analyze_certified_single_grid_code()
    distance_searches = [
        find_zero_syndrome_by_tanner_search(
            code, weight=weight, max_nodes=args.max_nodes
        )
        for weight in range(1, 7)
    ]
    distance_is_seven = all(
        result["search_exhaustive"] and result["support"] is None
        for result in distance_searches
    )
    grid = analysis["logical_grid"]
    if not distance_is_seven or not all(
        (
            analysis["structure"]["accepted"],
            grid["pairwise_disjoint"],
            grid["pairing_is_identity"],
            grid["x_translation_is_C8_grid_shift"],
            grid["y_translation_is_C4_grid_shift"],
        )
    ):
        raise RuntimeError("the saved candidate failed recertification")

    record = {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "polynomials": candidate.polynomial_text,
        "generator_relation": "G_i = F_i.T",
        "parameters": {"n": code.num_qubits, "k": code.dimension, "d": 7},
        "logical_grid_seed_support": list(certified_single_grid_support()),
        "distance_searches": distance_searches,
        **analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **record["parameters"]}, sort_keys=True))


if __name__ == "__main__":
    main()
