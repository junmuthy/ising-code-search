#!/usr/bin/env python3
"""Reproduce the weight-12 nonabelian four-grid code certificate."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from gala_search.s3_many_copy_result import (
    analyze_certified_s3_four_grid_code,
    certify_s3_four_grid_distance,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_DIR
        / "results"
        / "s3-many-copy"
        / "recommended-certificate.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure = analyze_certified_s3_four_grid_code()
    distance = certify_s3_four_grid_distance(max_nodes=args.max_nodes)
    if (
        not structure["structure"]["accepted"]
        or not structure["four_grids_pairwise_disjoint"]
        or structure["combined_rank_mod_stabilizers"] != 128
        or structure["combined_zx_pairing_rank"] != 128
        or not structure["pairing_is_permutation"]
        or not distance["certified"]
    ):
        raise RuntimeError("the saved four-grid code failed recertification")
    record = {**structure, "distance": distance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                **record["parameters"],
                "pairing_is_permutation": record["pairing_is_permutation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
