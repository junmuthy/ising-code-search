#!/usr/bin/env python3
"""Revisit saved S3 candidates using joint rather than diagonal ZX pairing."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from typing import Any

from gala_search.s3_fold_analysis import (
    internal_affine_permutation,
    structured_physical_permutation,
)
from gala_search.s3_joint_pairing import (
    analyze_joint_grid_combinations,
    build_candidate_code,
    candidate_from_record,
    enumerate_translation_grid_orbits,
    saturation_even_weight_obstruction,
)
from gala_search.s3_ising import BOTTOM_ORDER

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
TARGETS = {
    "l4-w16": "s3-l4-j1-w16-e1bb058d91825488",
    "l8-w16": "s3-l8-j2-w16-b77391bd2f2fd82b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(TARGETS), required=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--max-orbits", type=int, default=500)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _record(path: pathlib.Path, candidate_id: str) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["candidate_id"] == candidate_id:
            return record
    raise RuntimeError(f"candidate {candidate_id} is absent from {path}")


def main() -> None:
    args = parse_args()
    candidate_id = args.candidate_id or TARGETS[args.family]
    input_path = args.input or PROJECT_DIR / "results" / (
        "s3-l4-w16-seed240828.jsonl"
        if args.family == "l4-w16"
        else "s3-l8-w16-seed240827.jsonl"
    )
    output_path = args.output or PROJECT_DIR / "results" / (
        f"{candidate_id}-joint-pairing.json"
    )
    source = _record(input_path, candidate_id)
    candidate = candidate_from_record(source, args.family)
    code = build_candidate_code(candidate)
    started = time.perf_counter()
    enumeration = enumerate_translation_grid_orbits(
        code,
        weight=6,
        max_orbits=args.max_orbits,
        max_nodes=args.max_nodes,
    )
    maximum_grids = 2 if args.family == "l4-w16" else 3
    folds: dict[str, Any] = {
        "natural": analyze_joint_grid_combinations(
            code, enumeration["grids"], maximum_grids=maximum_grids
        )
    }
    if args.family == "l4-w16":
        alternative = structured_physical_permutation(
            (2, 3, 0, 1),
            internal_affine_permutation((0, 1, 2), (7, 0, 0, 3)),
        )
        folds["alternative"] = analyze_joint_grid_combinations(
            code,
            enumeration["grids"],
            fold_permutation=alternative,
            maximum_grids=maximum_grids,
        )
    result = {
        "candidate_id": candidate_id,
        "family": args.family,
        "parameters": {
            "n": code.num_qubits,
            "k": code.dimension,
            "d": source.get("certified_distance", 6),
        },
        "polynomials": source["polynomials"],
        "enumeration": enumeration,
        "folds": folds,
        "capacity": {
            str(num_grids): {
                "support_weight": num_grids * BOTTOM_ORDER * 6,
                "fits": num_grids * BOTTOM_ORDER * 6 <= code.num_qubits,
                "saturated_even_weight_obstruction": (
                    saturation_even_weight_obstruction(
                        num_qubits=code.num_qubits,
                        num_grids=num_grids,
                        logical_weight=6,
                    )
                ),
            }
            for num_grids in range(1, code.num_qubits // (BOTTOM_ORDER * 6) + 1)
        },
        "seconds": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "candidate_id": candidate_id,
                "num_orbits": enumeration["num_orbits"],
                "enumeration_exhaustive": enumeration["enumeration_exhaustive"],
                "hits": {
                    name: len(data["full_rank_hits"])
                    for name, data in folds.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
