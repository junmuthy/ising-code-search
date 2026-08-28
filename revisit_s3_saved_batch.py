#!/usr/bin/env python3
"""Rescreen every saved exact-distance-six S3 candidate for joint grids."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any

from gala_search.s3_joint_pairing import (
    analyze_joint_grid_combinations,
    build_candidate_code,
    candidate_from_record,
    enumerate_translation_grid_orbits,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("l4-w16", "l8-w16"), required=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--max-orbits", type=int, default=500)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--logical-weight", type=int, choices=(6, 7), default=6)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def _analyze(
    record: dict[str, Any],
    family: str,
    logical_weight: int,
    max_orbits: int,
    max_nodes: int,
) -> dict[str, Any]:
    try:
        candidate = candidate_from_record(record, family)  # type: ignore[arg-type]
        code = build_candidate_code(candidate)
        enumeration = enumerate_translation_grid_orbits(
            code,
            weight=logical_weight,
            max_orbits=max_orbits,
            max_nodes=max_nodes,
        )
        combinations = analyze_joint_grid_combinations(
            code,
            enumeration["grids"],
            maximum_grids=min(
                3, code.num_qubits // (32 * logical_weight)
            ),
        )
        return {
            "candidate_id": record["candidate_id"],
            "candidate": record["candidate"],
            "polynomials": record["polynomials"],
            "parameters": {
                "n": code.num_qubits,
                "k": code.dimension,
                "d": record.get("certified_distance"),
                "distance_lower_bound": record.get("distance_lower_bound"),
            },
            "enumeration": enumeration,
            "joint_pairing": combinations,
            "status": (
                "full_rank_joint_sector"
                if combinations["full_rank_hits"]
                else "no_full_rank_joint_sector"
            ),
        }
    except Exception as error:
        return {
            "candidate_id": record["candidate_id"],
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    default_stem = "s3-l4-w16-seed240828" if args.family == "l4-w16" else "s3-l8-w16-seed240827"
    input_path = args.input or PROJECT_DIR / "results" / f"{default_stem}.jsonl"
    suffix = "joint-pairing" if args.logical_weight == 6 else "weight7-joint-pairing"
    output_path = args.output or PROJECT_DIR / "results" / f"{default_stem}-{suffix}.jsonl"
    records = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]
    targets = [
        record
        for record in records
        if (
            record.get("certified_distance") == 6
            if args.logical_weight == 6
            else record.get("distance_lower_bound") == 7
        )
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _analyze,
                record,
                args.family,
                args.logical_weight,
                args.max_orbits,
                args.max_nodes,
            )
            for record in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "candidate_id": result["candidate_id"],
                        "status": result["status"],
                        "num_orbits": result.get("enumeration", {}).get(
                            "num_orbits"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    results.sort(key=lambda result: result["candidate_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": len(results),
                "statuses": Counter(result["status"] for result in results),
                "total_grid_orbits": sum(
                    result.get("enumeration", {}).get("num_orbits", 0)
                    for result in results
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
