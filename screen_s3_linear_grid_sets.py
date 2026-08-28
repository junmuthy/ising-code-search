#!/usr/bin/env python3
"""Enumerate and jointly pair translated grids in two-dimensional S3 codes."""

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

from gala_search.s3_ising import ProductMonomial
from gala_search.s3_linear import (
    Family,
    analyze_linear_grid_combinations,
    analyze_linear_translation_seed,
    build_linear_code,
    expected_zx_fold,
    find_linear_zero_syndrome_by_tanner_search,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-check-weight", type=int)
    parser.add_argument("--max-orbits", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def _entries(record: dict[str, Any]) -> tuple[tuple[ProductMonomial, ...], ...]:
    return tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["entries"]
    )


def _screen(record: dict[str, Any], max_orbits: int, max_nodes: int) -> dict[str, Any]:
    try:
        family: Family = record["family"]
        code = build_linear_code(_entries(record), family)
        fold = expected_zx_fold(code.num_qubits // 64, family)
        weight = int(record["grid"]["weight"])
        supports: list[list[int]] = []
        grids: list[dict[str, Any]] = []
        exhaustive = False
        last_search = None
        for _index in range(max_orbits):
            last_search = find_linear_zero_syndrome_by_tanner_search(
                code,
                weight=weight,
                max_nodes=max_nodes,
                require_graph_support=True,
                excluded_translation_orbits=supports,
            )
            if last_search["support"] is None:
                exhaustive = bool(last_search["search_exhaustive"])
                break
            supports.append(last_search["support"])
            if not last_search.get("is_nontrivial_logical"):
                continue
            grid = analyze_linear_translation_seed(
                code, last_search["support"], fold=fold
            )
            if grid["orbit_rank_mod_stabilizers"]:
                grids.append(grid)
        maximum_grids = min(3, code.num_qubits // (32 * weight))
        combinations = analyze_linear_grid_combinations(
            code,
            grids,
            fold=fold,
            maximum_grids=maximum_grids,
        )
        return {
            "source_candidate_id": record["source_candidate_id"],
            "source_family": record["source_family"],
            "family": family,
            "entries": record["entries"],
            "parameters": {
                "n": code.num_qubits,
                "k": code.dimension,
                "d": record.get("certified_distance"),
                "distance_lower_bound": record.get("distance_lower_bound"),
            },
            "maximum_check_weight": record["maximum_check_weight"],
            "logical_weight": weight,
            "num_grid_orbits": len(grids),
            "enumeration_exhaustive": exhaustive,
            "truncated": len(supports) >= max_orbits and not exhaustive,
            "last_search": last_search,
            "grids": grids,
            "combinations": combinations,
            "status": (
                "full_rank_grid_set"
                if combinations["full_rank_hits"]
                else "no_full_rank_grid_set"
            ),
        }
    except Exception as error:
        return {
            "source_candidate_id": record.get("source_candidate_id"),
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary_path = output_path.with_suffix(".summary.json")
    for path in (output_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    records = [
        json.loads(line)
        for path in args.input
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    targets = [
        record
        for record in records
        if record.get("grid")
        and (
            args.max_check_weight is None
            or record["maximum_check_weight"] <= args.max_check_weight
        )
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_screen, record, args.max_orbits, args.max_nodes)
            for record in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(futures),
                        "statuses": Counter(item["status"] for item in results),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    results.sort(key=lambda item: str(item.get("source_candidate_id")))
    summary = {
        "inputs": [str(path) for path in args.input],
        "targets": len(targets),
        "status_counts": Counter(item["status"] for item in results),
        "exhaustive": sum(bool(item.get("enumeration_exhaustive")) for item in results),
        "truncated": sum(bool(item.get("truncated")) for item in results),
        "total_grid_orbits": sum(item.get("num_grid_orbits", 0) for item in results),
        "max_check_weight": args.max_check_weight,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(output_path)
    with tempfile.NamedTemporaryFile("w", dir=summary_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({"output": str(output_path), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
