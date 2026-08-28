#!/usr/bin/env python3
"""Enumerate disjoint jointly ZX-paired grids in resized linear-S3 pilots."""

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
    analyze_linear_grid_combinations,
    analyze_linear_translation_seed,
    build_linear_fold_code,
    find_linear_zero_syndrome_by_tanner_search,
    linear_fold_permutation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
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
        code = build_linear_fold_code(
            _entries(record),
            active_rows=record["active_rows"],
            shift=record["shift"],
        )
        fold = linear_fold_permutation(
            half_blocks=record["half_blocks"],
            active_rows=record["active_rows"],
            shift=record["shift"],
        )
        weight = int(record["grid"]["weight"])
        supports: list[list[int]] = []
        grids: list[dict[str, Any]] = []
        exhaustive = False
        last_search: dict[str, Any] | None = None
        for _ in range(max_orbits):
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
        maximum_grids = min(
            int(record["target_grids"]), code.num_qubits // (32 * weight)
        )
        combinations = analyze_linear_grid_combinations(
            code, grids, fold=fold, maximum_grids=maximum_grids
        )
        target_hits = [
            hit
            for hit in combinations["full_rank_hits"]
            if hit["num_grids"] == record["target_grids"]
        ]
        return {
            key: record.get(key)
            for key in (
                "candidate_id",
                "pilot",
                "half_blocks",
                "active_rows",
                "target_grids",
                "shift",
                "entries",
                "n",
                "k",
                "maximum_check_weight",
                "certified_distance",
                "distance_lower_bound",
            )
        } | {
            "logical_weight": weight,
            "num_grid_orbits": len(grids),
            "enumeration_exhaustive": exhaustive,
            "truncated": len(supports) >= max_orbits and not exhaustive,
            "last_search": last_search,
            "grids": grids,
            "combinations": combinations,
            "target_hits": target_hits,
            "status": "target_grid_set" if target_hits else "no_target_grid_set",
        }
    except Exception as error:
        return {
            "candidate_id": record.get("candidate_id"),
            "pilot": record.get("pilot"),
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary = output.with_suffix(".summary.json")
    for path in (output, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    targets = [record for record in source if record.get("grid")]
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
    results.sort(key=lambda item: (str(item.get("pilot")), str(item.get("candidate_id"))))
    report = {
        "input": str(args.input),
        "targets": len(targets),
        "max_nodes": args.max_nodes,
        "max_orbits": args.max_orbits,
        "status_counts": Counter(item["status"] for item in results),
        "pilot_status_counts": {
            pilot: Counter(item["status"] for item in results if item.get("pilot") == pilot)
            for pilot in sorted(set(item.get("pilot") for item in results))
        },
        "exhaustive": sum(bool(item.get("enumeration_exhaustive")) for item in results),
        "total_grid_orbits": sum(int(item.get("num_grid_orbits", 0)) for item in results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(output)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary)
    print(json.dumps({"output": str(output), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
