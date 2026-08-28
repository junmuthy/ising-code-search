#!/usr/bin/env python3
"""Exact weight-5/6 and graph-grid screen for two-dimensional S3 survivors."""

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
    parser.add_argument("--max-nodes", type=int, default=2_000_000)
    parser.add_argument("--max-kernels-per-weight", type=int, default=32)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def _entries(record: dict[str, Any]) -> tuple[tuple[ProductMonomial, ...], ...]:
    return tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["entries"]
    )


def _logical_at_weight(
    code,
    *,
    weight: int,
    max_nodes: int,
    max_kernels: int,
    require_graph_support: bool = False,
) -> dict[str, Any]:
    excluded: list[list[int]] = []
    searches: list[dict[str, Any]] = []
    for _index in range(max_kernels):
        search = find_linear_zero_syndrome_by_tanner_search(
            code,
            weight=weight,
            max_nodes=max_nodes,
            require_graph_support=require_graph_support,
            excluded_translation_orbits=excluded,
        )
        searches.append(search)
        if search["support"] is None:
            return {
                "logical": None,
                "search_exhaustive": search["search_exhaustive"],
                "kernels_excluded": len(excluded),
                "searches": searches,
            }
        if search.get("is_nontrivial_logical"):
            return {
                "logical": search,
                "search_exhaustive": search["search_exhaustive"],
                "kernels_excluded": len(excluded),
                "searches": searches,
            }
        excluded.append(search["support"])
    return {
        "logical": None,
        "search_exhaustive": False,
        "kernels_excluded": len(excluded),
        "searches": searches,
        "kernel_limit_reached": True,
    }


def _screen(
    record: dict[str, Any], max_nodes: int, max_kernels: int
) -> dict[str, Any]:
    try:
        family: Family = record["family"]
        code = build_linear_code(_entries(record), family)
        result: dict[str, Any] = {
            "source_candidate_id": record["source_candidate_id"],
            "source_family": record["source_family"],
            "family": family,
            "entries": record["entries"],
            "n": code.num_qubits,
            "k": code.dimension,
            "maximum_check_weight": record["maximum_check_weight"],
        }
        weight_five = _logical_at_weight(
            code,
            weight=5,
            max_nodes=max_nodes,
            max_kernels=max_kernels,
        )
        result["weight_five"] = weight_five
        if weight_five["logical"] is not None:
            result["status"] = "logical_weight_five"
            return result
        if not weight_five["search_exhaustive"]:
            result["status"] = "weight_five_incomplete"
            return result

        weight_six = _logical_at_weight(
            code,
            weight=6,
            max_nodes=max_nodes,
            max_kernels=max_kernels,
        )
        result["weight_six"] = weight_six
        if weight_six["logical"] is not None:
            result["certified_distance"] = 6
            graph_search = (
                weight_six
                if weight_six["logical"].get("graph_supported")
                else _logical_at_weight(
                    code,
                    weight=6,
                    max_nodes=max_nodes,
                    max_kernels=max_kernels,
                    require_graph_support=True,
                )
            )
            result["weight_six_graph"] = graph_search
            graph = graph_search["logical"]
            if graph is None:
                result["status"] = (
                    "distance_six_no_graph"
                    if graph_search["search_exhaustive"]
                    else "distance_six_graph_incomplete"
                )
                return result
            fold = expected_zx_fold(code.num_qubits // 64, family)
            result["grid"] = analyze_linear_translation_seed(
                code, graph["support"], fold=fold
            )
            result["status"] = "distance_six_graph"
            return result
        if not weight_six["search_exhaustive"]:
            result["status"] = "weight_six_incomplete"
            return result

        result["distance_lower_bound"] = 7
        weight_seven_graph = _logical_at_weight(
            code,
            weight=7,
            max_nodes=max_nodes,
            max_kernels=max_kernels,
            require_graph_support=True,
        )
        result["weight_seven_graph"] = weight_seven_graph
        graph = weight_seven_graph["logical"]
        if graph is None:
            result["status"] = (
                "distance_at_least_seven_no_graph_seven"
                if weight_seven_graph["search_exhaustive"]
                else "distance_at_least_seven_graph_incomplete"
            )
            return result
        fold = expected_zx_fold(code.num_qubits // 64, family)
        result["grid"] = analyze_linear_translation_seed(
            code, graph["support"], fold=fold
        )
        result["status"] = "distance_at_least_seven_graph_seven"
        return result
    except Exception as error:
        return {
            "source_candidate_id": record.get("source_candidate_id"),
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
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
    targets = [record for record in records if record.get("accepted")]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_screen, record, args.max_nodes, args.max_kernels_per_weight)
            for record in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(futures):
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
        "source_records": len(records),
        "survivors_screened": len(targets),
        "status_counts": Counter(item["status"] for item in results),
        "grid_rank_counts": {
            f"logical={logical_rank},zx={zx_rank}": count
            for (logical_rank, zx_rank), count in Counter(
                (
                    item["grid"]["orbit_rank_mod_stabilizers"],
                    item["grid"]["zx_pairing_rank"],
                )
                for item in results
                if item.get("grid")
            ).items()
        },
        "max_nodes": args.max_nodes,
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
