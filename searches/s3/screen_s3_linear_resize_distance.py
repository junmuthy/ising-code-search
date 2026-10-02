#!/usr/bin/env python3
"""Distance-5/6 and first-grid screen for resized linear-S3 pilots."""

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
    analyze_linear_translation_seed,
    build_linear_fold_code,
    linear_fold_permutation,
)
from searches.s3.screen_s3_linear_distance import _logical_at_weight

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-check-weight", type=int)
    parser.add_argument("--require-even-syndrome", action="store_true")
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


def _screen(record: dict[str, Any], max_nodes: int, max_kernels: int) -> dict[str, Any]:
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
        result: dict[str, Any] = {
            key: record[key]
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
                "even_syndrome_parity",
            )
        }
        weight_five = _logical_at_weight(
            code, weight=5, max_nodes=max_nodes, max_kernels=max_kernels
        )
        result["weight_five"] = weight_five
        if weight_five["logical"] is not None:
            result["status"] = "logical_weight_five"
            return result
        if not weight_five["search_exhaustive"]:
            result["status"] = "weight_five_incomplete"
            return result
        weight_six = _logical_at_weight(
            code, weight=6, max_nodes=max_nodes, max_kernels=max_kernels
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
            if graph_search["logical"] is None:
                result["status"] = (
                    "distance_six_no_graph"
                    if graph_search["search_exhaustive"]
                    else "distance_six_graph_incomplete"
                )
                return result
            result["grid"] = analyze_linear_translation_seed(
                code, graph_search["logical"]["support"], fold=fold
            )
            result["status"] = "distance_six_graph"
            return result
        if not weight_six["search_exhaustive"]:
            result["status"] = "weight_six_incomplete"
            return result
        result["distance_lower_bound"] = 7
        weight_seven = _logical_at_weight(
            code,
            weight=7,
            max_nodes=max_nodes,
            max_kernels=max_kernels,
            require_graph_support=True,
        )
        result["weight_seven_graph"] = weight_seven
        if weight_seven["logical"] is None:
            result["status"] = (
                "distance_at_least_seven_no_graph_seven"
                if weight_seven["search_exhaustive"]
                else "distance_at_least_seven_graph_incomplete"
            )
            return result
        result["grid"] = analyze_linear_translation_seed(
            code, weight_seven["logical"]["support"], fold=fold
        )
        result["status"] = "distance_at_least_seven_graph_seven"
        return result
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
    targets = [
        record
        for record in source
        if record.get("accepted")
        and (
            args.max_check_weight is None
            or record["maximum_check_weight"] <= args.max_check_weight
        )
        and (
            not args.require_even_syndrome
            or record.get("even_syndrome_parity")
        )
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_screen, record, args.max_nodes, args.max_kernels_per_weight)
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
        "source_records": len(source),
        "survivors_screened": len(targets),
        "max_nodes": args.max_nodes,
        "max_check_weight": args.max_check_weight,
        "require_even_syndrome": args.require_even_syndrome,
        "status_counts": Counter(item["status"] for item in results),
        "pilot_status_counts": {
            pilot: Counter(item["status"] for item in results if item.get("pilot") == pilot)
            for pilot in sorted(set(item.get("pilot") for item in results))
        },
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
