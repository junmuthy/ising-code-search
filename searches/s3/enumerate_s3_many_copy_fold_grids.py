#!/usr/bin/env python3
"""Enumerate distinct weight-six grid orbits in folded L=12 codes."""

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

from gala_search.s3_ising import (
    ProductMonomial,
    analyze_translation_logical_seed,
    find_zero_syndrome_by_tanner_search,
)
from gala_search.s3_many_copy import (
    FoldCandidate,
    S3L12FoldW12Candidate,
    S3L12VertexFoldW12Candidate,
    build_s3_l12_fold_code,
    s3_l12_zx_fold_permutation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_DIR
    / "results"
    / "s3-many-copy"
    / "l12-j3-w12-fold-distance-grid.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results" / "s3-many-copy" / "l12-j3-w12-fold-grid-orbits.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--max-orbits", type=int, default=100)
    parser.add_argument("--vertex-reflection", action="store_true")
    parser.add_argument("--only-non-graph-witnesses", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, os.cpu_count() or 1))
    )
    return parser.parse_args()


def _enumerate(
    record: dict[str, Any],
    max_nodes: int,
    max_orbits: int,
    vertex_reflection: bool,
) -> dict[str, Any]:
    entries = tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["candidate"]["entries"]
    )
    candidate: FoldCandidate = (
        S3L12VertexFoldW12Candidate(entries)
        if vertex_reflection
        else S3L12FoldW12Candidate(entries)
    )
    try:
        code = build_s3_l12_fold_code(candidate)
        supports: list[list[int]] = []
        grids: list[dict[str, Any]] = []
        exhaustive = False
        for _index in range(max_orbits):
            search = find_zero_syndrome_by_tanner_search(
                code,
                weight=6,
                max_nodes=max_nodes,
                require_graph_support=True,
                excluded_translation_orbits=supports,
            )
            if search is None or search.get("support") is None:
                exhaustive = bool(search and search.get("search_exhaustive"))
                break
            support = search["support"]
            supports.append(support)
            grid = analyze_translation_logical_seed(
                code,
                support,
                zx_fold_permutation=s3_l12_zx_fold_permutation(candidate),
            )
            grids.append(
                {
                    "seed_support": support,
                    "internal_orbits": grid["internal_orbits"],
                    "orbit_rank_mod_stabilizers": grid[
                        "orbit_rank_mod_stabilizers"
                    ],
                    "zx_pairing_rank": grid["zx_pairing_rank"],
                }
            )
            if grid["zx_pairing_rank"] == 32:
                break
        best_rank = max((grid["zx_pairing_rank"] for grid in grids), default=0)
        return {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "num_orbits": len(grids),
            "enumeration_exhaustive": exhaustive,
            "best_zx_pairing_rank": best_rank,
            "full_rank_grid_found": best_rank == 32,
            "pairing_rank_counts": Counter(
                grid["zx_pairing_rank"] for grid in grids
            ),
            "grids": grids,
            "status": "full_rank_grid" if best_rank == 32 else "no_full_rank_grid",
        }
    except Exception as error:
        return {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    source = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    target_status = (
        "distance_six_non_graph_witness"
        if args.only_non_graph_witnesses
        else "grid_pairing_failed"
    )
    targets = [record for record in source if record["status"] == target_status]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _enumerate,
                record,
                args.max_nodes,
                args.max_orbits,
                args.vertex_reflection,
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
                        "num_orbits": result.get("num_orbits"),
                        "best_rank": result.get("best_zx_pairing_rank"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    results.sort(key=lambda result: result["candidate_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(results),
                "statuses": Counter(result["status"] for result in results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
