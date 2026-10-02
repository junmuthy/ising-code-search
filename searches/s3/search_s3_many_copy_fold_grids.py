#!/usr/bin/env python3
"""Search the d>=7 folded L=12 candidates for weight-seven logical grids."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
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
    PROJECT_DIR / "results" / "s3-many-copy" / "l12-j3-w12-fold-weight7-grids.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--vertex-reflection", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, os.cpu_count() or 1))
    )
    return parser.parse_args()


def _screen(
    record: dict[str, Any], max_nodes: int, vertex_reflection: bool
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
        search = find_zero_syndrome_by_tanner_search(
            code,
            weight=7,
            max_nodes=max_nodes,
            require_graph_support=True,
        )
        result: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "search": search,
            "grid": None,
        }
        if search is None or search.get("support") is None:
            result["status"] = (
                "no_weight7_grid"
                if search is not None and search.get("search_exhaustive")
                else "incomplete"
            )
            return result
        grid = analyze_translation_logical_seed(
            code,
            search["support"],
            zx_fold_permutation=s3_l12_zx_fold_permutation(candidate),
        )
        result["grid"] = {
            key: grid[key]
            for key in (
                "weight",
                "seed_support",
                "internal_orbits",
                "pairwise_disjoint",
                "z_orbit_in_kernel",
                "x_partner_orbit_in_kernel",
                "orbit_rank_mod_stabilizers",
                "zx_pairing_rank",
            )
        }
        result["status"] = (
            "full_rank_grid"
            if grid["orbit_rank_mod_stabilizers"] == 32
            and grid["zx_pairing_rank"] == 32
            else "grid_pairing_failed"
        )
        return result
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
    records = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    targets = [
        record for record in records if record["status"] == "distance_at_least_seven"
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _screen, record, args.max_nodes, args.vertex_reflection
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
    print(json.dumps({"output": str(args.output), "records": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
