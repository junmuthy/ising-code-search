#!/usr/bin/env python3
"""Distance-5/6 and translated-grid screen for folded L=12 survivors."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any, Iterable

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
    PROJECT_DIR / "results" / "s3-many-copy" / "l12-j3-w12-fold.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results"
    / "s3-many-copy"
    / "l12-j3-w12-fold-distance-grid.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--vertex-reflection", action="store_true")
    return parser.parse_args()


def _candidate_from_record(
    record: dict[str, Any], vertex_reflection: bool
) -> FoldCandidate:
    entries = tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["candidate"]["entries"]
    )
    return (
        S3L12VertexFoldW12Candidate(entries)
        if vertex_reflection
        else S3L12FoldW12Candidate(entries)
    )


def _screen(
    record: dict[str, Any], max_nodes: int, vertex_reflection: bool
) -> dict[str, Any]:
    candidate = _candidate_from_record(record, vertex_reflection)
    try:
        code = build_s3_l12_fold_code(candidate)
        weight_five = find_zero_syndrome_by_tanner_search(
            code, weight=5, max_nodes=max_nodes
        )
        result: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "polynomials": candidate.polynomial_text,
            "n": code.num_qubits,
            "k": code.dimension,
            "weight_five": weight_five,
            "accepted_distance": False,
            "grid": None,
        }
        if weight_five is None or not weight_five.get("search_exhaustive"):
            result["status"] = "weight_five_incomplete"
            return result
        if weight_five.get("support") is not None:
            result["status"] = "logical_weight_five"
            return result

        weight_six = find_zero_syndrome_by_tanner_search(
            code, weight=6, max_nodes=max_nodes
        )
        result["weight_six"] = weight_six
        if weight_six is None or not weight_six.get("search_exhaustive"):
            result["status"] = "weight_six_incomplete"
            return result
        result["accepted_distance"] = True
        if weight_six.get("support") is None:
            result["status"] = "distance_at_least_seven"
            return result
        result["certified_distance"] = 6
        if weight_six.get("graph_supported"):
            grid = analyze_translation_logical_seed(
                code,
                weight_six["support"],
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
        else:
            result["status"] = "distance_six_non_graph_witness"
        return result
    except Exception as error:
        return {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _write_jsonl_atomic(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_nodes < 1:
        raise SystemExit("worker and node limits must be positive")
    source = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    candidates = [record for record in source if record.get("accepted")]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _screen, record, args.max_nodes, args.vertex_reflection
            ): record
            for record in candidates
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "statuses": Counter(
                                record["status"] for record in records
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    records.sort(key=lambda record: record["candidate_id"])
    _write_jsonl_atomic(args.output, records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "statuses": Counter(record["status"] for record in records),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
