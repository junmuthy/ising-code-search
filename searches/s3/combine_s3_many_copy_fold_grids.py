#!/usr/bin/env python3
"""Test disjoint four-grid combinations for full combined ZX pairing."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any

import numpy as np

from gala_search.s3_ising import (
    ProductMonomial,
    _gf2_rank,
    _permute_columns,
    logical_translation_orbit,
)
from gala_search.s3_many_copy import (
    FoldCandidate,
    S3L12FoldW12Candidate,
    S3L12VertexFoldW12Candidate,
    build_s3_l12_fold_code,
    s3_l12_zx_fold_permutation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--vertex-reflection", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=1_000_000)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, os.cpu_count() or 1))
    )
    return parser.parse_args()


def _candidate(record: dict[str, Any], vertex: bool) -> FoldCandidate:
    entries = tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["candidate"]["entries"]
    )
    return (
        S3L12VertexFoldW12Candidate(entries)
        if vertex
        else S3L12FoldW12Candidate(entries)
    )


def _screen(
    record: dict[str, Any], vertex: bool, max_combinations: int
) -> dict[str, Any]:
    candidate = _candidate(record, vertex)
    try:
        code = build_s3_l12_fold_code(candidate)
        fold = s3_l12_zx_fold_permutation(candidate)
        grids = record["grids"]
        internal = [set(grid["internal_orbits"]) for grid in grids]
        orbit_cache: dict[int, np.ndarray] = {}
        combinations_tested = 0
        disjoint_combinations = 0
        best_pairing_rank = 0
        best_combination: tuple[int, ...] | None = None
        full_result: dict[str, Any] | None = None
        truncated = False
        for indices in itertools.combinations(range(len(grids)), 4):
            combinations_tested += 1
            if combinations_tested > max_combinations:
                truncated = True
                break
            used: set[int] = set()
            is_disjoint = True
            for index in indices:
                if used & internal[index]:
                    is_disjoint = False
                    break
                used.update(internal[index])
            if not is_disjoint:
                continue
            disjoint_combinations += 1
            for index in indices:
                if index not in orbit_cache:
                    seed = np.zeros(code.num_qubits, dtype=np.uint8)
                    seed[grids[index]["seed_support"]] = 1
                    orbit_cache[index] = logical_translation_orbit(
                        seed, num_blocks=12
                    )
            z_logicals = np.vstack([orbit_cache[index] for index in indices])
            x_logicals = _permute_columns(z_logicals, fold)
            pairing_rank = _gf2_rank((z_logicals @ x_logicals.T) % 2, code.field)
            if pairing_rank > best_pairing_rank:
                best_pairing_rank = pairing_rank
                best_combination = indices
            if pairing_rank != 128:
                continue
            combined_rank = (
                _gf2_rank(
                    np.vstack([np.asarray(code.matrix_z, dtype=np.uint8), z_logicals]),
                    code.field,
                )
                - code.code_z.rank
            )
            if combined_rank == 128:
                full_result = {
                    "indices": list(indices),
                    "seed_supports": [grids[index]["seed_support"] for index in indices],
                    "internal_orbits": [grids[index]["internal_orbits"] for index in indices],
                    "combined_rank_mod_stabilizers": combined_rank,
                    "combined_zx_pairing_rank": pairing_rank,
                }
                break
        return {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "num_available_grids": len(grids),
            "combinations_tested": combinations_tested,
            "disjoint_four_grid_combinations": disjoint_combinations,
            "best_combined_zx_pairing_rank": best_pairing_rank,
            "best_combination": list(best_combination) if best_combination else None,
            "full_rank_four_grid": full_result,
            "truncated": truncated,
            "status": "full_rank_four_grid" if full_result else "no_full_rank_four_grid",
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
    records = [
        json.loads(line)
        for path in args.input
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _screen,
                record,
                args.vertex_reflection,
                args.max_combinations,
            )
            for record in records
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
                        "disjoint_combinations": result.get(
                            "disjoint_four_grid_combinations"
                        ),
                        "best_rank": result.get("best_combined_zx_pairing_rank"),
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
