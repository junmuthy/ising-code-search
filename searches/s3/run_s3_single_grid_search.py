#!/usr/bin/env python3
"""Search self-dual S3 x C8 x C4 GALA codes for one clean Ising grid."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

from gala_search.s3_ising import (
    analyze_s3_l4_self_dual_w16_candidate,
    analyze_self_dual_translation_logical_seed,
    bottom_support_generates,
    build_s3_l4_self_dual_w16_code,
    find_zero_syndrome_by_tanner_search,
    has_odd_logical_support,
    random_s3_l4_self_dual_w16_candidates,
    self_dual_active_orthogonality_data,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--target-candidates", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _write_jsonl_atomic(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.raw_limit, args.target_candidates, args.max_nodes, args.progress_every) < 1:
        raise SystemExit("all count and search-limit arguments must be positive")
    output_path = args.output or (
        PROJECT_DIR
        / "results"
        / f"s3-l4-self-dual-single-grid-seed{args.seed}.jsonl"
    )

    counts: Counter[str] = Counter()
    hits: list[dict[str, Any]] = []
    started = time.perf_counter()
    raw_count = 0
    for candidate in random_s3_l4_self_dual_w16_candidates(
        args.raw_limit, seed=args.seed
    ):
        raw_count += 1
        algebraic = {
            **self_dual_active_orthogonality_data(candidate),
            "bottom_support_generates_C8xC4": bottom_support_generates(candidate),
        }
        if not all(algebraic.values()):
            continue
        counts["algebraic_survivors"] += 1

        code = build_s3_l4_self_dual_w16_code(candidate)
        if not has_odd_logical_support(code):
            counts["non_SWEL"] += 1
            continue
        counts["SWEL"] += 1

        lower_weight_searches: list[dict[str, Any]] = []
        rejected = False
        for weight in range(1, 6):
            result = find_zero_syndrome_by_tanner_search(
                code, weight=weight, max_nodes=args.max_nodes
            )
            lower_weight_searches.append(result)
            if not result["search_exhaustive"]:
                counts["lower_weight_cutoff"] += 1
                rejected = True
                break
            if result["support"] is not None:
                counts[f"kernel_weight_{weight}"] += 1
                rejected = True
                break
        if rejected:
            continue
        counts["distance_at_least_6"] += 1

        grid_search = find_zero_syndrome_by_tanner_search(
            code,
            weight=7,
            max_nodes=args.max_nodes,
            require_graph_support=True,
        )
        if not grid_search["search_exhaustive"]:
            counts["grid_search_cutoff"] += 1
            continue
        if grid_search["support"] is None:
            counts["no_weight_7_grid"] += 1
        else:
            grid = analyze_self_dual_translation_logical_seed(
                code, grid_search["support"]
            )
            required_grid_checks = {
                "graph_supported": grid["graph_supported"],
                "pairwise_disjoint": grid["pairwise_disjoint"],
                "z_orbit_in_kernel": grid["z_orbit_in_kernel"],
                "x_partner_orbit_in_kernel": grid["x_partner_orbit_in_kernel"],
                "rank_32_mod_stabilizers": grid["orbit_rank_mod_stabilizers"] == 32,
                "ZX_pairing_rank_32": grid["zx_pairing_rank"] == 32,
                "pairing_is_identity": grid["pairing_is_identity"],
                "x_translation_is_C8_grid_shift": grid[
                    "x_translation_is_C8_grid_shift"
                ],
                "y_translation_is_C4_grid_shift": grid[
                    "y_translation_is_C4_grid_shift"
                ],
            }
            if not all(required_grid_checks.values()):
                raise RuntimeError(
                    f"candidate {candidate.candidate_id} failed a grid invariant: "
                    f"{required_grid_checks}"
                )

            structure = analyze_s3_l4_self_dual_w16_candidate(candidate)
            if not structure["accepted"]:
                counts["structural_grid_rejected"] += 1
                print(
                    json.dumps(
                        {
                            "candidate_id": candidate.candidate_id,
                            "structural_grid_rejection": structure[
                                "rejection_reasons"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            weight_six = find_zero_syndrome_by_tanner_search(
                code, weight=6, max_nodes=args.max_nodes
            )
            if not weight_six["search_exhaustive"]:
                certified_distance: int | None = None
            elif weight_six["support"] is None:
                certified_distance = 7
            elif weight_six.get("is_nontrivial_logical"):
                certified_distance = 6
            else:
                certified_distance = None
            hit = {
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.to_dict(),
                "polynomials": candidate.polynomial_text,
                "raw_index": raw_count,
                "n": code.num_qubits,
                "k": code.dimension,
                "certified_distance": certified_distance,
                "structure": structure,
                "lower_weight_searches": lower_weight_searches,
                "weight_six_search": weight_six,
                "grid_search": grid_search,
                "logical_grid": grid,
            }
            hits.append(hit)
            counts["accepted_single_grid"] += 1
            _write_jsonl_atomic(output_path, hits)

        if counts["distance_at_least_6"] % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "raw": raw_count,
                        "counts": counts,
                        "hits": len(hits),
                        "seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(hits) >= args.target_candidates:
            break

    _write_jsonl_atomic(output_path, hits)
    summary = {
        "output": str(output_path),
        "raw": raw_count,
        "counts": counts,
        "hits": len(hits),
        "seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = output_path.with_suffix(".summary.json")
    _write_json_atomic(summary_path, summary)
    summary["summary"] = str(summary_path)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
