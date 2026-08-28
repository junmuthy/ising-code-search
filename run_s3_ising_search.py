#!/usr/bin/env python3
"""Search S3 x C8 x C4 polynomial GALA codes for Ising logical grids."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from collections import Counter
from typing import Any, Callable, Iterator

from gala_search.s3_ising import (
    S3L4W16Candidate,
    S3L8W16Candidate,
    active_orthogonality_data,
    analyze_s3_l4_w16_candidate,
    analyze_s3_l8_w16_candidate,
    analyze_translation_logical_seed,
    analyze_translation_logical_seeds,
    bottom_support_generates,
    build_s3_l4_w16_code,
    build_s3_l8_w16_code,
    find_zero_syndrome_by_tanner_search,
    l8_active_orthogonality_data,
    l8_bottom_support_generates,
    random_s3_l4_w16_candidates,
    random_s3_l8_w16_candidates,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("l4-w16", "l8-w16"), default="l4-w16"
    )
    parser.add_argument(
        "--accepted-candidates",
        type=int,
        default=100,
        help="number of cheap group-algebra survivors to retain",
    )
    parser.add_argument("--raw-limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=240828)
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def _compact_orbit(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "translated_supports"}


def _write_jsonl_atomic(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.accepted_candidates < 1 or args.raw_limit < 1 or args.max_nodes < 1:
        raise SystemExit("candidate counts and --max-nodes must be positive")

    if args.family == "l4-w16":
        generator: Callable[..., Iterator[S3L4W16Candidate | S3L8W16Candidate]] = (
            random_s3_l4_w16_candidates
        )
        active = active_orthogonality_data
        bottom = bottom_support_generates
        build = build_s3_l4_w16_code
        analyze = analyze_s3_l4_w16_candidate
        default_name = f"s3-l4-w16-seed{args.seed}.jsonl"
    else:
        generator = random_s3_l8_w16_candidates
        active = l8_active_orthogonality_data
        bottom = l8_bottom_support_generates
        build = build_s3_l8_w16_code
        analyze = analyze_s3_l8_w16_candidate
        default_name = f"s3-l8-w16-seed{args.seed}.jsonl"
    output_path = args.output or PROJECT_DIR / "results" / default_name

    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    raw_count = 0
    for candidate in generator(args.raw_limit, seed=args.seed):
        raw_count += 1
        cheap_checks = {
            **active(candidate),
            "bottom_support_generates_C8xC4": bottom(candidate),
        }
        if not all(cheap_checks.values()):
            continue

        code = build(candidate)
        structure = analyze(candidate, include_graph_metrics=False)
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "polynomials": candidate.polynomial_text,
            "raw_index": raw_count,
            "n": code.num_qubits,
            "k": code.dimension,
            "structure": structure,
        }
        if not structure["accepted"]:
            counts["lifted_rejected"] += 1
            records.append(record)
            if len(records) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "raw": raw_count,
                            "algebraic_survivors": len(records),
                            "counts": counts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if len(records) >= args.accepted_candidates:
                break
            continue
        counts["lifted_valid"] += 1

        kernel_searches: list[dict[str, Any]] = []
        first_kernel: dict[str, Any] | None = None
        for weight in range(1, 7):
            kernel = find_zero_syndrome_by_tanner_search(
                code, weight=weight, max_nodes=args.max_nodes
            )
            kernel_searches.append(kernel)
            if not kernel["search_exhaustive"] or kernel["support"] is not None:
                first_kernel = kernel
                break
        record["kernel_searches"] = kernel_searches

        lower_weights_exhausted = (
            len(kernel_searches) >= 5
            and all(
                result["search_exhaustive"] and result["support"] is None
                for result in kernel_searches[:5]
            )
        )
        if first_kernel is not None and first_kernel["support"] is not None:
            counts[f"minimum_kernel_weight_{first_kernel['weight']}"] += 1
            if lower_weights_exhausted and first_kernel.get("is_nontrivial_logical"):
                record["certified_distance"] = first_kernel["weight"]
                # The verified ZX fold is a qubit permutation exchanging X and
                # Z checks, so the two CSS distances are equal.
                record["certified_distance_x"] = first_kernel["weight"]
                record["certified_distance_z"] = first_kernel["weight"]
        elif first_kernel is None:
            counts["no_kernel_through_weight_6"] += 1
            record["distance_lower_bound"] = 7
        else:
            counts["tanner_search_cutoff"] += 1

        if lower_weights_exhausted:
            graph = find_zero_syndrome_by_tanner_search(
                code,
                weight=6,
                max_nodes=args.max_nodes,
                require_graph_support=True,
            )
            record["graph_seed_search"] = graph
            if graph["support"] is not None and graph.get("is_nontrivial_logical"):
                sector = analyze_translation_logical_seed(code, graph["support"])
                record["logical_sectors"] = [_compact_orbit(sector)]
                counts["graph_logical_sector"] += 1

                supports = [graph["support"]]
                used = set(sector["internal_orbits"])
                target_sectors = candidate.num_blocks // 2
                while len(supports) < target_sectors:
                    next_graph = find_zero_syndrome_by_tanner_search(
                        code,
                        weight=6,
                        max_nodes=args.max_nodes,
                        require_graph_support=True,
                        forbidden_internal_orbits=sorted(used),
                    )
                    if next_graph["support"] is None:
                        break
                    next_sector = analyze_translation_logical_seed(
                        code, next_graph["support"]
                    )
                    if next_sector["orbit_rank_mod_stabilizers"] == 0:
                        break
                    supports.append(next_graph["support"])
                    used.update(next_sector["internal_orbits"])
                    record["logical_sectors"].append(_compact_orbit(next_sector))

                combined = analyze_translation_logical_seeds(code, supports)
                record["combined_logical_sectors"] = {
                    "num_sectors": combined["num_sectors"],
                    "num_logicals": combined["num_logicals"],
                    "pairwise_disjoint": combined["pairwise_disjoint"],
                    "combined_rank_mod_stabilizers": combined[
                        "combined_rank_mod_stabilizers"
                    ],
                    "combined_zx_pairing_rank": combined[
                        "combined_zx_pairing_rank"
                    ],
                }

        records.append(record)
        if len(records) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "raw": raw_count,
                        "algebraic_survivors": len(records),
                        "counts": counts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(records) >= args.accepted_candidates:
            break

    _write_jsonl_atomic(output_path, records)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "raw": raw_count,
                "algebraic_survivors": len(records),
                "counts": counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
