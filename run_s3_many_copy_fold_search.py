#!/usr/bin/env python3
"""Pilot the weight-12, uniform-degree, translation-ZX-fold L=12 family."""

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

from gala_search.s3_ising import find_logical_up_to_weight_four
from gala_search.s3_many_copy import (
    FoldCandidate,
    S3L12W12Candidate,
    _bottom_support_generates,
    analyze_s3_l12_fold_candidate,
    build_s3_l12_fold_code,
    random_s3_l12_fold_candidates,
    s3_l12_fold_active_orthogonality_data,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results" / "s3-many-copy" / "l12-j3-w12-fold.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-limit", type=int, default=200_000)
    parser.add_argument("--algebraic-target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=311000)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--skip-weight-four-screen", action="store_true")
    parser.add_argument("--vertex-reflection", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _analyze_worker(
    candidate: FoldCandidate, screen_weight_four: bool
) -> dict[str, Any]:
    try:
        result = analyze_s3_l12_fold_candidate(candidate)
        if result["accepted"] and screen_weight_four:
            code = build_s3_l12_fold_code(candidate)
            result["logical_up_to_weight_four"] = find_logical_up_to_weight_four(
                code
            )
            if result["logical_up_to_weight_four"] is not None:
                result["accepted"] = False
                result["rejection_reasons"].append("logical_up_to_weight_four")
        return result
    except Exception as error:
        return {
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reasons": ["worker_error"],
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
    if min(args.raw_limit, args.algebraic_target, args.workers) < 1:
        raise SystemExit("search limits and worker count must be positive")
    candidates: list[FoldCandidate] = []
    counters: Counter[str] = Counter()
    for candidate in random_s3_l12_fold_candidates(
        args.raw_limit,
        seed=args.seed,
        vertex_reflection=args.vertex_reflection,
    ):
        counters["raw"] += 1
        orthogonality = s3_l12_fold_active_orthogonality_data(candidate)
        if not orthogonality["active_offsets_zero"]:
            counters["active_orthogonality_rejected"] += 1
            continue
        counters["active_orthogonality"] += 1
        if not orthogonality["latent_offset_3_nonzero"]:
            counters["latent_zero_rejected"] += 1
            continue
        counters["active_with_latent"] += 1
        exact_wrapper = S3L12W12Candidate(
            candidate.entries, "uniform_degree3"
        )
        if not _bottom_support_generates(exact_wrapper):
            counters["bottom_disconnected_rejected"] += 1
            continue
        counters["algebraic_survivors"] += 1
        candidates.append(candidate)
        if len(candidates) >= args.algebraic_target:
            break
    print(
        json.dumps(
            {
                "stage": "algebraic",
                "seed": args.seed,
                "counts": counters,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _analyze_worker,
                candidate,
                not args.skip_weight_four_screen,
            ): candidate
            for candidate in candidates
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "stage": "lifted",
                            "completed": completed,
                            "accepted": sum(
                                bool(record.get("accepted")) for record in records
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    records.sort(key=lambda record: record["candidate_id"])
    _write_jsonl_atomic(args.output, records)
    statuses = Counter(
        "accepted"
        if record.get("accepted")
        else record.get("rejection_reasons", ["unknown"])[0]
        for record in records
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "statuses": statuses,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
