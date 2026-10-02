#!/usr/bin/env python3
"""Search the ``C_28 x C_4`` and ``C_7 x C_4`` order-seven-fibre codes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from collections.abc import Iterable
from typing import Any

from gala_search.abelian_fibre_codes import (
    FibreCodeCandidate,
    analyze_candidate,
    certify_distance,
    iter_four_term_candidates,
    iter_six_term_candidates,
    make_six_term_candidate,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_DIR / "results" / "fibre-codes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--target",
        action="append",
        choices=("n224-k32", "n56-k8"),
        help="repeat to select targets; defaults to both",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--no-symmetry-quotient", action="store_true")
    parser.add_argument("--skip-graph-metrics", action="store_true")
    return parser.parse_args()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _analyze_safely(
    candidate: FibreCodeCandidate, include_graph_metrics: bool
) -> dict[str, Any]:
    try:
        return analyze_candidate(
            candidate, include_graph_metrics=include_graph_metrics
        )
    except Exception as error:
        return {
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "accepted": False,
            "rejection_reasons": ["worker_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def candidates_for_target(
    target: str, *, quotient_symmetries: bool
) -> list[FibreCodeCandidate]:
    if target == "n224-k32":
        candidates = list(
            iter_six_term_candidates(
                4, quotient_symmetries=quotient_symmetries
            )
        )
        inherited = make_six_term_candidate(4, 3, 1, 3, 6, 2)
        candidates.sort(
            key=lambda candidate: (
                candidate.equivalence_key != inherited.equivalence_key,
                candidate,
            )
        )
        return candidates
    if target == "n56-k8":
        return sorted(
            [
                *iter_four_term_candidates(
                    quotient_symmetries=quotient_symmetries
                ),
                *iter_six_term_candidates(
                    1, quotient_symmetries=quotient_symmetries
                ),
            ],
            key=lambda candidate: (len(candidate.support), candidate),
        )
    raise ValueError(f"unsupported target: {target}")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    output_dir = RESULTS_ROOT / args.run_name
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    targets = args.target or ["n224-k32", "n56-k8"]
    include_graph_metrics = not args.skip_graph_metrics
    summaries = []
    for target in targets:
        candidates = candidates_for_target(
            target, quotient_symmetries=not args.no_symmetry_quotient
        )
        print(
            f"Analyzing {len(candidates)} candidates for {target} with "
            f"{args.workers} workers",
            flush=True,
        )
        records_by_id: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    _analyze_safely, candidate, include_graph_metrics
                ): candidate
                for candidate in candidates
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                candidate = futures[future]
                records_by_id[candidate.candidate_id] = future.result()
                if completed % 50 == 0 or completed == len(candidates):
                    accepted = sum(
                        bool(record.get("accepted"))
                        for record in records_by_id.values()
                    )
                    print(
                        f"{target}: completed {completed}/{len(candidates)}; "
                        f"accepted={accepted}",
                        flush=True,
                    )
        records = [records_by_id[candidate.candidate_id] for candidate in candidates]
        _write_jsonl(output_dir / f"{target}-structural.jsonl", records)
        statuses = Counter(
            "accepted"
            if record.get("accepted")
            else record.get("rejection_reasons", ["unknown"])[0]
            for record in records
        )
        summary = {
            "target": target,
            "candidates": len(candidates),
            "accepted": sum(bool(record.get("accepted")) for record in records),
            "status_counts": dict(sorted(statuses.items())),
            "structural_results": str(
                output_dir / f"{target}-structural.jsonl"
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    overall = {
        "schema_version": 1,
        "arguments": vars(args),
        "targets": summaries,
        "output": str(output_dir),
    }
    _write_json(output_dir / "summary.json", overall)
    print(json.dumps(overall, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
