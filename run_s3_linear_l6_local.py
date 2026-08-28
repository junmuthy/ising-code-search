#!/usr/bin/env python3
"""One-monomial neighborhood of a saved native L6 linear-S3 candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import pathlib
import tempfile
from collections import Counter

from gala_search.s3_ising import ProductMonomial
from gala_search.s3_linear import (
    linear_bottom_support_generates,
    linear_fold_active_orthogonality_data,
)
from run_s3_linear_l6_native import _analyze, _serial

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-check-weight", type=int, default=16)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary = output.with_suffix(".summary.json")
    for path in (output, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    source = next(
        record
        for record in map(json.loads, args.input.read_text().splitlines())
        if record["candidate_id"] == args.candidate_id
    )
    entries = tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in source["entries"]
    )
    anchor = ProductMonomial(0, 0, 0)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top, xx, yy in itertools.product(range(6), range(8), range(4))
    ]
    candidates: dict[tuple[tuple[ProductMonomial, ...], ...], dict] = {}
    counts: Counter[str] = Counter()
    for entry_index, entry in enumerate(entries):
        for term_index, old_term in enumerate(entry):
            if old_term == anchor:
                continue
            for new_term in all_terms:
                if new_term == old_term or new_term in entry:
                    continue
                mutated = [list(item) for item in entries]
                mutated[entry_index][term_index] = new_term
                normalized = tuple(tuple(sorted(item)) for item in mutated)
                counts["raw"] += 1
                if normalized in candidates:
                    counts["duplicate"] += 1
                    continue
                orthogonality = linear_fold_active_orthogonality_data(
                    normalized, active_rows=2, shift=0
                )
                if not orthogonality["active_offsets_zero"]:
                    counts["active_orthogonality_rejected"] += 1
                    continue
                if not linear_bottom_support_generates(normalized):
                    counts["bottom_disconnected_rejected"] += 1
                    continue
                counts["algebraic_survivors"] += 1
                digest = hashlib.sha256(repr(normalized).encode()).hexdigest()[:16]
                candidates[normalized] = {
                    "candidate_id": f"s3-linear-l6-j2-w8-local-{digest}",
                    "parent_candidate_id": args.candidate_id,
                    "pilot": "l6-j2-w8-local",
                    "half_blocks": 3,
                    "active_rows": 2,
                    "target_grids": 2,
                    "shift": 0,
                    "term_weight_pattern": [1, 1, 2],
                    "entries": _serial(normalized),
                    "orthogonality": orthogonality,
                }
    records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_analyze, record, args.max_check_weight)
            for record in candidates.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["candidate_id"])
    report = {
        "parent_candidate_id": args.candidate_id,
        "max_check_weight": args.max_check_weight,
        "algebraic_counts": counts,
        "structural_status_counts": Counter(
            "survivor"
            if record.get("accepted")
            else record.get("rejection_reasons", ["unknown"])[0]
            for record in records
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary)
    print(json.dumps({"output": str(output), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
