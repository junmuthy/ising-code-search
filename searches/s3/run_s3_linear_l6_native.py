#!/usr/bin/env python3
"""Native L=6, J=1 polynomial search in the faithful 2D S3 lift."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import pathlib
import random
import tempfile
import traceback
from collections import Counter
from typing import Any, Iterable

import numpy as np

from gala_search.s3_ising import ProductMonomial, find_logical_up_to_weight_four
from gala_search.s3_linear import (
    build_linear_fold_code,
    is_zx_fold,
    linear_bottom_support_generates,
    linear_fold_active_orthogonality_data,
    linear_fold_permutation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
PATTERNS = ((1, 1, 2), (1, 2, 2), (2, 2, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-per-pattern", type=int, default=100_000)
    parser.add_argument("--target-per-pattern", type=int, default=200)
    parser.add_argument("--max-check-weight", type=int, default=16)
    parser.add_argument("--active-rows", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=606060)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def _serial(entries: tuple[tuple[ProductMonomial, ...], ...]) -> list[list[dict[str, int]]]:
    return [
        [{"top": term.top, "x": term.x, "y": term.y} for term in entry]
        for entry in entries
    ]


def _candidate_id(active_rows: int, pattern: tuple[int, ...], entries: object) -> str:
    digest = hashlib.sha256(repr((active_rows, pattern, entries)).encode()).hexdigest()[:16]
    return f"s3-linear-l6-j{active_rows}-w{2 * sum(pattern)}-{digest}"


def _sample_pattern(
    pattern: tuple[int, ...],
    *,
    raw_limit: int,
    target: int,
    seed: int,
    active_rows: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    terms = [
        ProductMonomial(top, xx, yy)
        for top, xx, yy in itertools.product(range(6), range(8), range(4))
    ]
    anchor = ProductMonomial(0, 0, 0)
    non_anchor = [term for term in terms if term != anchor]
    seen: set[tuple[tuple[ProductMonomial, ...], ...]] = set()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for _ in range(raw_limit):
        entries: list[tuple[ProductMonomial, ...]] = []
        for index, weight in enumerate(pattern):
            if index == 0:
                entries.append(tuple(sorted((anchor, *rng.sample(non_anchor, weight - 1)))))
            else:
                entries.append(tuple(sorted(rng.sample(terms, weight))))
        normalized = tuple(entries)
        if normalized in seen:
            counts["duplicate"] += 1
            continue
        seen.add(normalized)
        counts["raw"] += 1
        orthogonality = linear_fold_active_orthogonality_data(
            normalized, active_rows=active_rows, shift=0
        )
        if not orthogonality["active_offsets_zero"]:
            counts["active_orthogonality_rejected"] += 1
            continue
        counts["active_orthogonality"] += 1
        if orthogonality["latent_offsets"] and not orthogonality["some_latent_offset_nonzero"]:
            counts["latent_zero_rejected"] += 1
            continue
        if not linear_bottom_support_generates(normalized):
            counts["bottom_disconnected_rejected"] += 1
            continue
        counts["algebraic_survivors"] += 1
        records.append(
            {
                "candidate_id": _candidate_id(active_rows, pattern, normalized),
                "pilot": f"l6-j{active_rows}-w{2 * sum(pattern)}",
                "half_blocks": 3,
                "active_rows": active_rows,
                "target_grids": 2,
                "shift": 0,
                "term_weight_pattern": list(pattern),
                "entries": _serial(normalized),
                "orthogonality": orthogonality,
            }
        )
        if len(records) >= target:
            break
    return records, counts


def _entries(record: dict[str, Any]) -> tuple[tuple[ProductMonomial, ...], ...]:
    return tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["entries"]
    )


def _analyze(record: dict[str, Any], max_check_weight: int) -> dict[str, Any]:
    try:
        code = build_linear_fold_code(
            _entries(record), active_rows=record["active_rows"], shift=0
        )
        fold = linear_fold_permutation(
            half_blocks=3, active_rows=record["active_rows"], shift=0
        )
        hx = np.asarray(code.matrix_x, dtype=np.uint8)
        hz = np.asarray(code.matrix_z, dtype=np.uint8)
        row_x = np.count_nonzero(hx, axis=1)
        row_z = np.count_nonzero(hz, axis=1)
        column_x = np.count_nonzero(hx, axis=0)
        column_z = np.count_nonzero(hz, axis=0)
        checks = {
            "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
            "expected_ZX_fold": is_zx_fold(code, fold),
            "dimension_holds_two_grids": code.dimension >= 64,
            f"maximum_check_weight_at_most_{max_check_weight}": bool(
                row_x.max(initial=0) <= max_check_weight
                and row_z.max(initial=0) <= max_check_weight
            ),
        }
        reasons = [name for name, passed in checks.items() if not passed]
        low_weight = None
        if not reasons:
            low_weight = find_logical_up_to_weight_four(code)
            if low_weight is not None:
                reasons.append("logical_up_to_weight_four")
        return {
            **record,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "checks": checks,
            "n": code.num_qubits,
            "k": code.dimension,
            "rank_x": code.code_x.rank,
            "rank_z": code.code_z.rank,
            "row_weights_x": sorted(set(map(int, row_x))),
            "row_weights_z": sorted(set(map(int, row_z))),
            "maximum_check_weight": int(max(row_x.max(initial=0), row_z.max(initial=0))),
            "column_degrees_x": sorted(set(map(int, column_x))),
            "column_degrees_z": sorted(set(map(int, column_z))),
            "even_syndrome_parity": bool(
                np.all(column_x % 2 == 0) and np.all(column_z % 2 == 0)
            ),
            "logical_up_to_weight_four": low_weight,
        }
    except Exception as error:
        return {
            **record,
            "accepted": False,
            "rejection_reasons": ["worker_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _write_jsonl(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary = output.with_suffix(".summary.json")
    for path in (output, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    sampled: list[dict[str, Any]] = []
    algebraic_counts: dict[str, dict[str, int]] = {}
    for index, pattern in enumerate(PATTERNS):
        records, counts = _sample_pattern(
            pattern,
            raw_limit=args.raw_per_pattern,
            target=args.target_per_pattern,
            seed=args.seed + index,
            active_rows=args.active_rows,
        )
        sampled.extend(records)
        name = f"l6-j{args.active_rows}-w{2 * sum(pattern)}"
        algebraic_counts[name] = dict(counts)
        print(json.dumps({"stage": "algebraic", "pilot": name, "counts": counts}), flush=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_analyze, record, args.max_check_weight) for record in sampled]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0 or len(results) == len(futures):
                print(
                    json.dumps(
                        {
                            "stage": "lifted",
                            "completed": len(results),
                            "total": len(futures),
                            "accepted": sum(bool(item.get("accepted")) for item in results),
                        }
                    ),
                    flush=True,
                )
    results.sort(key=lambda item: (item["pilot"], item["candidate_id"]))
    report = {
        "seed": args.seed,
        "target_per_pattern": args.target_per_pattern,
        "raw_per_pattern": args.raw_per_pattern,
        "max_check_weight": args.max_check_weight,
        "active_rows": args.active_rows,
        "algebraic_counts": algebraic_counts,
        "status_counts": {
            pilot: Counter(
                "survivor"
                if item.get("accepted")
                else item.get("rejection_reasons", ["unknown"])[0]
                for item in results
                if item["pilot"] == pilot
            )
            for pilot in sorted(set(item["pilot"] for item in results))
        },
    }
    _write_jsonl(output, results)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary)
    print(json.dumps({"output": str(output), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
