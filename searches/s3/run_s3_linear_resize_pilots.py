#!/usr/bin/env python3
"""Resize saved L12 polynomials into fast L10/L14/L16 linear-S3 pilots."""

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
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

from gala_search.s3_ising import ProductMonomial, find_logical_up_to_weight_four
from gala_search.s3_linear import (
    BLOCK_SIZE,
    build_linear_fold_code,
    is_zx_fold,
    linear_bottom_support_generates,
    linear_fold_active_orthogonality_data,
    linear_fold_permutation,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Pilot:
    name: str
    half_blocks: int
    active_rows: int
    target_grids: int
    resize: str


PILOTS = (
    Pilot("l10-j2-delete-one", 5, 2, 3, "delete-one"),
    Pilot("l14-j3-insert-one-zero", 7, 3, 4, "insert-one-zero"),
    Pilot("l16-j3-insert-two-zero", 8, 3, 5, "insert-two-zero"),
    Pilot("l16-j4-insert-two-zero", 8, 4, 5, "insert-two-zero"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--target-per-pilot", type=int, default=100)
    parser.add_argument("--max-check-weight", type=int, default=16)
    parser.add_argument("--seed", type=int, default=220222)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    return parser.parse_args()


def _read_sources(path: pathlib.Path) -> list[tuple[tuple[ProductMonomial, ...], ...]]:
    sources: set[tuple[tuple[ProductMonomial, ...], ...]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        entries = tuple(
            tuple(ProductMonomial(**term) for term in entry)
            for entry in record["entries"]
        )
        if len(entries) == 6:
            sources.add(entries)
    return sorted(sources)


def _resizes(
    entries: tuple[tuple[ProductMonomial, ...], ...], resize: str
) -> Iterator[tuple[tuple[ProductMonomial, ...], ...]]:
    if resize == "delete-one":
        for deleted in range(6):
            yield entries[:deleted] + entries[deleted + 1 :]
    elif resize == "insert-one-zero":
        for positions in itertools.combinations(range(7), 6):
            output = [()] * 7
            for position, entry in zip(positions, entries):
                output[position] = entry
            yield tuple(output)
    elif resize == "insert-two-zero":
        for positions in itertools.combinations(range(8), 6):
            output = [()] * 8
            for position, entry in zip(positions, entries):
                output[position] = entry
            yield tuple(output)
    else:  # pragma: no cover
        raise ValueError(f"unknown resize operation: {resize}")


def _serial_entries(entries: tuple[tuple[ProductMonomial, ...], ...]) -> list[list[dict[str, int]]]:
    return [
        [{"top": term.top, "x": term.x, "y": term.y} for term in entry]
        for entry in entries
    ]


def _candidate_id(
    pilot: Pilot,
    entries: tuple[tuple[ProductMonomial, ...], ...],
    shift: int,
) -> str:
    digest = hashlib.sha256(repr((pilot, shift, entries)).encode()).hexdigest()[:16]
    return f"s3-linear-{pilot.name}-s{shift}-{digest}"


def _reservoir(
    sources: Iterable[tuple[tuple[ProductMonomial, ...], ...]],
    pilot: Pilot,
    *,
    target: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    reservoir: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen: set[tuple[int, tuple[tuple[ProductMonomial, ...], ...]]] = set()
    for source in sources:
        for entries in _resizes(source, pilot.resize):
            valid_shifts = (0, pilot.half_blocks // 2) if pilot.half_blocks % 2 == 0 else (0,)
            for shift in valid_shifts:
                key = (shift, entries)
                if key in seen:
                    counters["duplicate"] += 1
                    continue
                seen.add(key)
                counters["raw"] += 1
                orthogonality = linear_fold_active_orthogonality_data(
                    entries, active_rows=pilot.active_rows, shift=shift
                )
                if not orthogonality["active_offsets_zero"]:
                    counters["active_orthogonality_rejected"] += 1
                    continue
                counters["active_orthogonality"] += 1
                if not orthogonality["some_latent_offset_nonzero"]:
                    counters["latent_zero_rejected"] += 1
                    continue
                if not linear_bottom_support_generates(entries):
                    counters["bottom_disconnected_rejected"] += 1
                    continue
                counters["algebraic_survivors"] += 1
                record = {
                    "candidate_id": _candidate_id(pilot, entries, shift),
                    "pilot": pilot.name,
                    "half_blocks": pilot.half_blocks,
                    "active_rows": pilot.active_rows,
                    "target_grids": pilot.target_grids,
                    "shift": shift,
                    "entries": _serial_entries(entries),
                    "orthogonality": orthogonality,
                }
                count = counters["algebraic_survivors"]
                if len(reservoir) < target:
                    reservoir.append(record)
                else:
                    replacement = rng.randrange(count)
                    if replacement < target:
                        reservoir[replacement] = record
    return reservoir, counters


def _parse_entries(record: dict[str, Any]) -> tuple[tuple[ProductMonomial, ...], ...]:
    return tuple(
        tuple(ProductMonomial(**term) for term in entry)
        for entry in record["entries"]
    )


def _analyze(record: dict[str, Any], max_check_weight: int) -> dict[str, Any]:
    try:
        entries = _parse_entries(record)
        code = build_linear_fold_code(
            entries,
            active_rows=record["active_rows"],
            shift=record["shift"],
        )
        fold = linear_fold_permutation(
            half_blocks=record["half_blocks"],
            active_rows=record["active_rows"],
            shift=record["shift"],
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
            "dimension_holds_target_grids": code.dimension >= 32 * record["target_grids"],
            f"maximum_check_weight_at_most_{max_check_weight}": bool(
                row_x.max(initial=0) <= max_check_weight
                and row_z.max(initial=0) <= max_check_weight
            ),
        }
        reasons = [name for name, value in checks.items() if not value]
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
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary = output.with_suffix(".summary.json")
    for path in (output, summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if min(args.target_per_pilot, args.workers) < 1:
        raise SystemExit("targets and workers must be positive")
    sources = _read_sources(args.input)
    sampled: list[dict[str, Any]] = []
    algebraic_counts: dict[str, dict[str, int]] = {}
    for index, pilot in enumerate(PILOTS):
        records, counts = _reservoir(
            sources,
            pilot,
            target=args.target_per_pilot,
            rng=random.Random(args.seed + index),
        )
        sampled.extend(records)
        algebraic_counts[pilot.name] = dict(counts)
        print(json.dumps({"stage": "algebraic", "pilot": pilot.name, "counts": counts}), flush=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_analyze, record, args.max_check_weight)
            for record in sampled
        ]
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
    status_counts = {
        pilot.name: Counter(
            "survivor"
            if item.get("accepted")
            else item.get("rejection_reasons", ["unknown"])[0]
            for item in results
            if item["pilot"] == pilot.name
        )
        for pilot in PILOTS
    }
    report = {
        "input": str(args.input),
        "num_unique_sources": len(sources),
        "seed": args.seed,
        "target_per_pilot": args.target_per_pilot,
        "max_check_weight": args.max_check_weight,
        "algebraic_counts": algebraic_counts,
        "lifted_status_counts": status_counts,
    }
    _write_jsonl(output, results)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary)
    print(json.dumps({"output": str(output), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
