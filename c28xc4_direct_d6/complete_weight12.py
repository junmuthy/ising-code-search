#!/usr/bin/env python3
"""Exactly complete mixed-sector stabilizer presentations through weight 12."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
import time
from typing import Any

import cvxpy as cp
import numpy as np

from algebra import (
    GROUP_ORDER,
    LONG_ORDER,
    QUOTIENT_Y_ORDER,
    build_stabilizer_basis,
    gf2_rref,
    gf8_contains,
    physical_seed_from_sparse,
    quotient_annihilator_basis,
)
from search import build_sparse_catalog, enumerate_ideal_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--minimum-initial-rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def gf2_nullspace(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    rref, pivots = gf2_rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for row, free_column in enumerate(free):
        basis[row, free_column] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            basis[row, pivot_column] = rref[pivot_row, free_column]
    return basis


def translate(vector: np.ndarray[Any, Any], shift_x: int, shift_y: int) -> np.ndarray[Any, Any]:
    grid = np.asarray(vector, dtype=np.uint8).reshape(LONG_ORDER, QUOTIENT_Y_ORDER)
    return np.roll(grid, (shift_x, shift_y), axis=(0, 1)).reshape(-1)


def translated_rows(seed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [
            translate(seed, shift_x, shift_y)
            for shift_x in range(LONG_ORDER)
            for shift_y in range(QUOTIENT_Y_ORDER)
        ],
        dtype=np.uint8,
    )


def canonical_seed(seed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    rows = translated_rows(seed)
    keys = [np.packbits(row, bitorder="little").tobytes() for row in rows]
    return rows[min(range(len(rows)), key=lambda index: keys[index])]


def deduplicate_seeds(seeds: list[np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
    records: dict[bytes, np.ndarray[Any, Any]] = {}
    for seed in seeds:
        canonical = canonical_seed(seed)
        key = np.packbits(canonical, bitorder="little").tobytes()
        records.setdefault(key, canonical)
    return list(records.values())


def span_from_seeds(seeds: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    if not seeds:
        return np.zeros((0, GROUP_ORDER), dtype=np.uint8)
    return gf2_rref(np.vstack([translated_rows(seed) for seed in seeds]))[0]


def displacement_subgroup(seeds: list[np.ndarray[Any, Any]]) -> set[tuple[int, int]]:
    generators: set[tuple[int, int]] = set()
    for seed in seeds:
        support = [divmod(int(index), QUOTIENT_Y_ORDER) for index in np.flatnonzero(seed)]
        if not support:
            continue
        anchor_x, anchor_y = support[0]
        generators.update(
            ((xx - anchor_x) % LONG_ORDER, (yy - anchor_y) % QUOTIENT_Y_ORDER)
            for xx, yy in support[1:]
        )
    signed = generators | {
        (-xx % LONG_ORDER, -yy % QUOTIENT_Y_ORDER) for xx, yy in generators
    }
    reached = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        xx, yy = frontier.pop()
        for dx, dy in signed:
            target = ((xx + dx) % LONG_ORDER, (yy + dy) % QUOTIENT_Y_ORDER)
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return reached


def _base_word_constraints(stabilizer: np.ndarray[Any, Any]) -> tuple[Any, Any, list[Any]]:
    coefficients = cp.Variable(len(stabilizer), boolean=True)
    word = cp.Variable(stabilizer.shape[1], boolean=True)
    slack = cp.Variable(stabilizer.shape[1], integer=True)
    constraints = [
        stabilizer.T.astype(int) @ coefficients == word + 2 * slack,
        slack >= 0,
        slack <= stabilizer.sum(axis=0).astype(int) // 2,
        word[0] == 1,
        cp.sum(word) <= 12,
    ]
    return coefficients, word, constraints


def find_word_outside_span(
    stabilizer: np.ndarray[Any, Any], span: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any] | None, str, float]:
    stabilizer, pivots = gf2_rref(stabilizer)
    span_coefficients = span[:, pivots] if len(span) else np.zeros((0, len(stabilizer)), dtype=np.uint8)
    functionals = gf2_nullspace(span_coefficients)
    if not len(functionals):
        return None, "full", 0.0
    coefficients, word, constraints = _base_word_constraints(stabilizer)
    outside_bits = cp.Variable(len(functionals), boolean=True)
    outside_slack = cp.Variable(len(functionals), integer=True)
    constraints.extend(
        [
            functionals.astype(int) @ coefficients == outside_bits + 2 * outside_slack,
            outside_slack >= 0,
            outside_slack <= functionals.sum(axis=1).astype(int) // 2,
            cp.sum(outside_bits) >= 1,
        ]
    )
    problem = cp.Problem(cp.Minimize(cp.sum(word)), constraints)
    started = time.perf_counter()
    result = problem.solve(solver="HIGHS")
    elapsed = time.perf_counter() - started
    if problem.status == cp.INFEASIBLE:
        return None, str(problem.status), elapsed
    if problem.status != cp.OPTIMAL or word.value is None or not np.isfinite(result):
        raise RuntimeError(f"unexpected span-completion status: {problem.status}")
    return np.rint(word.value).astype(np.uint8), str(problem.status), elapsed


def find_connecting_word(
    stabilizer: np.ndarray[Any, Any], subgroup: set[tuple[int, int]]
) -> tuple[np.ndarray[Any, Any] | None, str, float]:
    outside = [
        xx * QUOTIENT_Y_ORDER + yy
        for xx in range(LONG_ORDER)
        for yy in range(QUOTIENT_Y_ORDER)
        if (xx, yy) not in subgroup
    ]
    if not outside:
        return None, "connected", 0.0
    _coefficients, word, constraints = _base_word_constraints(gf2_rref(stabilizer)[0])
    constraints.append(cp.sum(word[outside]) >= 1)
    problem = cp.Problem(cp.Minimize(cp.sum(word)), constraints)
    started = time.perf_counter()
    result = problem.solve(solver="HIGHS")
    elapsed = time.perf_counter() - started
    if problem.status == cp.INFEASIBLE:
        return None, str(problem.status), elapsed
    if problem.status != cp.OPTIMAL or word.value is None or not np.isfinite(result):
        raise RuntimeError(f"unexpected connectivity-completion status: {problem.status}")
    return np.rint(word.value).astype(np.uint8), str(problem.status), elapsed


def seed_record(seed: np.ndarray[Any, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "weight": int(seed.sum()),
        "support": [
            list(divmod(int(index), QUOTIENT_Y_ORDER)) for index in np.flatnonzero(seed)
        ],
        "orbit_rank": len(gf2_rref(translated_rows(seed))[0]),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    output_jsonl = args.output / "completion.jsonl"
    completed_ids: set[str] = set()
    if args.resume and output_jsonl.exists():
        completed_ids = {json.loads(line)["candidate_id"] for line in output_jsonl.open()}

    source_records = {
        record["candidate_id"]: record
        for record in (json.loads(line) for line in (args.source / "candidates.jsonl").open())
        if record["distance"] == 6
    }
    catalog = build_sparse_catalog()
    candidates, _counts = enumerate_ideal_candidates(catalog)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    work = []
    for candidate_id in source_records:
        candidate = candidate_map[candidate_id]
        ideal = candidate.ideal_basis
        annihilator = quotient_annihilator_basis(list(ideal))
        stabilizer = build_stabilizer_basis(ideal, annihilator)
        seeds = [row.copy() for row in stabilizer if int(row.sum()) <= 12]
        seeds.extend(
            physical_seed_from_sparse(entry.polynomial, dagger=False)
            for entry in catalog
            if gf8_contains(ideal, entry.polynomial.vector)
        )
        seeds.extend(
            physical_seed_from_sparse(entry.polynomial, dagger=True)
            for entry in catalog
            if gf8_contains(annihilator, entry.polynomial.vector)
        )
        seeds = deduplicate_seeds(seeds)
        span = span_from_seeds(seeds)
        if len(span) >= args.minimum_initial_rank:
            work.append((len(span), candidate_id, stabilizer, seeds, span))
    work.sort(key=lambda item: (-item[0], item[1]))
    if args.maximum is not None:
        work = work[: args.maximum]
    atomic_json(
        args.output / "catalog-summary.json",
        {
            "source": str(args.source),
            "distance_six_candidates": len(source_records),
            "selected_candidates": len(work),
            "already_completed": len(completed_ids),
            "minimum_initial_rank": args.minimum_initial_rank,
            "maximum": args.maximum,
            "weight_ceiling": 12,
            "method": "exact mixed-sector MILP span and connectivity completion",
        },
    )
    started = time.perf_counter()
    newly_completed = 0
    for position, (initial_rank, candidate_id, stabilizer, seeds, span) in enumerate(work, start=1):
        if candidate_id in completed_ids:
            continue
        print(
            json.dumps(
                {
                    "stage": "candidate_start",
                    "position": position,
                    "total": len(work),
                    "candidate_id": candidate_id,
                    "initial_rank": initial_rank,
                    "initial_subgroup_size": len(displacement_subgroup(seeds)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        added: list[tuple[np.ndarray[Any, Any], str]] = []
        solves: list[dict[str, Any]] = []
        while len(span) < len(stabilizer):
            word, status, elapsed = find_word_outside_span(stabilizer, span)
            solve = {
                "kind": "span",
                "rank_before": len(span),
                "status": status,
                "elapsed_seconds": round(elapsed, 6),
                "word_weight": None if word is None else int(word.sum()),
            }
            solves.append(solve)
            print(json.dumps({"candidate_id": candidate_id, **solve}, sort_keys=True), flush=True)
            if word is None:
                break
            canonical = canonical_seed(word)
            seeds = deduplicate_seeds([*seeds, canonical])
            added.append((canonical, "exact_span_completion"))
            span = span_from_seeds(seeds)

        while len(span) == len(stabilizer):
            subgroup = displacement_subgroup(seeds)
            if len(subgroup) == GROUP_ORDER:
                break
            word, status, elapsed = find_connecting_word(stabilizer, subgroup)
            solve = {
                "kind": "connectivity",
                "subgroup_size_before": len(subgroup),
                "status": status,
                "elapsed_seconds": round(elapsed, 6),
                "word_weight": None if word is None else int(word.sum()),
            }
            solves.append(solve)
            print(json.dumps({"candidate_id": candidate_id, **solve}, sort_keys=True), flush=True)
            if word is None:
                break
            canonical = canonical_seed(word)
            seeds = deduplicate_seeds([*seeds, canonical])
            added.append((canonical, "exact_connectivity_completion"))

        final_subgroup = displacement_subgroup(seeds)
        record = {
            "candidate_id": candidate_id,
            "initial_rank": initial_rank,
            "final_rank": len(span),
            "complete_weight_12_span": len(span) == len(stabilizer),
            "final_displacement_subgroup_size": len(final_subgroup),
            "connected_weight_12_presentation": len(span) == len(stabilizer) and len(final_subgroup) == GROUP_ORDER,
            "initial_seed_count": len(seeds) - len(added),
            "added_seeds": [seed_record(seed, source) for seed, source in added],
            "solves": solves,
        }
        append_jsonl(output_jsonl, record)
        newly_completed += 1
        progress = {
            "processed_this_run": newly_completed,
            "selected_candidates": len(work),
            "last_candidate_id": candidate_id,
            "last_final_rank": len(span),
            "last_subgroup_size": len(final_subgroup),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_json(args.output / "progress.json", progress)
        print(json.dumps({"stage": "candidate_complete", **record}, sort_keys=True), flush=True)

    records = [json.loads(line) for line in output_jsonl.open()] if output_jsonl.exists() else []
    summary = {
        "complete": len(records) == len(work),
        "records": len(records),
        "selected_candidates": len(work),
        "complete_weight_12_span": sum(record["complete_weight_12_span"] for record in records),
        "connected_weight_12_presentation": sum(record["connected_weight_12_presentation"] for record in records),
        "final_rank_histogram": {
            str(rank): sum(record["final_rank"] == rank for record in records)
            for rank in sorted({record["final_rank"] for record in records})
        },
        "final_subgroup_histogram": {
            str(size): sum(record["final_displacement_subgroup_size"] == size for record in records)
            for size in sorted({record["final_displacement_subgroup_size"] for record in records})
        },
        "total_seconds_this_run": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
