#!/usr/bin/env python3
"""Search sparse F generators and solve the n=32 CSS equations for G."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import tempfile
from typing import Any

from gala_search.single_row import (
    SINGLE_ROW_SCHEMA_VERSION,
    analyze_seed_fold,
    iter_structured_gl_folds,
    search_fixed_f_solved_g,
    search_seed_fold_witnesses,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="gl2-c4-l4-j2-solved-g")
    parser.add_argument("--seed", type=int, default=260827)
    parser.add_argument("--trials-per-fold", type=int, default=20)
    parser.add_argument("--target-witnesses", type=int, default=100)
    parser.add_argument("--witness-index", type=int, action="append")
    parser.add_argument("--witness-start", type=int)
    parser.add_argument("--witness-end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--f-trials", type=int, default=10_000)
    parser.add_argument("--g-samples", type=int, default=32)
    parser.add_argument("--f-weight-min", type=int, default=4)
    parser.add_argument("--f-weight-max", type=int, default=10)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--maximum-hits", type=int, default=3)
    return parser.parse_args()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _search_witness(
    index: int, witness: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    all_folds = tuple(iter_structured_gl_folds(4))
    compatible = tuple(
        fold
        for fold in all_folds
        if analyze_seed_fold(witness["seed_support"], fold)["zx_pairing_rank"] == 4
    )
    search = search_fixed_f_solved_g(
        support=witness["seed_support"],
        data_folds=compatible,
        f_trials=arguments["f_trials"],
        g_samples=arguments["g_samples"],
        f_weight_min=arguments["f_weight_min"],
        f_weight_max=arguments["f_weight_max"],
        maximum_check_weight=arguments["maximum_check_weight"],
        random_seed=arguments["seed"] + 1009 * index,
        maximum_hits=arguments["maximum_hits"],
    )
    return {
        "schema_version": SINGLE_ROW_SCHEMA_VERSION,
        "witness_index": index,
        "seed_support": witness["seed_support"],
        "search": search,
    }


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_DIR / "results" / "single-row" / "n32-k4" / args.run_name
    if output_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_result = search_seed_fold_witnesses(
        seed=args.seed,
        trials_per_fold=args.trials_per_fold,
        target_witnesses=args.target_witnesses,
    )
    witnesses = seed_result.pop("witnesses")
    if args.witness_index:
        selected_indices = sorted(set(args.witness_index))
    elif args.witness_start is not None or args.witness_end is not None:
        start = 0 if args.witness_start is None else args.witness_start
        end = len(witnesses) if args.witness_end is None else args.witness_end
        selected_indices = list(range(start, end))
    else:
        selected_indices = list(range(min(10, len(witnesses))))
    if not selected_indices or min(selected_indices) < 0 or max(selected_indices) >= len(witnesses):
        raise SystemExit(
            f"witness selection must be nonempty and lie in [0, {len(witnesses)})"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    records_by_index: dict[int, dict[str, Any]] = {}
    for index in selected_indices:
        path = output_dir / f"witness-{index}.json"
        if args.resume and path.exists():
            records_by_index[index] = json.loads(path.read_text())
    pending = [index for index in selected_indices if index not in records_by_index]
    arguments = vars(args)

    def save(record: dict[str, Any]) -> None:
        index = record["witness_index"]
        records_by_index[index] = record
        _write_json(output_dir / f"witness-{index}.json", record)
        search = record["search"]
        print(json.dumps({"witness_index": index, **search}, sort_keys=True), flush=True)

    if args.workers == 1:
        for index in pending:
            save(_search_witness(index, witnesses[index], arguments))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_search_witness, index, witnesses[index], arguments): index
                for index in pending
            }
            for future in concurrent.futures.as_completed(futures):
                save(future.result())

    records = [records_by_index[index] for index in sorted(records_by_index)]
    summary = {
        "schema_version": SINGLE_ROW_SCHEMA_VERSION,
        "target": "[[32,4,6]]",
        "ansatz": "sparse F, affine CSS/kernel solve for G",
        "arguments": vars(args),
        "seed_stage": seed_result,
        "records": len(records),
        "accepted_candidates": sum(
            hit["analysis"]["accepted"]
            for record in records
            for hit in record["search"]["hits"]
        ),
        "output": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
