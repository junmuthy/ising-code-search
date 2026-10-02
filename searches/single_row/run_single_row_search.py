#!/usr/bin/env python3
"""Run the ``[[32,4,6]]`` GL(2,2) x C4 single-row search."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import tempfile
from collections import Counter
from typing import Any

from gala_search.single_row import (
    SINGLE_ROW_SCHEMA_VERSION,
    TranslationFold,
    add_automatic_css_relation,
    analyze_seed_fold,
    iter_structured_gl_folds,
    polynomial_constraint_matrix,
    search_seed_fold_witnesses,
    seed_constraint_matrix,
    solve_constraint_space,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="gl2-c4-l4-j2-seed260827")
    parser.add_argument("--seed", type=int, default=260827)
    parser.add_argument("--trials-per-fold", type=int, default=20)
    parser.add_argument("--target-witnesses", type=int, default=100)
    parser.add_argument("--polynomial-witnesses", type=int, default=20)
    parser.add_argument("--witness-index", type=int, action="append")
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--enumeration-limit", type=int, default=20)
    parser.add_argument("--random-samples", type=int, default=20_000)
    parser.add_argument("--maximum-hits", type=int, default=3)
    parser.add_argument("--q-mode", choices=("monomial", "all"), default="monomial")
    parser.add_argument(
        "--ansatz",
        choices=("automatic", "automatic-relaxed", "independent"),
        default="automatic",
    )
    parser.add_argument("--seed-only", action="store_true")
    return parser.parse_args()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: pathlib.Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for value in values:
            output.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def _fold(record: dict[str, Any]) -> TranslationFold:
    return TranslationFold(tuple(record["fibre_images"]), tuple(record["fibre_shifts"]))


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_DIR / "results" / "single-row" / "n32-k4" / args.run_name
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    seed_result = search_seed_fold_witnesses(
        seed=args.seed,
        trials_per_fold=args.trials_per_fold,
        target_witnesses=args.target_witnesses,
    )
    witnesses = [
        {
            "schema_version": SINGLE_ROW_SCHEMA_VERSION,
            "witness_index": index,
            **record,
        }
        for index, record in enumerate(seed_result.pop("witnesses"))
    ]
    _write_jsonl(output_dir / "seed-witnesses.jsonl", witnesses)
    print(
        json.dumps({"stage": "seed-fold", **seed_result, "witnesses": len(witnesses)}),
        flush=True,
    )

    attempts: list[dict[str, Any]] = []
    if not args.seed_only:
        if args.witness_index:
            requested = set(args.witness_index)
            selected = [item for item in witnesses if item["witness_index"] in requested]
        else:
            selected = witnesses[: args.polynomial_witnesses]
        check_folds = tuple(iter_structured_gl_folds(2))
        if args.ansatz in {"automatic", "automatic-relaxed"}:
            q_values = (
                (1, 2, 4, 8)
                if args.q_mode == "monomial"
                else tuple(range(1, 16))
            )
            relations = ((1, 0), (0, 1), (1, 1))
        else:
            q_values = (0,)
            relations = ((0, 0),)
        stop = False
        for witness in selected:
            data_fold = _fold(witness["fold"])
            if args.ansatz == "automatic-relaxed":
                base = seed_constraint_matrix(witness["seed_support"])
                base_by_check_fold = {check_fold: base for check_fold in check_folds}
                active_check_folds = check_folds[:1]
                alternative_folds = tuple(
                    fold
                    for fold in iter_structured_gl_folds(4)
                    if analyze_seed_fold(witness["seed_support"], fold)[
                        "zx_pairing_rank"
                    ]
                    == 4
                )
            else:
                base_by_check_fold = {
                    check_fold: polynomial_constraint_matrix(
                        witness["seed_support"],
                        data_fold=data_fold,
                        check_fold=check_fold,
                    )
                    for check_fold in check_folds
                }
                active_check_folds = check_folds
                alternative_folds = ()
            for check_fold, q_bits, relation in itertools.product(
                active_check_folds, q_values, relations
            ):
                constraints = base_by_check_fold[check_fold]
                if args.ansatz in {"automatic", "automatic-relaxed"}:
                    constraints = add_automatic_css_relation(
                        constraints,
                        q_bits=q_bits,
                        a=relation[0],
                        b=relation[1],
                    )
                synthesis = solve_constraint_space(
                    constraints,
                    support=witness["seed_support"],
                    data_fold=data_fold,
                    check_fold=check_fold,
                    maximum_check_weight=args.maximum_check_weight,
                    enumeration_limit=args.enumeration_limit,
                    random_samples=args.random_samples,
                    random_seed=(
                        args.seed
                        + 1009 * witness["witness_index"]
                        + 101 * q_bits
                        + 7 * relation[0]
                        + relation[1]
                    ),
                    maximum_hits=args.maximum_hits,
                    require_exact_forward=args.ansatz != "automatic-relaxed",
                    alternative_data_folds=alternative_folds,
                )
                attempt = {
                    "schema_version": SINGLE_ROW_SCHEMA_VERSION,
                    "witness_index": witness["witness_index"],
                    "seed_support": witness["seed_support"],
                    "data_fold": witness["fold"],
                    "check_fold": check_fold.to_dict(),
                    "ansatz": {
                        "family": (
                            "automatic_css"
                            if args.ansatz == "automatic"
                            else "automatic_css_rowspace_fold"
                            if args.ansatz == "automatic-relaxed"
                            else "independent"
                        ),
                        "q_bits": q_bits,
                        "a": relation[0],
                        "b": relation[1],
                    },
                    "synthesis": synthesis,
                }
                attempts.append(attempt)
                print(
                    json.dumps(
                        {
                            "stage": "polynomial",
                            "attempt": len(attempts),
                            "witness_index": witness["witness_index"],
                            "q_bits": q_bits,
                            "relation": relation,
                            "status": synthesis["status"],
                            "nullity": synthesis["constraint_nullity"],
                            "capacity": synthesis["rank_capacity_x"],
                            "hits": len(synthesis["hits"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if any(hit["analysis"]["accepted"] for hit in synthesis["hits"]):
                    stop = True
                    break
            if stop:
                break
        _write_jsonl(output_dir / "polynomial-attempts.jsonl", attempts)

    status_counts = Counter(item["synthesis"]["status"] for item in attempts)
    accepted = [
        hit
        for attempt in attempts
        for hit in attempt["synthesis"]["hits"]
        if hit["analysis"]["accepted"]
    ]
    summary = {
        "schema_version": SINGLE_ROW_SCHEMA_VERSION,
        "target": {
            "parameters": "[[32,4,6]]",
            "translation_group": "C4",
            "representation": "GL(2,2) ~= S3",
            "protograph": {"L": 4, "J": 2},
        },
        "arguments": vars(args),
        "seed_stage": seed_result,
        "witnesses": len(witnesses),
        "attempts": len(attempts),
        "status_counts": dict(status_counts),
        "accepted_candidates": len(accepted),
        "output": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
