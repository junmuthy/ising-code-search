#!/usr/bin/env python3
"""Run the seed-first ``[[256,32,>=6]]`` half-checkerboard pilot."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import tempfile
from collections import Counter
from typing import Any

from gala_search.half_grid import (
    HALF_GRID_SCHEMA_VERSION,
    StructuredGLFold,
    analyze_polynomial_solution,
    add_automatic_css_relation,
    deserialize_entries,
    polynomial_constraint_matrix,
    search_fold_seed_witnesses,
    solve_sparse_polynomial_constraints,
    standard_fold_control,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="gl2-l4-j2-seed260826")
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--logical-weight", type=int, choices=(6, 7), default=6)
    parser.add_argument("--trials-per-fold", type=int, default=20)
    parser.add_argument("--standard-trials", type=int, default=2_000)
    parser.add_argument("--target-witnesses", type=int, default=100)
    parser.add_argument("--polynomial-witnesses", type=int, default=2)
    parser.add_argument(
        "--witness-index",
        type=int,
        action="append",
        help="synthesize only selected saved witness indices (repeatable)",
    )
    parser.add_argument(
        "--check-fold-index",
        type=int,
        choices=range(4),
        action="append",
        help="select check folds 0..3 in lexicographic order (repeatable)",
    )
    parser.add_argument("--maximum-terms", type=int, default=16)
    parser.add_argument("--milp-seconds", type=float, default=10)
    parser.add_argument("--rank-probe-trials", type=int, default=0)
    parser.add_argument("--rank-probe-restarts", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--css-relation",
        choices=("independent", "identity", "swap"),
        default="independent",
        help="optionally impose G=F or (G0,G1)=(F1,F0) for automatic CSS",
    )
    parser.add_argument(
        "--css-x-shift",
        type=int,
        action="append",
        help="scan a common central x shift in the automatic-CSS relation",
    )
    parser.add_argument(
        "--css-y-shift",
        type=int,
        action="append",
        help="scan a common central y shift in the automatic-CSS relation",
    )
    parser.add_argument(
        "--feasibility-only",
        action="store_true",
        help="find any solution under --maximum-terms instead of minimizing",
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


def _fold(data: dict[str, Any]) -> StructuredGLFold:
    return StructuredGLFold(
        tuple(data["forward_blocks"]), tuple(data["backward_blocks"])
    )


def main() -> None:
    args = parse_args()
    if min(
        args.trials_per_fold,
        args.standard_trials,
        args.target_witnesses,
        args.maximum_terms,
    ) < 1:
        raise SystemExit("all search limits must be positive")
    output_dir = PROJECT_DIR / "results" / "half-grid" / args.run_name
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    standard_control = standard_fold_control(
        seed=args.seed + 1,
        weight=args.logical_weight,
        trials_per_fold=args.standard_trials,
    )
    print(
        json.dumps(
            {"stage": "standard-fold-control", **standard_control},
            sort_keys=True,
        ),
        flush=True,
    )

    seed_result = search_fold_seed_witnesses(
        seed=args.seed,
        weight=args.logical_weight,
        trials_per_fold=args.trials_per_fold,
        target_witnesses=args.target_witnesses,
    )
    seed_records = [
        {
            "schema_version": HALF_GRID_SCHEMA_VERSION,
            "witness_index": index,
            **record,
        }
        for index, record in enumerate(seed_result.pop("witnesses"))
    ]
    _write_jsonl(output_dir / "seed-witnesses.jsonl", seed_records)
    print(
        json.dumps(
            {
                "stage": "seed-fold",
                **seed_result,
                "witnesses": len(seed_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    attempts: list[dict[str, Any]] = []
    if not args.seed_only:
        check_maps = tuple(itertools.permutations(range(2)))
        check_folds = [
            StructuredGLFold(forward, backward)
            for forward in check_maps
            for backward in check_maps
        ]
        if args.check_fold_index:
            check_folds = [check_folds[index] for index in args.check_fold_index]
        if args.witness_index:
            requested = set(args.witness_index)
            selected_witnesses = [
                witness
                for witness in seed_records
                if witness["witness_index"] in requested
            ]
            missing = requested - {
                witness["witness_index"] for witness in selected_witnesses
            }
            if missing:
                raise SystemExit(
                    f"requested witness indices were not found: {sorted(missing)}"
                )
        else:
            selected_witnesses = seed_records[: args.polynomial_witnesses]
        x_shifts = args.css_x_shift or [0]
        y_shifts = args.css_y_shift or [0]
        if args.css_relation == "independent" and (
            any(value % 8 for value in x_shifts)
            or any(value % 4 for value in y_shifts)
        ):
            raise SystemExit("CSS shifts require --css-relation identity or swap")
        for witness in selected_witnesses:
            data_fold = _fold(witness["fold"])
            for check_fold in check_folds:
                base_constraints = polynomial_constraint_matrix(
                    witness["seed_support"],
                    data_fold=data_fold,
                    check_fold=check_fold,
                )
                for x_shift, y_shift in itertools.product(x_shifts, y_shifts):
                    constraints = add_automatic_css_relation(
                        base_constraints,
                        args.css_relation,
                        x_shift=x_shift,
                        y_shift=y_shift,
                    )
                    synthesis = solve_sparse_polynomial_constraints(
                        constraints,
                        maximum_terms=args.maximum_terms,
                        time_limit=args.milp_seconds,
                        minimize_terms=not args.feasibility_only,
                        rank_probe_trials=args.rank_probe_trials,
                        rank_probe_restarts=args.rank_probe_restarts,
                        random_seed=(
                            args.seed
                            + witness["witness_index"]
                            + 101 * (x_shift % 8)
                            + 1009 * (y_shift % 4)
                        ),
                        probe_only=args.probe_only,
                    )
                    attempt: dict[str, Any] = {
                        "schema_version": HALF_GRID_SCHEMA_VERSION,
                        "witness_index": witness["witness_index"],
                        "seed_support": witness["seed_support"],
                        "data_fold": witness["fold"],
                        "check_fold": check_fold.to_dict(),
                        "css_relation": args.css_relation,
                        "css_shift": {"x": x_shift % 8, "y": y_shift % 4},
                        "synthesis": synthesis,
                    }
                    if synthesis["entries"] is not None:
                        try:
                            entries = deserialize_entries(synthesis["entries"])
                            attempt["candidate_analysis"] = analyze_polynomial_solution(
                                entries,
                                seed_support=witness["seed_support"],
                                data_fold=data_fold,
                                check_fold=check_fold,
                            )
                        except Exception as error:
                            attempt["candidate_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                    attempts.append(attempt)
                    print(
                        json.dumps(
                            {
                                "stage": "polynomial",
                                "attempt": len(attempts),
                                "witness_index": witness["witness_index"],
                                "check_fold": check_fold.to_dict(),
                                "css_shift": attempt["css_shift"],
                                "constraint_rank": synthesis["constraint_rank"],
                                "solver_status": synthesis["solver_status"],
                                "objective_terms": synthesis.get("objective_terms"),
                                "accepted_structurally": attempt.get(
                                    "candidate_analysis", {}
                                ).get("accepted_structurally", False),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        _write_jsonl(output_dir / "polynomial-attempts.jsonl", attempts)

    status_counts = Counter(
        "structural_hit"
        if attempt.get("candidate_analysis", {}).get("accepted_structurally")
        else "polynomial_found_but_rejected"
        if attempt.get("synthesis", {}).get("entries") is not None
        else f"solver_status_{attempt.get('synthesis', {}).get('solver_status')}"
        for attempt in attempts
    )
    summary = {
        "schema_version": HALF_GRID_SCHEMA_VERSION,
        "target": {
            "parameters": "[[256,32,>=6]]",
            "translation_group": "C8 x C4",
            "representation": "GL(2,2) ~= S3",
            "protograph": {"L": 4, "J": 2},
            "logical_weight": args.logical_weight,
            "two_checkerboard_total_n": 512,
        },
        "arguments": vars(args),
        "seed_stage": {**seed_result, "witnesses": len(seed_records)},
        "standard_fold_control": standard_control,
        "polynomial_attempts": len(attempts),
        "polynomial_status_counts": dict(status_counts),
        "files": {
            "seed_witnesses": str(output_dir / "seed-witnesses.jsonl"),
            "polynomial_attempts": (
                str(output_dir / "polynomial-attempts.jsonl") if attempts else None
            ),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output": str(output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
