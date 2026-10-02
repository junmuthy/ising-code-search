#!/usr/bin/env python3
"""Checkpointed direct n=30 mixed-orbit folded-CSS C4 search."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

import z3

from search import (
    FoldedCSSSolver,
    TARGET_RANK,
    analyze_basis,
    canonical_fold_catalog,
    module_partitions,
    n24_no_go_certificate,
    supports_through_weight,
)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def parse_module_type(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if sum(parsed) != TARGET_RANK or any(not 1 <= item <= 4 for item in parsed):
        raise argparse.ArgumentTypeError(
            f"module type must partition {TARGET_RANK} into parts 1..4"
        )
    if tuple(sorted(parsed, reverse=True)) != parsed:
        raise argparse.ArgumentTypeError("module type must be nonincreasing")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold-index", type=int, action="append")
    parser.add_argument("--module-type", type=parse_module_type, action="append")
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--solver-timeout-seconds", type=float, default=10)
    parser.add_argument("--maximum-solver-timeout-seconds", type=float, default=60)
    parser.add_argument("--maximum-models-per-task", type=int, default=100)
    parser.add_argument("--cuts-per-sector-per-model", type=int, default=64)
    parser.add_argument(
        "--initial-cuts",
        type=pathlib.Path,
        help="Optional JSONL distance cuts to replay before solving one task.",
    )
    parser.add_argument("--stop-on-hit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    tasks_dir = output / "tasks"
    tasks_dir.mkdir()
    started = time.perf_counter()

    proof = n24_no_go_certificate()
    atomic_json(output / "n24-no-go-certificate.json", proof)
    print(json.dumps({"stage": "n24-certificate", **proof}, sort_keys=True), flush=True)

    folds = canonical_fold_catalog()
    selected_fold_indices = args.fold_index or list(range(len(folds)))
    selected_folds = [(index, folds[index]) for index in selected_fold_indices]
    for index, fold in selected_folds:
        atomic_json(output / f"fold-{index:02d}.json", fold.analyze())

    types = args.module_type or list(module_partitions())
    # Test the two natural one-dimension extensions of the n=28 frontier first.
    preferred = {(4, 4, 4, 1): 0, (4, 4, 3, 2): 1}
    types = sorted(
        set(types), key=lambda value: (preferred.get(value, 2), len(value), value)
    )
    tasks = [
        (fold_index, fold, module_type)
        for fold_index, fold in selected_folds
        for module_type in types
    ]
    aggregate: Counter[str] = Counter()
    task_summaries: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []

    def checkpoint(status: str, **extra: Any) -> None:
        record = {
            "status": status,
            "completed_tasks": len(task_summaries),
            "total_tasks": len(tasks),
            "models": aggregate["models"],
            "distance_cuts": aggregate["distance_cuts"],
            "distance_counts": {
                key.removeprefix("distance_"): value
                for key, value in sorted(aggregate.items())
                if key.startswith("distance_") and key != "distance_cuts"
            },
            "exact_unsat_tasks": aggregate["exact_unsat_tasks"],
            "unknown_tasks": aggregate["unknown_tasks"],
            "bounded_tasks": aggregate["bounded_tasks"],
            "hits": len(hits),
            "seconds": round(time.perf_counter() - started, 6),
            **extra,
        }
        atomic_json(output / "progress.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint("starting", low_weight_supports_per_sector=supports_through_weight(5))
    stop = False
    for task_index, (fold_index, fold, module_type) in enumerate(tasks):
        task_name = f"task-{task_index:04d}-f{fold_index}-m{'-'.join(map(str,module_type))}"
        transcript = tasks_dir / f"{task_name}.jsonl"
        cut_file = tasks_dir / f"{task_name}-distance-cuts.jsonl"
        checkpoint(
            "building",
            active_task=task_name,
            fold=fold.name,
            module_type=list(module_type),
        )
        built = time.perf_counter()
        solver = FoldedCSSSolver(
            geometry=fold,
            module_type=module_type,
            maximum_check_weight=args.maximum_check_weight,
            timeout_ms=max(1, int(1000 * args.solver_timeout_seconds)),
        )
        replayed_cuts = 0
        if args.initial_cuts:
            if len(tasks) != 1:
                raise SystemExit("--initial-cuts requires exactly one selected task")
            with args.initial_cuts.open(encoding="utf-8") as cut_input:
                for line in cut_input:
                    record = json.loads(line)
                    solver.add_distance_cut(record["sector"], int(record["operator"]))
                    replayed_cuts += 1
            checkpoint(
                "cuts-replayed",
                active_task=task_name,
                replayed_cuts=replayed_cuts,
            )
        task_record: dict[str, Any] = {
            "task_index": task_index,
            "task_name": task_name,
            "fold_index": fold_index,
            "fold": fold.analyze(),
            "module_type": list(module_type),
            "models": 0,
            "distance_cuts": 0,
            "status": "running",
            "build_seconds": round(time.perf_counter() - built, 6),
            "replayed_cuts": replayed_cuts,
        }
        best_score = (-1, -10**9)
        for model_index in range(args.maximum_models_per_task):
            checkpoint(
                "solving",
                active_task=task_name,
                model_index=model_index,
                active_distance_cuts=solver.distance_cuts,
            )
            active_timeout = args.solver_timeout_seconds
            while True:
                solver.set_timeout(max(1, int(1000 * active_timeout)))
                check_started = time.perf_counter()
                status = solver.check()
                check_seconds = time.perf_counter() - check_started
                if status != z3.unknown or active_timeout >= args.maximum_solver_timeout_seconds:
                    break
                next_timeout = min(
                    args.maximum_solver_timeout_seconds, 2 * active_timeout
                )
                checkpoint(
                    "solver-retry",
                    active_task=task_name,
                    model_index=model_index,
                    previous_timeout_seconds=active_timeout,
                    next_timeout_seconds=next_timeout,
                    reason_unknown=solver.solver.reason_unknown(),
                )
                active_timeout = next_timeout
            if status == z3.unsat:
                task_record.update(
                    {
                        "status": "exact_unsat_after_cegis",
                        "models": model_index,
                        "distance_cuts": solver.distance_cuts,
                        "final_check_seconds": round(check_seconds, 6),
                    }
                )
                aggregate["exact_unsat_tasks"] += 1
                break
            if status == z3.unknown:
                task_record.update(
                    {
                        "status": "solver_unknown",
                        "models": model_index,
                        "distance_cuts": solver.distance_cuts,
                        "reason_unknown": solver.solver.reason_unknown(),
                        "final_check_seconds": round(check_seconds, 6),
                    }
                )
                aggregate["unknown_tasks"] += 1
                break

            basis = solver.concrete_basis()
            analysis = analyze_basis(basis, fold)
            distance = int(analysis["distance"])
            task_record["models"] = model_index + 1
            aggregate["models"] += 1
            aggregate[f"distance_{distance}"] += 1
            model_record = {
                "model_index": model_index,
                "check_seconds": round(check_seconds, 6),
                "distance": distance,
                "x_distance": analysis["x_distance"]["distance"],
                "z_distance": analysis["z_distance"]["distance"],
                "x_low_logicals": analysis["x_distance"]["low_logical_operators_total"],
                "z_low_logicals": analysis["z_distance"]["low_logical_operators_total"],
                "tanner_connected": analysis["tanner_connected"],
                "distance_cuts_before": solver.distance_cuts,
            }
            low_total = int(
                analysis["x_distance"]["low_logical_operators_total"]
                + analysis["z_distance"]["low_logical_operators_total"]
            )
            score = (distance, -low_total)
            if score > best_score:
                best_score = score
                atomic_json(
                    tasks_dir / f"{task_name}-best.json",
                    {
                        "model_index": model_index,
                        "score": list(score),
                        "module_type": list(module_type),
                        "fold_index": fold_index,
                        "analysis": analysis,
                    },
                )
            if distance >= 6 and analysis["tanner_connected"]:
                append_jsonl(transcript, model_record)
                candidate = {
                    "task_name": task_name,
                    "fold_index": fold_index,
                    "module_type": list(module_type),
                    "analysis": analysis,
                }
                hits.append(candidate)
                atomic_json(tasks_dir / f"{task_name}-candidate.json", candidate)
                task_record.update(
                    {
                        "status": "hit",
                        "distance": distance,
                        "distance_cuts": solver.distance_cuts,
                    }
                )
                aggregate["hits"] += 1
                if args.stop_on_hit:
                    stop = True
                break

            added = 0
            for sector in ("z", "x"):
                low = analysis[f"{sector}_distance"]["low_logical_operators"]
                for operator in low[: args.cuts_per_sector_per_model]:
                    solver.add_distance_cut(sector, int(operator))
                    append_jsonl(
                        cut_file,
                        {
                            "model_index": model_index,
                            "sector": sector,
                            "operator": int(operator),
                            "weight": int(operator).bit_count(),
                        },
                    )
                    added += 1
            if not analysis["tanner_connected"] and not added:
                solver.block_current_seeds()
            aggregate["distance_cuts"] += added
            task_record["distance_cuts"] = solver.distance_cuts
            model_record["distance_cuts_added"] = added
            append_jsonl(transcript, model_record)
            checkpoint(
                "model-checked",
                active_task=task_name,
                model_index=model_index,
                model_distance=distance,
                model_connected=analysis["tanner_connected"],
                cuts_added=added,
            )
        else:
            task_record.update(
                {
                    "status": "model_limit",
                    "models": args.maximum_models_per_task,
                    "distance_cuts": solver.distance_cuts,
                }
            )
            aggregate["bounded_tasks"] += 1

        task_record["seconds"] = round(time.perf_counter() - built, 6)
        atomic_json(tasks_dir / f"{task_name}-summary.json", task_record)
        task_summaries.append(task_record)
        append_jsonl(output / "task-summaries.jsonl", task_record)
        checkpoint("task-complete", last_task=task_name, last_status=task_record["status"])
        if stop:
            break

    atomic_json(output / "candidates.json", hits)
    summary = {
        "schema_version": 1,
        "target": "[[30,4,>=6]] mixed-orbit C4-invariant permutation-ZX-folded CSS code",
        "n24_no_go": proof,
        "arguments": {
            **vars(args),
            "output_root": str(args.output_root),
            "initial_cuts": str(args.initial_cuts) if args.initial_cuts else None,
            "module_type": [list(item) for item in args.module_type or []],
        },
        "folds": [{"index": index, **fold.analyze()} for index, fold in selected_folds],
        "module_types": [list(item) for item in types],
        "planned_tasks": len(tasks),
        "completed_tasks": len(task_summaries),
        "aggregate": dict(sorted(aggregate.items())),
        "hits": len(hits),
        "seconds": round(time.perf_counter() - started, 6),
        "scope": (
            "seven regular C4 fibres plus one C2 orbit; zero-shift involutive "
            "normalizer-fold catalog; every task is exact only "
            "when its status is exact_unsat_after_cegis or hit"
        ),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint("complete", summary_path=str(output / "summary.json"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
