#!/usr/bin/env python3
"""Checkpointed direct folded-CSS search for [[n,2,6]], n in {14,16,18,20}."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import z3

from inverse_c2_minimum.search import (
    TARGET_DISTANCE,
    FoldedCSSSolver,
    analyze_basis,
    canonical_fold_catalog,
    geometry_catalog,
    module_types,
    n12_no_go_certificate,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def parse_module_type(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if any(item not in (1, 2) for item in parsed):
        raise argparse.ArgumentTypeError("C2 module parts must be one or two")
    if tuple(sorted(parsed, reverse=True)) != parsed:
        raise argparse.ArgumentTypeError("module type must be nonincreasing")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--length", type=int, choices=(14, 16, 18, 20), action="append"
    )
    parser.add_argument("--geometry-name", action="append")
    parser.add_argument("--fold-index", type=int, action="append")
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument(
        "--fold-stop",
        type=int,
        help="exclusive canonical fold-index bound within every selected geometry",
    )
    parser.add_argument("--module-type", type=parse_module_type, action="append")
    parser.add_argument("--maximum-check-weight", type=int, default=8)
    parser.add_argument("--solver-timeout-seconds", type=float, default=3)
    parser.add_argument("--maximum-solver-timeout-seconds", type=float, default=20)
    parser.add_argument("--maximum-models-per-task", type=int, default=100)
    parser.add_argument("--cuts-per-sector-per-model", type=int, default=256)
    parser.add_argument("--maximum-tasks", type=int)
    parser.add_argument("--checkpoint-every-models", type=int, default=10)
    parser.add_argument(
        "--initial-cuts",
        type=Path,
        help="replay a saved JSONL cut frontier; requires exactly one task",
    )
    parser.add_argument(
        "--eager-distance",
        action="store_true",
        help="encode all errors through weight five before the first solver call",
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

    no_go = n12_no_go_certificate()
    atomic_json(output / "n12-no-go-certificate.json", no_go)
    print(json.dumps({"stage": "n12-certificate", **no_go}, sort_keys=True), flush=True)

    lengths = args.length or [14]
    catalog_records: list[dict[str, Any]] = []
    tasks: list[tuple[Any, Any, tuple[int, ...]]] = []
    for n in lengths:
        target_rank = (n - 2) // 2
        selected_types = args.module_type or list(module_types(target_rank))
        for module_type in selected_types:
            if sum(module_type) != target_rank:
                raise SystemExit(
                    f"module type {module_type} does not sum to rank {target_rank} for n={n}"
                )
        for geometry in geometry_catalog(n):
            if args.geometry_name and geometry.name not in args.geometry_name:
                continue
            folds = canonical_fold_catalog(geometry)
            selected_folds = [
                fold
                for fold in folds
                if fold.index >= args.fold_start
                and (args.fold_stop is None or fold.index < args.fold_stop)
                and (args.fold_index is None or fold.index in args.fold_index)
            ]
            catalog_record = {
                "geometry": geometry.analyze(),
                "canonical_folds": len(folds),
                "selected_folds": len(selected_folds),
                "raw_valid_folds": sum(fold.raw_multiplicity for fold in folds),
            }
            catalog_records.append(catalog_record)
            atomic_json(output / f"catalog-{geometry.name}.json", catalog_record)
            print(json.dumps({"stage": "catalog", **catalog_record}, sort_keys=True), flush=True)
            for fold in selected_folds:
                for module_type in selected_types:
                    tasks.append((geometry, fold, tuple(module_type)))
    tasks.sort(
        key=lambda item: (
            item[0].n,
            item[0].logical_weight,
            -item[0].slack_transpositions,
            item[1].index,
            -item[2].count(2),
        )
    )
    if args.maximum_tasks is not None:
        tasks = tasks[: args.maximum_tasks]
    if args.initial_cuts and len(tasks) != 1:
        raise SystemExit("--initial-cuts requires exactly one selected task")

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

    checkpoint("starting", maximum_check_weight=args.maximum_check_weight)
    stop = False
    for task_index, (geometry, fold, module_type) in enumerate(tasks):
        module_name = "-".join(map(str, module_type))
        task_name = (
            f"task-{task_index:04d}-{geometry.name}-"
            f"f{fold.index:04d}-m{module_name}"
        )
        transcript = tasks_dir / f"{task_name}.jsonl"
        cut_file = tasks_dir / f"{task_name}-distance-cuts.jsonl"
        checkpoint(
            "building",
            active_task=task_name,
            geometry=geometry.name,
            fold_index=fold.index,
            module_type=list(module_type),
        )
        built = time.perf_counter()
        solver = FoldedCSSSolver(
            fold=fold,
            module_type=module_type,
            maximum_check_weight=args.maximum_check_weight,
            timeout_ms=max(1, int(1000 * args.solver_timeout_seconds)),
        )
        seen_cuts: set[tuple[str, int]] = set()
        replayed_cuts = 0
        if args.initial_cuts:
            with args.initial_cuts.open(encoding="utf-8") as cut_input:
                for line in cut_input:
                    record = json.loads(line)
                    key = (str(record["sector"]), int(record["operator"]))
                    if key in seen_cuts:
                        continue
                    seen_cuts.add(key)
                    solver.add_distance_cut(*key)
                    replayed_cuts += 1
            checkpoint(
                "cuts-replayed",
                active_task=task_name,
                replayed_cuts=replayed_cuts,
            )
        eager_constraints = 0
        if args.eager_distance:
            eager_started = time.perf_counter()
            eager_constraints = solver.add_full_distance_constraints()
            checkpoint(
                "eager-distance-built",
                active_task=task_name,
                eager_distance_constraints=eager_constraints,
                eager_build_seconds=round(time.perf_counter() - eager_started, 6),
            )
        task_record: dict[str, Any] = {
            "task_index": task_index,
            "task_name": task_name,
            "geometry": geometry.analyze(),
            "fold": fold.analyze(),
            "module_type": list(module_type),
            "models": 0,
            "distance_cuts": 0,
            "status": "running",
            "build_seconds": round(time.perf_counter() - built, 6),
            "eager_distance_constraints": eager_constraints,
            "replayed_cuts": replayed_cuts,
        }
        best_score = (-1, -10**9)
        for model_index in range(args.maximum_models_per_task):
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
                        "status": (
                            "exact_unsat_after_eager_distance"
                            if args.eager_distance
                            else "exact_unsat_after_cegis"
                        ),
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
                        "fold_index": fold.index,
                        "analysis": analysis,
                    },
                )
            if distance >= TARGET_DISTANCE and analysis["tanner_connected"]:
                append_jsonl(transcript, model_record)
                candidate = {
                    "task_name": task_name,
                    "fold_index": fold.index,
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
                sector_added = 0
                for operator in low:
                    key = (sector, int(operator))
                    if key in seen_cuts:
                        continue
                    seen_cuts.add(key)
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
                    sector_added += 1
                    if sector_added >= args.cuts_per_sector_per_model:
                        break
            if not added:
                solver.block_current_seeds()
            aggregate["distance_cuts"] += added
            task_record["distance_cuts"] = solver.distance_cuts
            model_record["distance_cuts_added"] = added
            append_jsonl(transcript, model_record)
            if (
                model_index == 0
                or (model_index + 1) % args.checkpoint_every_models == 0
            ):
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
        "target": "[[n,2,>=6]] C2-invariant permutation-ZX-folded CSS code",
        "n12_no_go": no_go,
        "arguments": {
            **vars(args),
            "output_root": str(args.output_root),
            "initial_cuts": str(args.initial_cuts) if args.initial_cuts else None,
            "module_type": [list(item) for item in args.module_type or []],
        },
        "catalogs": catalog_records,
        "planned_tasks": len(tasks),
        "completed_tasks": len(task_summaries),
        "aggregate": dict(sorted(aggregate.items())),
        "hits": len(hits),
        "seconds": round(time.perf_counter() - started, 6),
        "scope": (
            "complete canonical commuting-involution fold catalog for the selected "
            "logical support weights and slack C2 cycle types; a task is exact only "
            "when its status is exact_unsat_after_cegis or hit"
        ),
    }
    atomic_json(output / "summary.json", summary)
    checkpoint("complete", summary_path=str(output / "summary.json"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
