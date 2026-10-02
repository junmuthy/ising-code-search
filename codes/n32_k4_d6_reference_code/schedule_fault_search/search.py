#!/usr/bin/env python3
"""Resumably enumerate and fault-screen alternative clean depth-12 schedules."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from codes.n32_k4_d6_reference_code.schedule_fault_search.data import code_for_schedule
from codes.n32_k4_d6_reference_code.schedule_fault_search.scheduler import (
    ScheduleEnumerator,
    build_schedule_record,
    matrix_from_supports,
)
from codes.n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_through_four,
    classify_explained_errors,
    heuristic_screen,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.circuit import (
    NoiseModel,
    build_bulk_circuit,
    build_memory_circuit,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import atomic_json, utc_now


MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR.parent
PRESENTATION_PATH = (
    REFERENCE_DIR / "schedule" / "all_weight8_translation_symmetric_v1" / "presentation.json"
)
FOLDS_PATH = (
    REFERENCE_DIR
    / "followups"
    / "fold_rescan_and_second_translation_v1"
    / "distance-six-folds.jsonl"
)
REFERENCE_SCHEDULE_PATH = (
    REFERENCE_DIR
    / "followups"
    / "fold_rescan_and_second_translation_v1"
    / "optimal-clean-joint-12-layer.json"
)


def parse_indices(specification: str) -> list[int]:
    output: set[int] = set()
    for term in specification.split(","):
        term = term.strip()
        if not term:
            continue
        if "-" in term:
            first_text, last_text = term.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"invalid descending fold range: {term}")
            output.update(range(first, last + 1))
        else:
            output.add(int(term))
    if not output:
        raise ValueError("at least one fold index is required")
    return sorted(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-indices", default="0")
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--solver-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--minimum-color-changes",
        type=int,
        default=4,
        help="minimum orbit-color Hamming distance between sampled schedules",
    )
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument(
        "--exclude-records",
        type=Path,
        action="append",
        default=[],
        help="candidate JSONL from an earlier run whose color neighborhoods should be excluded",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-survivor", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compact_heuristic(
    result: dict[str, Any], schedule: dict[str, Any]
) -> dict[str, Any]:
    errors = result["explained_errors"]
    return {
        "upper_bound": result["upper_bound"],
        "elapsed_seconds": result["elapsed_seconds"],
        "fault_locations": classify_explained_errors(errors, schedule),
        "explained_errors": errors,
    }


def exact_locations(result: dict[str, Any], schedule: dict[str, Any]) -> list[dict[str, Any]]:
    witness = result.get("witness")
    if not witness:
        return []
    errors = [
        effect.get("representative_circuit_error")
        for effect in witness.get("effects", [])
        if effect.get("representative_circuit_error")
    ]
    return classify_explained_errors(errors, schedule)


def screen_circuit(
    circuit: Any,
    *,
    stage: str,
    schedule: dict[str, Any],
    heartbeat_seconds: float,
    progress: Any,
) -> tuple[dict[str, Any], bool, str]:
    """Apply heuristic rejection and exact four-fault exclusion to one circuit."""

    heuristic_raw = heuristic_screen(
        circuit,
        heartbeat_seconds=heartbeat_seconds,
        progress=progress,
    )
    heuristic = compact_heuristic(heuristic_raw, schedule)
    result: dict[str, Any] = {"heuristic": heuristic}
    upper_bound = heuristic["upper_bound"]
    if upper_bound is not None and int(upper_bound) <= 4:
        return result, False, f"heuristic-d{upper_bound}"
    exact = certify_through_four(
        circuit,
        heartbeat_seconds=heartbeat_seconds,
        progress=progress,
    )
    exact["fault_locations"] = exact_locations(exact, schedule)
    result["exact_through_four"] = exact
    if int(exact["lower_bound"]) < 5:
        return result, False, "exact-d4"
    return result, True, "lower-bound-5"


def main() -> None:
    args = parse_args()
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be positive")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.resume and not args.output.exists():
        raise SystemExit(f"resume output directory does not exist: {args.output}")

    fold_indices = parse_indices(args.fold_indices)
    presentation = json.loads(PRESENTATION_PATH.read_text(encoding="utf-8"))
    folds = [
        json.loads(line)
        for line in FOLDS_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(index < 0 or index >= len(folds) for index in fold_indices):
        raise SystemExit(f"fold indices must lie in 0..{len(folds) - 1}")
    checks_x = matrix_from_supports(presentation["checks_x"])
    configuration = {
        "schema_version": 1,
        "fold_indices": fold_indices,
        "max_candidates": args.max_candidates,
        "depth": args.depth,
        "solver_timeout_seconds": args.solver_timeout_seconds,
        "heartbeat_seconds": args.heartbeat_seconds,
        "checkpoint_every": args.checkpoint_every,
        "random_seed": args.random_seed,
        "minimum_color_changes": args.minimum_color_changes,
        "probability": args.probability,
        "exclude_records": [str(path.resolve()) for path in args.exclude_records],
        "presentation": str(PRESENTATION_PATH),
        "folds": str(FOLDS_PATH),
        "reference_schedule": str(REFERENCE_SCHEDULE_PATH),
        "screen": "staged exact-through-four screening: CNOT-only bulk X/Z, CNOT-only three-round memory X/Z, then full-noise three-round memory X/Z",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    configuration_path = args.output / "configuration.json"
    records_path = args.output / "candidates.jsonl"
    checkpoint_path = args.output / "progress.json"
    pending_path = args.output / "pending-candidate.json"
    if args.resume:
        saved_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        if saved_configuration != configuration:
            raise SystemExit("resume configuration does not match the saved search")
    else:
        atomic_json(configuration_path, configuration)

    previous = load_jsonl(records_path)
    pending: dict[str, Any] | None = None
    if args.resume and pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
    excluded = [
        record
        for path in args.exclude_records
        for record in load_jsonl(path)
    ]
    if len(previous) > args.max_candidates:
        raise SystemExit("saved candidate count exceeds configured maximum")
    existing_ids = {str(record["schedule_id"]) for record in previous}
    enumerators: dict[int, ScheduleEnumerator] = {}
    reference = json.loads(REFERENCE_SCHEDULE_PATH.read_text(encoding="utf-8"))
    for fold_index in fold_indices:
        enumerator = ScheduleEnumerator(
            checks_x,
            folds[fold_index]["permutation"],
            presentation["translation_action_on_x_checks"],
            depth=args.depth,
            timeout_seconds=args.solver_timeout_seconds,
            random_seed=args.random_seed + 1009 * fold_index,
            minimum_color_changes=args.minimum_color_changes,
        )
        if fold_index == int(reference["fold_index"]):
            enumerator.exclude(reference["edge_orbit_colors_zero_based"])
        for record in previous:
            if int(record["fold_index"]) == fold_index:
                enumerator.exclude(record["edge_orbit_colors_zero_based"])
        for record in excluded:
            if int(record["fold_index"]) == fold_index:
                enumerator.exclude(record["edge_orbit_colors_zero_based"])
        if pending is not None and int(pending["record"]["fold_index"]) == fold_index:
            enumerator.exclude(pending["record"]["edge_orbit_colors_zero_based"])
        enumerators[fold_index] = enumerator

    status_counts = Counter(str(record["status"]) for record in previous)
    fold_counts = Counter(int(record["fold_index"]) for record in previous)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started": utc_now(),
        "updated": utc_now(),
        "configuration": configuration,
        "candidates_completed": len(previous),
        "status_histogram": dict(sorted(status_counts.items())),
        "fold_histogram": {str(key): value for key, value in sorted(fold_counts.items())},
        "active": None,
    }
    if args.resume and checkpoint_path.exists():
        old_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        state["started"] = old_state.get("started", state["started"])
    atomic_json(checkpoint_path, state)
    started = time.monotonic()
    active_folds = list(fold_indices)
    cursor = len(previous) % len(active_folds)

    def progress(fold_index: int, candidate_number: int, basis: str, event: dict[str, Any]) -> None:
        state["updated"] = utc_now()
        state["active"] = {
            "fold_index": fold_index,
            "candidate_number": candidate_number,
            "basis": basis,
            **event,
        }
        atomic_json(checkpoint_path, state)
        fields = " ".join(
            f"{key}={value}" for key, value in event.items() if key != "timestamp"
        )
        print(
            f"checkpoint {event.get('timestamp', utc_now())} phase=screen "
            f"candidate={candidate_number}/{args.max_candidates} fold={fold_index} "
            f"basis={basis} {fields}",
            flush=True,
        )

    while (len(previous) < args.max_candidates or pending is not None) and active_folds:
        if pending is not None:
            schedule = pending["schedule"]
            record = pending["record"]
            pending = None
            fold_index = int(record["fold_index"])
            identifier = str(record["schedule_id"])
            candidate_number = int(record["candidate_number"])
            print(
                f"checkpoint {utc_now()} phase=resume-pending candidate={candidate_number}/"
                f"{args.max_candidates} fold={fold_index} schedule_id={identifier}",
                flush=True,
            )
        else:
            cursor %= len(active_folds)
            fold_index = active_folds[cursor]
            enumeration = enumerators[fold_index].next()
            if enumeration.status != "sat" or enumeration.colors is None:
                print(
                    f"checkpoint {utc_now()} phase=enumerate fold={fold_index} "
                    f"status={enumeration.status} reason={enumeration.reason_unknown}",
                    flush=True,
                )
                active_folds.pop(cursor)
                continue
            cursor += 1
            schedule = build_schedule_record(
                checks_x,
                folds[fold_index]["permutation"],
                enumerators[fold_index].edge_orbit,
                enumeration.colors,
                depth=args.depth,
                fold_index=fold_index,
                fold=folds[fold_index],
            )
            identifier = str(schedule["schedule_id"])
            if identifier in existing_ids:
                continue
            candidate_number = len(previous) + 1
            record = {
                "schema_version": 1,
                "timestamp": utc_now(),
                "candidate_number": candidate_number,
                "schedule_id": identifier,
                "fold_index": fold_index,
                "edge_orbit_colors_zero_based": list(enumeration.colors),
                "solver_seconds": round(enumeration.solver_seconds, 6),
                "status": "screening",
                "fault_screen": {},
            }
            atomic_json(pending_path, {"schedule": schedule, "record": record})
        code = code_for_schedule(schedule)
        cnot_noise = NoiseModel.cnot_only(args.probability)
        full_noise = NoiseModel(args.probability)
        stages = [
            (
                "bulk-cnot-X",
                lambda: build_bulk_circuit("X", noise=cnot_noise, code=code),
            ),
            (
                "bulk-cnot-Z",
                lambda: build_bulk_circuit("Z", noise=cnot_noise, code=code),
            ),
            (
                "memory-r3-cnot-X",
                lambda: build_memory_circuit("X", 3, noise=cnot_noise, code=code),
            ),
            (
                "memory-r3-cnot-Z",
                lambda: build_memory_circuit("Z", 3, noise=cnot_noise, code=code),
            ),
            (
                "memory-r3-full-X",
                lambda: build_memory_circuit("X", 3, noise=full_noise, code=code),
            ),
            (
                "memory-r3-full-Z",
                lambda: build_memory_circuit("Z", 3, noise=full_noise, code=code),
            ),
        ]
        passed_all = True
        for stage, build in stages:
            stage_result, passed, reason = screen_circuit(
                build(),
                stage=stage,
                schedule=schedule,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=lambda event, label=stage: progress(
                    fold_index, candidate_number, label, event
                ),
            )
            record["fault_screen"][stage] = stage_result
            if not passed:
                record["status"] = f"rejected-{stage}-{reason}"
                passed_all = False
                break
        if passed_all:
            record["status"] = "memory-r3-full-survivor-lower-bound-5"
            survivor_path = args.output / "survivors" / f"{identifier}.schedule.json"
            atomic_json(survivor_path, schedule)
            record["survivor_schedule"] = str(survivor_path)

        record["elapsed_seconds"] = round(
            enumeration.solver_seconds
            + sum(
                float(basis_result["heuristic"]["elapsed_seconds"])
                + float(basis_result.get("exact_through_four", {}).get("elapsed_seconds", 0))
                for basis_result in record["fault_screen"].values()
            ),
            3,
        )
        append_jsonl(records_path, record)
        pending_path.unlink(missing_ok=True)
        previous.append(record)
        existing_ids.add(identifier)
        status_counts[record["status"]] += 1
        fold_counts[fold_index] += 1
        state.update(
            {
                "updated": utc_now(),
                "candidates_completed": len(previous),
                "status_histogram": dict(sorted(status_counts.items())),
                "fold_histogram": {
                    str(key): value for key, value in sorted(fold_counts.items())
                },
                "active": None,
                "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3),
            }
        )
        atomic_json(checkpoint_path, state)
        if candidate_number % args.checkpoint_every == 0 or "survivor" in record["status"]:
            print(
                f"checkpoint {utc_now()} phase=search candidate={candidate_number}/"
                f"{args.max_candidates} fold={fold_index} status={record['status']} "
                f"histogram={json.dumps(dict(sorted(status_counts.items())), sort_keys=True)}",
                flush=True,
            )
        if args.stop_on_survivor and "survivor" in record["status"]:
            break

    state.update(
        {
            "status": "complete" if len(previous) >= args.max_candidates else "exhausted",
            "completed": utc_now(),
            "updated": utc_now(),
            "candidates_completed": len(previous),
            "status_histogram": dict(sorted(status_counts.items())),
            "fold_histogram": {str(key): value for key, value in sorted(fold_counts.items())},
            "active": None,
            "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3),
        }
    )
    atomic_json(checkpoint_path, state)
    atomic_json(args.output / "summary.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
