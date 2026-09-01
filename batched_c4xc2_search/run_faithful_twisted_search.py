#!/usr/bin/env python3
"""Checkpointed faithful-GL(2,2), x-reflected compact GALA search."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

import numpy as np
import qldpc

from .distance import certify_distance_at_least
from .faithful_twisted_halfswap_model import (
    analyze_faithful_twisted_half_swap_structure,
    build_faithful_twisted_half_swap_code,
    random_faithful_commutant_candidates,
    random_faithful_twisted_half_swap_candidates,
)
from .logicals import (
    batch_schemes,
    dress_logicals_for_batches,
    find_regular_logical_modules,
    logical_action_profile,
    orbit_array,
    translation_algebra_profile,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-weight", type=int, required=True)
    parser.add_argument("--g-weight", type=int)
    parser.add_argument("--commutant-p-weight", type=int)
    parser.add_argument("--commutant-q-weight", type=int)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260901)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-weight-ceiling", type=int, default=16)
    parser.add_argument("--minimum-distance", type=int, default=6)
    parser.add_argument("--distance-seconds", type=float, default=20)
    parser.add_argument("--logical-trials", type=int, default=4096)
    parser.add_argument("--logical-limit", type=int, default=32)
    parser.add_argument("--dressing-seconds", type=float, default=15)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-seconds", type=float, default=15)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def load_state(path: pathlib.Path) -> tuple[set[str], Counter[str]]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    if not path.exists():
        return seen, counts
    with path.open() as records:
        for line in records:
            if not line.strip():
                continue
            record = json.loads(line)
            seen.add(record["candidate_id"])
            counts[record["stage"]] += 1
            for reason in record["structure"]["rejection_reasons"]:
                counts[f"reject:{reason}"] += 1
    return seen, counts


def make_summary(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    counts: Counter[str],
    processed_total: int,
    processed_this_invocation: int,
    started: float,
    complete: bool,
) -> dict[str, Any]:
    commutant = args.commutant_p_weight is not None
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "run_directory": str(run_dir),
        "family": (
            "s3-linear-l2-commutant-x-reflected-halfswap"
            if commutant
            else "s3-linear-l2-independent-FG-x-reflected-halfswap"
        ),
        "profile": (
            [args.f_weight, args.commutant_p_weight, args.commutant_q_weight]
            if commutant
            else [args.f_weight, args.g_weight]
        ),
        "nominal_check_weight": (
            None if commutant else args.f_weight + args.g_weight
        ),
        "check_weight_ceiling": args.check_weight_ceiling,
        "minimum_distance": args.minimum_distance,
        "seed": args.seed,
        "requested_candidates": args.candidates,
        "processed_total": processed_total,
        "processed_this_invocation": processed_this_invocation,
        "counts": dict(counts),
        "complete": complete,
        "seconds_this_invocation": round(time.perf_counter() - started, 6),
        "qldpc_source": str(pathlib.Path(qldpc.__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    commutant_weights = (args.commutant_p_weight, args.commutant_q_weight)
    commutant = any(weight is not None for weight in commutant_weights)
    if commutant and not all(weight is not None for weight in commutant_weights):
        raise SystemExit("both commutant support weights must be supplied")
    if args.candidates < 1 or args.f_weight < 1:
        raise SystemExit("candidate count and support weights must be positive")
    if commutant:
        if args.g_weight is not None:
            raise SystemExit("do not combine --g-weight with the commutant ansatz")
        if args.commutant_p_weight < 0 or args.commutant_q_weight < 1:
            raise SystemExit(
                "commutant p weight must be nonnegative and q weight positive"
            )
    else:
        if args.g_weight is None or args.g_weight < 1:
            raise SystemExit("--g-weight is required for the independent ansatz")
        if args.f_weight + args.g_weight > args.check_weight_ceiling:
            raise SystemExit("the requested profile exceeds the nominal ceiling")

    run_dir = args.results_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(
            "run directory already exists; pass --resume or choose another run-id: "
            f"{run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "candidates.jsonl"
    hits_path = run_dir / "hits.jsonl"
    summary_path = run_dir / "summary.json"
    seen, counts = load_state(records_path) if args.resume else (set(), Counter())
    config = {
        "family": (
            "s3-linear-l2-commutant-x-reflected-halfswap"
            if commutant
            else "s3-linear-l2-independent-FG-x-reflected-halfswap"
        ),
        "profile": (
            [args.f_weight, *commutant_weights]
            if commutant
            else [args.f_weight, args.g_weight]
        ),
        "candidate_limit": args.candidates,
        "seed": args.seed,
        "check_weight_ceiling": args.check_weight_ceiling,
        "minimum_distance": args.minimum_distance,
        "translation_algebra_required_dimension": 8,
        "distance_seconds": args.distance_seconds,
        "logical_trials": args.logical_trials,
        "logical_limit": args.logical_limit,
        "dressing_seconds": args.dressing_seconds,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit("resume configuration does not match the saved run")
    else:
        atomic_json(config_path, config)

    processed_total = len(seen)
    processed_this_invocation = 0
    started = time.perf_counter()
    last_checkpoint = started
    if commutant:
        stream = random_faithful_commutant_candidates(
            f_weight=args.f_weight,
            p_weight=args.commutant_p_weight,
            q_weight=args.commutant_q_weight,
            check_weight_ceiling=args.check_weight_ceiling,
            count=args.candidates,
            seed=args.seed,
        )
    else:
        stream = random_faithful_twisted_half_swap_candidates(
            f_weight=args.f_weight,
            g_weight=args.g_weight,
            count=args.candidates,
            seed=args.seed,
        )
    for raw_index, candidate in enumerate(stream, start=1):
        if candidate.candidate_id in seen:
            continue
        processed_total += 1
        processed_this_invocation += 1
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "raw_index": raw_index,
            "candidate": candidate.to_dict(),
        }
        structure = analyze_faithful_twisted_half_swap_structure(
            candidate, check_weight_ceiling=args.check_weight_ceiling
        )
        record["structure"] = structure
        if not structure["accepted"]:
            record["stage"] = "structural_rejected"
        else:
            code = build_faithful_twisted_half_swap_code(candidate)
            algebra = translation_algebra_profile(code, candidate)
            record["translation_algebra"] = algebra
            if not algebra["group_relations_verified"]:
                record["stage"] = "translation_relations_rejected"
            elif not algebra["passes_regular_grid_necessary_gate"]:
                record["stage"] = "translation_algebra_rejected"
            else:
                distance = certify_distance_at_least(
                    code,
                    args.minimum_distance,
                    time_limit=args.distance_seconds,
                    equal_xz_by_permutation=True,
                )
                record["distance"] = distance
                if not distance["certified"]:
                    record["stage"] = (
                        "distance_rejected"
                        if any(
                            result["support"] is not None
                            for result in distance["directions"].values()
                        )
                        else "distance_unresolved"
                    )
                else:
                    action = logical_action_profile(
                        code,
                        candidate,
                        random_trials=args.logical_trials,
                        seed=args.seed ^ raw_index,
                    )
                    record["logical_action"] = action
                    if not action["full_rank_classes"]:
                        record["stage"] = "no_full_rank_translation_orbit"
                    else:
                        modules = find_regular_logical_modules(
                            code,
                            candidate,
                            random_trials=args.logical_trials,
                            seed=args.seed ^ raw_index,
                            limit=args.logical_limit,
                        )
                        record["logical_modules"] = modules
                        if not modules:
                            record["stage"] = "no_regular_zx_module"
                        else:
                            dressed_records = []
                            accepted = False
                            for module_index, module in enumerate(modules):
                                base = orbit_array(module, code.num_qubits)
                                for scheme_name in batch_schemes():
                                    if scheme_name in module["raw_batch_successes"]:
                                        dressed = {
                                            "scheme": scheme_name,
                                            "feasible": True,
                                            "already_disjoint_without_dressing": True,
                                        }
                                    else:
                                        dressed = dress_logicals_for_batches(
                                            code,
                                            base,
                                            scheme_name,
                                            time_limit=args.dressing_seconds,
                                        )
                                    dressed_records.append(
                                        {"module_index": module_index, **dressed}
                                    )
                                    if dressed.get("feasible") is True:
                                        accepted = True
                                        break
                                if accepted:
                                    break
                            record["dressed_batching"] = dressed_records
                            record["stage"] = (
                                "accepted"
                                if accepted
                                else "batching_failed_or_unresolved"
                            )
                            if accepted:
                                append_jsonl(hits_path, record)
                                np.savez_compressed(
                                    run_dir / f"{candidate.candidate_id}-checks.npz",
                                    matrix_x=np.asarray(code.matrix_x, dtype=np.uint8),
                                    matrix_z=np.asarray(code.matrix_z, dtype=np.uint8),
                                )

        counts[record["stage"]] += 1
        for reason in structure["rejection_reasons"]:
            counts[f"reject:{reason}"] += 1
        append_jsonl(records_path, record)
        now = time.perf_counter()
        if (
            processed_this_invocation % args.progress_every == 0
            or now - last_checkpoint >= args.checkpoint_seconds
        ):
            summary = make_summary(
                args=args,
                run_dir=run_dir,
                counts=counts,
                processed_total=processed_total,
                processed_this_invocation=processed_this_invocation,
                started=started,
                complete=False,
            )
            atomic_json(summary_path, summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            last_checkpoint = now

    final = make_summary(
        args=args,
        run_dir=run_dir,
        counts=counts,
        processed_total=processed_total,
        processed_this_invocation=processed_this_invocation,
        started=started,
        complete=True,
    )
    atomic_json(summary_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
