#!/usr/bin/env python3
"""Checkpointed search for batched ``C4 x C2`` Ising logical grids."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import qldpc

from .distance import certify_distance_at_least
from .logicals import (
    batch_schemes,
    dress_logicals_for_batches,
    find_regular_logical_modules,
    orbit_array,
)
from .model import (
    Candidate,
    Monomial,
    analyze_structure,
    build_code,
    quotient_s3_seed,
    random_candidates,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_profile(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(","))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("a profile is a comma-separated list of positive weights")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("quotient-seed", "abelian", "s3-natural", "s3-linear"),
        default="quotient-seed",
    )
    parser.add_argument("--profile", type=parse_profile, default=(2, 2))
    parser.add_argument("--fold-axis", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--run-id")
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-weight-ceiling", type=int, default=16)
    parser.add_argument("--minimum-distance", type=int, default=6)
    parser.add_argument("--distance-seconds", type=float, default=20)
    parser.add_argument("--logical-trials", type=int, default=64)
    parser.add_argument("--logical-limit", type=int, default=4)
    parser.add_argument("--dressing-seconds", type=float, default=20)
    parser.add_argument(
        "--batch-schemes",
        default=(
            "four_C2_x_squared,four_C2_y,four_C2_x_squared_y,"
            "two_C4_rows,two_C4_diagonals,two_C2xC2_columns"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-seconds", type=float, default=30)
    return parser.parse_args()


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def _accumulate_record(counts: Counter[str], record: dict[str, Any]) -> None:
    """Rebuild cumulative stage counts from one durable candidate record."""
    structure = record["structure"]
    if not structure["accepted"]:
        counts["structural_rejected"] += 1
        for reason in structure["rejection_reasons"]:
            counts[f"reject:{reason}"] += 1
        return
    counts["structural_survivor"] += 1
    distance = record.get("distance")
    if distance is None:
        return
    if not distance["certified"]:
        if any(
            result["support"] is not None
            for result in distance["directions"].values()
        ):
            counts["distance_rejected"] += 1
        else:
            counts["distance_unresolved"] += 1
        return
    counts["distance_certified"] += 1
    modules = record.get("logical_modules", [])
    if not modules:
        counts["no_regular_logical_module"] += 1
        return
    counts["regular_logical_module"] += 1
    if any(
        dressed.get("feasible") is True
        for dressed in record.get("dressed_batching", [])
    ):
        counts["accepted"] += 1
    else:
        counts["batch_dressing_failed_or_unresolved"] += 1


def _load_state(path: pathlib.Path) -> tuple[set[str], Counter[str]]:
    if not path.exists():
        return set(), Counter()
    output: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open() as records:
        for line in records:
            if line.strip():
                record = json.loads(line)
                output.add(record["candidate_id"])
                _accumulate_record(counts, record)
    return output, counts


def _candidate_stream(args: argparse.Namespace) -> Iterable[Candidate]:
    if args.family == "quotient-seed":
        yield quotient_s3_seed()
        return
    if args.family == "abelian" and len(args.profile) == 2:
        yield from itertools.islice(
            exhaustive_abelian_l4(args.profile, fold_axis=args.fold_axis),
            args.candidates,
        )
        return
    top = {
        "abelian": "trivial",
        "s3-natural": "s3-natural",
        "s3-linear": "s3-linear",
    }[args.family]
    yield from random_candidates(
        top_representation=top,
        entry_weights=args.profile,
        count=args.candidates,
        seed=args.seed,
        fold_axis=args.fold_axis,
    )


def exhaustive_abelian_l4(
    profile: Sequence[int], *, fold_axis: int = 0
) -> Iterable[Candidate]:
    """Enumerate the normalized two-entry abelian support family exactly."""
    if len(profile) != 2:
        raise ValueError("the exhaustive enumerator currently supports L=4 only")
    terms = [Monomial(0, xx, yy) for xx in range(4) for yy in range(2)]
    anchor = Monomial(0, 0, 0)
    for rest, second in itertools.product(
        itertools.combinations([term for term in terms if term != anchor], profile[0] - 1),
        itertools.combinations(terms, profile[1]),
    ):
        yield Candidate(
            top_representation="trivial",
            entries=(tuple(sorted((anchor, *rest))), tuple(sorted(second))),
            fold_axis=fold_axis,
        )


def _summary(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    counts: Counter[str],
    processed_total: int,
    processed_this_invocation: int,
    started: float,
    complete: bool,
) -> dict[str, Any]:
    nominal_weight = 16 if args.family == "quotient-seed" else 2 * sum(args.profile)
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "run_directory": str(run_dir),
        "family": args.family,
        "profile": None if args.family == "quotient-seed" else list(args.profile),
        "fold_axis": 0 if args.family == "quotient-seed" else args.fold_axis,
        "nominal_check_weight": nominal_weight,
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
    if args.candidates < 1 or args.logical_trials < 0 or args.logical_limit < 1:
        raise SystemExit("candidate and logical limits must be positive")
    if args.family != "quotient-seed" and not 0 <= args.fold_axis < len(args.profile):
        raise SystemExit("fold axis must lie between zero and L/2 - 1")
    if 2 * sum(args.profile) > args.check_weight_ceiling:
        raise SystemExit("the requested profile exceeds the check-weight ceiling")
    scheme_names = tuple(name for name in args.batch_schemes.split(",") if name)
    unknown = set(scheme_names) - set(batch_schemes())
    if unknown:
        raise SystemExit(f"unknown batching schemes: {sorted(unknown)}")

    default_id = (
        f"{args.family}-l{2 * len(args.profile)}-fold{args.fold_axis}-"
        f"w{2 * sum(args.profile)}-"
        f"seed{args.seed}-v1"
    )
    run_dir = args.results_root / (args.run_id or default_id)
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists; pass --resume or choose another --run-id: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "candidates.jsonl"
    hits_path = run_dir / "hits.jsonl"
    summary_path = run_dir / "summary.json"
    processed_ids, counts = (
        _load_state(records_path) if args.resume else (set(), Counter())
    )

    config = {
        "family": args.family,
        "profile": None if args.family == "quotient-seed" else list(args.profile),
        "fold_axis": 0 if args.family == "quotient-seed" else args.fold_axis,
        "nominal_check_weight": (
            16 if args.family == "quotient-seed" else 2 * sum(args.profile)
        ),
        "candidate_limit": args.candidates,
        "seed": args.seed,
        "check_weight_ceiling": args.check_weight_ceiling,
        "minimum_distance": args.minimum_distance,
        "distance_seconds": args.distance_seconds,
        "logical_trials": args.logical_trials,
        "logical_limit": args.logical_limit,
        "dressing_seconds": args.dressing_seconds,
        "batch_schemes": list(scheme_names),
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        existing.setdefault("fold_axis", 0)
        if existing != config:
            raise SystemExit("resume configuration does not match the saved run")
    else:
        _atomic_json(config_path, config)

    processed_total = len(processed_ids)
    processed_this_invocation = 0
    started = time.perf_counter()
    last_checkpoint = started
    candidates = _candidate_stream(args)
    for raw_index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id in processed_ids:
            continue
        processed_total += 1
        processed_this_invocation += 1
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "raw_index": raw_index,
            "candidate": candidate.to_dict(),
        }
        structure = analyze_structure(
            candidate, check_weight_ceiling=args.check_weight_ceiling
        )
        record["structure"] = structure
        if not structure["accepted"]:
            counts["structural_rejected"] += 1
            for reason in structure["rejection_reasons"]:
                counts[f"reject:{reason}"] += 1
        else:
            counts["structural_survivor"] += 1
            code = build_code(candidate)
            distance = certify_distance_at_least(
                code,
                args.minimum_distance,
                time_limit=args.distance_seconds,
                equal_xz_by_permutation=True,
            )
            record["distance"] = distance
            if not distance["certified"]:
                if any(
                    result["support"] is not None
                    for result in distance["directions"].values()
                ):
                    counts["distance_rejected"] += 1
                else:
                    counts["distance_unresolved"] += 1
            else:
                counts["distance_certified"] += 1
                modules = find_regular_logical_modules(
                    code,
                    candidate,
                    random_trials=args.logical_trials,
                    seed=args.seed ^ raw_index,
                    limit=args.logical_limit,
                )
                record["logical_modules"] = modules
                if not modules:
                    counts["no_regular_logical_module"] += 1
                else:
                    counts["regular_logical_module"] += 1
                    dressed_records = []
                    accepted_batching = False
                    for module_index, module in enumerate(modules):
                        base = orbit_array(module, code.num_qubits)
                        for scheme_name in scheme_names:
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
                                accepted_batching = True
                                break
                        if accepted_batching:
                            break
                    record["dressed_batching"] = dressed_records
                    if accepted_batching:
                        counts["accepted"] += 1
                        _append_jsonl(hits_path, record)
                        np.savez_compressed(
                            run_dir / f"{candidate.candidate_id}-checks.npz",
                            matrix_x=np.asarray(code.matrix_x, dtype=np.uint8),
                            matrix_z=np.asarray(code.matrix_z, dtype=np.uint8),
                        )
                    else:
                        counts["batch_dressing_failed_or_unresolved"] += 1

        _append_jsonl(records_path, record)
        now = time.perf_counter()
        if (
            processed_this_invocation % args.progress_every == 0
            or now - last_checkpoint >= args.checkpoint_seconds
        ):
            checkpoint = _summary(
                args=args,
                run_dir=run_dir,
                counts=counts,
                processed_total=processed_total,
                processed_this_invocation=processed_this_invocation,
                started=started,
                complete=False,
            )
            _atomic_json(summary_path, checkpoint)
            print(json.dumps(checkpoint, sort_keys=True), flush=True)
            last_checkpoint = now

    final = _summary(
        args=args,
        run_dir=run_dir,
        counts=counts,
        processed_total=processed_total,
        processed_this_invocation=processed_this_invocation,
        started=started,
        complete=True,
    )
    _atomic_json(summary_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
