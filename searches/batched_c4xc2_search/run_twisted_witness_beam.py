#!/usr/bin/env python3
"""Witness-guided beam search within the x-reflected compact GALA family."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any, Iterable

import numpy as np

from .distance import certify_distance_at_least, low_weight_logical_spectrum
from .logicals import (
    batch_schemes,
    dress_logicals_for_batches,
    find_regular_logical_modules,
    logical_action_profile,
    orbit_array,
    translation_algebra_profile,
)
from .twisted_halfswap_model import (
    TwistedHalfSwapCandidate,
    analyze_twisted_half_swap_structure,
    build_twisted_half_swap_code,
    sample_twisted_double_neighbors,
    twisted_support_neighbors,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--mutation-radius", type=int, choices=(1, 2), default=1)
    parser.add_argument("--neighbors-per-parent", type=int, default=1000)
    parser.add_argument("--mutation-seed", type=int, default=260901)
    parser.add_argument("--check-weight-ceiling", type=int, default=16)
    parser.add_argument("--distance-seconds", type=float, default=20)
    parser.add_argument("--logical-trials", type=int, default=4096)
    parser.add_argument("--logical-limit", type=int, default=32)
    parser.add_argument("--dressing-seconds", type=float, default=15)
    parser.add_argument("--progress-every", type=int, default=100)
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


def source_candidates(sources: Iterable[pathlib.Path]) -> list[TwistedHalfSwapCandidate]:
    candidates: dict[str, TwistedHalfSwapCandidate] = {}
    for source in sources:
        records_path = source / "candidates.jsonl" if source.is_dir() else source
        if records_path.suffix == ".json":
            loaded = json.loads(records_path.read_text())
            records = loaded if isinstance(loaded, list) else [loaded]
        else:
            with records_path.open() as input_records:
                records = [json.loads(line) for line in input_records if line.strip()]
        for record in records:
            if record.get("translation_algebra", {}).get("dimension") != 8:
                continue
            candidate = TwistedHalfSwapCandidate.from_dict(record["candidate"])
            candidates[candidate.candidate_id] = candidate
    return [candidates[key] for key in sorted(candidates)]


def load_records(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return output
    with path.open() as records:
        for line in records:
            if line.strip():
                record = json.loads(line)
                output[record["candidate_id"]] = record
    return output


def record_score(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Lexicographically prefer clearance, then fewer witnesses, then rate."""
    spectrum = record.get("low_weight_spectrum")
    if spectrum is None:
        return (-1, -10**12, -10**12, -10**12, -1)
    counts = spectrum["counts"]
    clearance = spectrum["minimum_logical_weight_found"] or 5
    distance = record.get("distance")
    if distance and distance.get("certified"):
        clearance = 6
    elif distance:
        found = [
            result["weight"]
            for result in distance["directions"].values()
            if result["weight"] is not None
        ]
        if found:
            clearance = min(found)
    return (
        int(clearance),
        -int(counts["2"]),
        -int(counts["3"]),
        -int(counts["4"]),
        int(record["structure"]["k"]),
    )


def eligible_for_beam(record: dict[str, Any]) -> bool:
    return bool(
        record["structure"]["accepted"]
        and record.get("translation_algebra", {}).get("dimension") == 8
        and record.get("low_weight_spectrum") is not None
    )


def evaluate_candidate(
    candidate: TwistedHalfSwapCandidate,
    *,
    generation: int,
    parent_id: str | None,
    mutation: dict[str, Any] | None,
    args: argparse.Namespace,
    hits_path: pathlib.Path,
    run_dir: pathlib.Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "generation": generation,
        "parent_id": parent_id,
        "mutation": mutation,
        "candidate": candidate.to_dict(),
    }
    structure = analyze_twisted_half_swap_structure(
        candidate, check_weight_ceiling=args.check_weight_ceiling
    )
    record["structure"] = structure
    if not structure["accepted"]:
        record["stage"] = "structural_rejected"
        return record

    code = build_twisted_half_swap_code(candidate)
    algebra = translation_algebra_profile(code, candidate)
    record["translation_algebra"] = algebra
    if not algebra["passes_regular_grid_necessary_gate"]:
        record["stage"] = "translation_algebra_rejected"
        return record

    spectrum = low_weight_logical_spectrum(code, maximum_weight=4)
    record["low_weight_spectrum"] = spectrum
    if not spectrum["clear_through_maximum_weight"]:
        record["stage"] = "low_weight_logical"
        record["score"] = list(record_score(record))
        return record

    distance = certify_distance_at_least(
        code,
        6,
        time_limit=args.distance_seconds,
        equal_xz_by_permutation=True,
    )
    record["distance"] = distance
    if not distance["certified"]:
        record["stage"] = (
            "weight_five_logical"
            if any(
                result["support"] is not None
                for result in distance["directions"].values()
            )
            else "distance_unresolved"
        )
        record["score"] = list(record_score(record))
        return record

    action = logical_action_profile(
        code,
        candidate,
        random_trials=args.logical_trials,
        seed=260831 ^ int(candidate.candidate_id[-8:], 16),
    )
    record["logical_action"] = action
    if not action["full_rank_classes"]:
        record["stage"] = "no_full_rank_translation_orbit"
        record["score"] = list(record_score(record))
        return record

    modules = find_regular_logical_modules(
        code,
        candidate,
        random_trials=args.logical_trials,
        seed=260831 ^ int(candidate.candidate_id[-8:], 16),
        limit=args.logical_limit,
    )
    record["logical_modules"] = modules
    if not modules:
        record["stage"] = "no_regular_zx_module"
        record["score"] = list(record_score(record))
        return record

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
            dressed_records.append({"module_index": module_index, **dressed})
            if dressed.get("feasible") is True:
                accepted = True
                break
        if accepted:
            break
    record["dressed_batching"] = dressed_records
    record["stage"] = "accepted" if accepted else "batching_failed_or_unresolved"
    record["score"] = list(record_score(record))
    if accepted:
        append_jsonl(hits_path, record)
        np.savez_compressed(
            run_dir / f"{candidate.candidate_id}-checks.npz",
            matrix_x=np.asarray(code.matrix_x, dtype=np.uint8),
            matrix_z=np.asarray(code.matrix_z, dtype=np.uint8),
        )
    return record


def stage_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(record["stage"] for record in records))


def checkpoint(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    records: dict[str, dict[str, Any]],
    generation: int,
    generation_evaluated: int,
    generation_total: int,
    started: float,
    complete: bool,
) -> None:
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "complete": complete,
        "generation": generation,
        "requested_generations": args.generations,
        "beam_width": args.beam_width,
        "evaluated_unique": len(records),
        "generation_evaluated": generation_evaluated,
        "generation_total": generation_total,
        "counts": stage_counts(records.values()),
        "best_score": max(
            (record_score(record) for record in records.values()),
            default=None,
        ),
        "seconds_this_invocation": round(time.perf_counter() - started, 6),
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def select_beam(
    candidate_ids: Iterable[str],
    records: dict[str, dict[str, Any]],
    width: int,
) -> list[dict[str, Any]]:
    eligible = [records[candidate_id] for candidate_id in set(candidate_ids) if eligible_for_beam(records[candidate_id])]
    return sorted(
        eligible,
        key=lambda record: (record_score(record), record["candidate_id"]),
        reverse=True,
    )[:width]


def main() -> None:
    args = parse_args()
    if args.generations < 1 or args.beam_width < 1:
        raise SystemExit("generations and beam width must be positive")
    if args.neighbors_per_parent < 1:
        raise SystemExit("neighbors per parent must be positive")
    seeds = source_candidates(args.sources)
    if not seeds:
        raise SystemExit("no dimension-eight seed candidates found")
    run_dir = args.results_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "evaluations.jsonl"
    hits_path = run_dir / "hits.jsonl"
    records = load_records(records_path) if args.resume else {}
    config = {
        "sources": [str(source) for source in args.sources],
        "seed_candidates": len(seeds),
        "generations": args.generations,
        "beam_width": args.beam_width,
        "mutation_radius": args.mutation_radius,
        "neighbors_per_parent": args.neighbors_per_parent,
        "mutation_seed": args.mutation_seed,
        "check_weight_ceiling": args.check_weight_ceiling,
        "distance_seconds": args.distance_seconds,
        "logical_trials": args.logical_trials,
        "logical_limit": args.logical_limit,
        "dressing_seconds": args.dressing_seconds,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit("resume configuration does not match")
    else:
        atomic_json(config_path, config)

    started = time.perf_counter()
    last_checkpoint = started
    seed_ids = []
    for index, candidate in enumerate(seeds, start=1):
        seed_ids.append(candidate.candidate_id)
        if candidate.candidate_id not in records:
            record = evaluate_candidate(
                candidate,
                generation=0,
                parent_id=None,
                mutation=None,
                args=args,
                hits_path=hits_path,
                run_dir=run_dir,
            )
            records[candidate.candidate_id] = record
            append_jsonl(records_path, record)
        now = time.perf_counter()
        if index % args.progress_every == 0 or now - last_checkpoint >= args.checkpoint_seconds:
            checkpoint(
                args=args,
                run_dir=run_dir,
                records=records,
                generation=0,
                generation_evaluated=index,
                generation_total=len(seeds),
                started=started,
                complete=False,
            )
            last_checkpoint = now
    beam = select_beam(seed_ids, records, args.beam_width)
    atomic_json(run_dir / "beam-generation-0.json", beam)

    for generation in range(1, args.generations + 1):
        neighbor_data: dict[
            str, tuple[TwistedHalfSwapCandidate, str, dict[str, Any]]
        ] = {}
        for parent in beam:
            parent_candidate = TwistedHalfSwapCandidate.from_dict(parent["candidate"])
            if args.mutation_radius == 1:
                neighbors = twisted_support_neighbors(parent_candidate)
            else:
                parent_seed = int(parent_candidate.candidate_id[-16:], 16)
                neighbors = sample_twisted_double_neighbors(
                    parent_candidate,
                    count=args.neighbors_per_parent,
                    seed=args.mutation_seed ^ parent_seed ^ (generation << 32),
                )
            for neighbor, mutation in neighbors:
                neighbor_data.setdefault(
                    neighbor.candidate_id,
                    (neighbor, parent_candidate.candidate_id, mutation),
                )
        ordered = [neighbor_data[key] for key in sorted(neighbor_data)]
        evaluated = 0
        for candidate, parent_id, mutation in ordered:
            evaluated += 1
            if candidate.candidate_id not in records:
                record = evaluate_candidate(
                    candidate,
                    generation=generation,
                    parent_id=parent_id,
                    mutation=mutation,
                    args=args,
                    hits_path=hits_path,
                    run_dir=run_dir,
                )
                records[candidate.candidate_id] = record
                append_jsonl(records_path, record)
            now = time.perf_counter()
            if (
                evaluated % args.progress_every == 0
                or now - last_checkpoint >= args.checkpoint_seconds
            ):
                checkpoint(
                    args=args,
                    run_dir=run_dir,
                    records=records,
                    generation=generation,
                    generation_evaluated=evaluated,
                    generation_total=len(ordered),
                    started=started,
                    complete=False,
                )
                last_checkpoint = now
        pool_ids = [record["candidate_id"] for record in beam] + list(neighbor_data)
        beam = select_beam(pool_ids, records, args.beam_width)
        atomic_json(run_dir / f"beam-generation-{generation}.json", beam)

    checkpoint(
        args=args,
        run_dir=run_dir,
        records=records,
        generation=args.generations,
        generation_evaluated=0,
        generation_total=0,
        started=started,
        complete=True,
    )


if __name__ == "__main__":
    main()
