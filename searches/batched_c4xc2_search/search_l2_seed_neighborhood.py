#!/usr/bin/env python3
"""Checkpointed local search around a compact L=2 regular-grid seed."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any, Iterable

import numpy as np

from .distance import certify_distance_at_least
from .logicals import (
    batch_schemes,
    dress_logicals_for_batches,
    find_regular_logical_modules,
    logical_action_profile,
    orbit_array,
)
from .model import Candidate, Monomial, analyze_structure, build_code

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--radius", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--distance-seconds", type=float, default=20)
    parser.add_argument("--logical-trials", type=int, default=4096)
    parser.add_argument("--logical-limit", type=int, default=32)
    parser.add_argument("--dressing-seconds", type=float, default=15)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=float, default=10)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def load_seed(source: pathlib.Path, candidate_id: str) -> Candidate:
    records_path = source / "candidates.jsonl" if source.is_dir() else source
    with records_path.open() as records:
        for line in records:
            record = json.loads(line)
            if record["candidate_id"] != candidate_id:
                continue
            data = record.get("candidate") or record["structure"]["candidate"]
            return Candidate.from_dict(data)
    raise SystemExit(f"candidate not found in source: {candidate_id}")


def mutations(seed: Candidate, radius: int) -> Iterable[tuple[Candidate, dict[str, Any]]]:
    if seed.top_representation != "s3-natural" or seed.half_blocks != 1:
        raise ValueError("the neighborhood search expects a natural-S3 L=2 seed")
    support = set(seed.entries[0])
    anchor = Monomial(0, 0, 0)
    if anchor not in support:
        raise ValueError("the normalized seed must contain the identity monomial")
    universe = {
        Monomial(top, xx, yy)
        for top in range(6)
        for xx in range(4)
        for yy in range(2)
    }
    removable = sorted(support - {anchor})
    available = sorted(universe - support)
    for removed in itertools.combinations(removable, radius):
        for added in itertools.combinations(available, radius):
            entry = tuple(sorted((support - set(removed)) | set(added)))
            yield (
                Candidate("s3-natural", (entry,)),
                {
                    "removed": [term.__dict__ for term in removed],
                    "added": [term.__dict__ for term in added],
                },
            )


def load_state(path: pathlib.Path) -> tuple[set[str], Counter[str]]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    if not path.exists():
        return seen, counts
    with path.open() as records:
        for line in records:
            record = json.loads(line)
            seen.add(record["candidate_id"])
            counts[record["stage"]] += 1
    return seen, counts


def main() -> None:
    args = parse_args()
    seed = load_seed(args.source, args.candidate_id)
    raw_total = 280 if args.radius == 1 else 16_380
    run_dir = args.results_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "candidates.jsonl"
    hits_path = run_dir / "hits.jsonl"
    seen, counts = load_state(records_path) if args.resume else (set(), Counter())
    config = {
        "source": str(args.source),
        "seed_candidate_id": seed.candidate_id,
        "radius": args.radius,
        "raw_candidates": raw_total,
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
    processed = len(seen)
    for raw_index, (candidate, mutation) in enumerate(
        mutations(seed, args.radius), start=1
    ):
        if candidate.candidate_id in seen:
            continue
        processed += 1
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "raw_index": raw_index,
            "candidate": candidate.to_dict(),
            "mutation": mutation,
        }
        structure = analyze_structure(candidate)
        record["structure"] = structure
        if not structure["accepted"]:
            record["stage"] = "structural_rejected"
        else:
            code = build_code(candidate)
            distance = certify_distance_at_least(
                code,
                6,
                time_limit=args.distance_seconds,
                equal_xz_by_permutation=True,
            )
            record["distance"] = distance
            if not distance["certified"]:
                record["stage"] = "distance_rejected"
            else:
                action = logical_action_profile(
                    code,
                    candidate,
                    random_trials=args.logical_trials,
                    seed=260831 ^ raw_index,
                )
                record["logical_action"] = action
                if not action["full_rank_classes"]:
                    record["stage"] = "no_full_rank_translation_orbit"
                else:
                    modules = find_regular_logical_modules(
                        code,
                        candidate,
                        random_trials=args.logical_trials,
                        seed=260831 ^ raw_index,
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
                            "accepted" if accepted else "batching_failed_or_unresolved"
                        )
                        if accepted:
                            append_jsonl(hits_path, record)
                            np.savez_compressed(
                                run_dir / f"{candidate.candidate_id}-checks.npz",
                                matrix_x=np.asarray(code.matrix_x, dtype=np.uint8),
                                matrix_z=np.asarray(code.matrix_z, dtype=np.uint8),
                            )
        counts[record["stage"]] += 1
        append_jsonl(records_path, record)
        now = time.perf_counter()
        if (
            processed % args.progress_every == 0
            or now - last_checkpoint >= args.checkpoint_seconds
            or processed == raw_total
        ):
            summary = {
                "run_id": args.run_id,
                "complete": processed == raw_total,
                "processed": processed,
                "total": raw_total,
                "counts": dict(counts),
                "seconds": round(now - started, 6),
            }
            atomic_json(run_dir / "summary.json", summary)
            print(json.dumps(summary), flush=True)
            last_checkpoint = now


if __name__ == "__main__":
    main()
