#!/usr/bin/env python3
"""Screen compact L=2 structural survivors for the regular grid module."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

from .logicals import find_regular_logical_modules, logical_action_profile
from .model import Candidate, build_code

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--random-trials", type=int, default=4096)
    parser.add_argument("--module-limit", type=int, default=32)
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


def structural_records(source: pathlib.Path) -> list[dict[str, Any]]:
    records_path = source / "candidates.jsonl" if source.is_dir() else source
    output = []
    with records_path.open() as records:
        for line in records:
            record = json.loads(line)
            if record["structure"]["accepted"]:
                output.append(record)
    return output


def main() -> None:
    args = parse_args()
    run_dir = args.results_root / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path = run_dir / "candidates.jsonl"
    all_records = [
        (source, record)
        for source in args.sources
        for record in structural_records(source)
    ]
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    for index, (source, record) in enumerate(all_records, start=1):
        item_started = time.perf_counter()
        candidate = Candidate.from_dict(record["candidate"])
        code = build_code(candidate)
        profile = logical_action_profile(
            code,
            candidate,
            random_trials=args.random_trials,
            seed=260831 ^ int(record["raw_index"]),
        )
        modules = []
        if profile["full_rank_classes"]:
            modules = find_regular_logical_modules(
                code,
                candidate,
                random_trials=args.random_trials,
                seed=260831 ^ int(record["raw_index"]),
                limit=args.module_limit,
            )
        counts["structural_survivor"] += 1
        counts[
            "full_rank_translation_orbit"
            if profile["full_rank_classes"]
            else "no_full_rank_translation_orbit"
        ] += 1
        counts[
            "regular_zx_module" if modules else "no_regular_zx_module"
        ] += 1
        item = {
            "candidate_id": candidate.candidate_id,
            "source_run": str(source),
            "raw_index": record["raw_index"],
            "structure": record["structure"],
            "distance": record.get("distance"),
            "logical_action": profile,
            "logical_modules": modules,
            "seconds": round(time.perf_counter() - item_started, 6),
        }
        append_jsonl(output_path, item)
        summary = {
            "run_id": args.run_id,
            "complete": index == len(all_records),
            "processed": index,
            "total": len(all_records),
            "counts": dict(counts),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(run_dir / "summary.json", summary)
        if index % 10 == 0 or index == len(all_records):
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
