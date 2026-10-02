#!/usr/bin/env python3
"""Deterministically reanalyze distance-qualified compact L=2 candidates."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

import numpy as np

from .logicals import (
    batch_schemes,
    dress_logicals_for_batches,
    find_regular_logical_modules,
    orbit_array,
)
from .model import Candidate, build_code

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--module-limit", type=int, default=32)
    parser.add_argument("--dressing-seconds", type=float, default=15)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def certified_records(source: pathlib.Path) -> list[dict[str, Any]]:
    records_path = source / "candidates.jsonl" if source.is_dir() else source
    output = []
    with records_path.open() as records:
        for line in records:
            record = json.loads(line)
            if record.get("distance", {}).get("certified") is True:
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
        for record in certified_records(source)
    ]
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    for index, (source, record) in enumerate(all_records, start=1):
        candidate = Candidate.from_dict(record["candidate"])
        code = build_code(candidate)
        item_started = time.perf_counter()
        modules = find_regular_logical_modules(
            code,
            candidate,
            random_trials=0,
            limit=args.module_limit,
            exhaustive_dimension=12,
        )
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
        counts["distance_qualified"] += 1
        counts["regular_logical_module" if modules else "no_regular_logical_module"] += 1
        if accepted:
            counts["accepted"] += 1
        output = {
            "candidate_id": candidate.candidate_id,
            "source_run": str(source),
            "candidate": record["candidate"],
            "structure": record["structure"],
            "distance": record["distance"],
            "logical_modules": modules,
            "dressed_batching": dressed_records,
            "accepted": accepted,
            "seconds": round(time.perf_counter() - item_started, 6),
        }
        append_jsonl(output_path, output)
        if accepted:
            np.savez_compressed(
                run_dir / f"{candidate.candidate_id}-checks.npz",
                matrix_x=np.asarray(code.matrix_x, dtype=np.uint8),
                matrix_z=np.asarray(code.matrix_z, dtype=np.uint8),
            )
        summary = {
            "run_id": args.run_id,
            "complete": index == len(all_records),
            "processed": index,
            "total": len(all_records),
            "counts": dict(counts),
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_json(run_dir / "summary.json", summary)
        print(json.dumps({**summary, "candidate_id": candidate.candidate_id}), flush=True)


if __name__ == "__main__":
    main()
