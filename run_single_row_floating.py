#!/usr/bin/env python3
"""Run the floating-k ``n=32`` GL(2,2) x C4 exact-fold search."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import tempfile
import traceback
from collections import Counter
from typing import Any

from gala_search.single_row import polynomial_constraint_matrix
from gala_search.single_row_floating import (
    SCHEMA_VERSION,
    certify_floating_candidate,
    find_second_c4_sector,
    fold_from_record,
    search_floating_constraint_space,
    identity_fold,
    select_self_dual_seed_witnesses,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_DIR / "results" / "single-row" / "n32-floating"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed-witnesses", type=int, default=16)
    parser.add_argument("--seed", type=int, default=320806)
    parser.add_argument("--maximum-check-weight", type=int, default=12)
    parser.add_argument("--maximum-saved-hits-per-k", type=int, default=20)
    parser.add_argument("--maximum-certifications", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2))
    )
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


def _search_task(
    task_index: int,
    witness_index: int,
    witness: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        data_fold = fold_from_record(witness["fold"])
        check_fold = identity_fold(4)
        constraints = polynomial_constraint_matrix(
            witness["seed_support"],
            data_fold=data_fold,
            check_fold=check_fold,
        )
        search = search_floating_constraint_space(
            constraints,
            support=witness["seed_support"],
            data_fold=data_fold,
            check_fold=check_fold,
            maximum_check_weight=arguments["maximum_check_weight"],
            maximum_saved_hits_per_k=arguments["maximum_saved_hits_per_k"],
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness_index": witness_index,
            "witness": witness,
            "search": search,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_index": task_index,
            "witness_index": witness_index,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    output_dir = RESULTS_ROOT / args.run_name
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    witnesses = select_self_dual_seed_witnesses(
        count=args.seed_witnesses, random_seed=args.seed
    )
    _write_jsonl(output_dir / "seed-fold-witnesses.jsonl", witnesses)
    tasks = []
    for witness_index, witness in enumerate(witnesses):
        tasks.append((len(tasks), witness_index, witness))
    print(
        f"Searching {len(tasks)} self-dual identity-fold spaces from "
        f"{len(witnesses)} odd seed witnesses with {args.workers} workers",
        flush=True,
    )
    records_by_index = {}
    arguments = vars(args)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_search_task, *task, arguments): task[0]
            for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records_by_index[record["task_index"]] = record
            if completed % 25 == 0 or completed == len(tasks):
                hits = sum(
                    item.get("search", {}).get("counters", {}).get(
                        "structural_hits", 0
                    )
                    for item in records_by_index.values()
                )
                print(
                    f"Completed {completed}/{len(tasks)} spaces; "
                    f"structural hits={hits}",
                    flush=True,
                )
    records = [records_by_index[index] for index in range(len(tasks))]
    _write_jsonl(output_dir / "generator-spaces.jsonl", records)
    errors = [record for record in records if "error" in record]
    hits = [
        hit
        for record in records
        for hit in record.get("search", {}).get("hits", [])
    ]
    unique_hits = {}
    for hit in hits:
        key = tuple(hit["coefficients"]), tuple(
            hit["data_fold"]["fibre_images"]
        )
        unique_hits.setdefault(key, hit)
    ordered_hits = sorted(
        unique_hits.values(),
        key=lambda hit: (
            not hit["logical_grid"]["pairing_is_permutation"],
            -hit["k"],
            hit["maximum_check_weight"],
            hit["coefficient_weight"],
        ),
    )
    _write_jsonl(output_dir / "structural-candidates.jsonl", ordered_hits)
    certifications = []
    accepted = None
    for hit in ordered_hits[: args.maximum_certifications]:
        certificate = certify_floating_candidate(hit)
        certifications.append(certificate)
        _write_jsonl(output_dir / "distance-certifications.jsonl", certifications)
        print(
            json.dumps(
                {
                    "stage": "distance",
                    "tested": len(certifications),
                    "k": hit["k"],
                    "check_weight": hit["maximum_check_weight"],
                    "distance": certificate["certified_distance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if certificate["accepted"]:
            accepted = certificate
            break
    if accepted is not None:
        accepted["second_c4_sector"] = find_second_c4_sector(
            accepted["candidate"]
        )
        _write_json(output_dir / "recommended-certificate.json", accepted)
    dimension_counts: Counter[int] = Counter()
    for record in records:
        for dimension, count in record.get("search", {}).get(
            "dimension_counts", {}
        ).items():
            dimension_counts[int(dimension)] += int(count)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": "[[32,k,>=6]] with k>=4 and one protected C4 row",
        "arguments": vars(args),
        "witnesses": len(witnesses),
        "generator_spaces": len(tasks),
        "worker_errors": len(errors),
        "structural_candidates": len(ordered_hits),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "distance_certifications": len(certifications),
        "accepted": accepted is not None,
        "recommended": (
            str(output_dir / "recommended-certificate.json")
            if accepted is not None
            else None
        ),
        "output": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
