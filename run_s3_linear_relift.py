#!/usr/bin/env python3
"""Fast-screen saved S3 polynomials through the two-dimensional SL(2,2) lift."""

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

from gala_search.s3_linear import analyze_linear_record, infer_family

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = PROJECT_DIR / "results" / "s3-linear-fast-falsification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--skip-weight-four-screen", action="store_true")
    return parser.parse_args()


def _worker(record: dict[str, Any], screen_weight_four: bool) -> dict[str, Any]:
    try:
        return analyze_linear_record(
            record, screen_weight_four=screen_weight_four
        )
    except Exception as error:
        return {
            "source_candidate_id": record.get("candidate_id"),
            "accepted": False,
            "rejection_reasons": ["worker_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _status(record: dict[str, Any]) -> str:
    return (
        "survivor"
        if record.get("accepted")
        else record.get("rejection_reasons", ["unknown"])[0]
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    summary_path = output_path.with_suffix(".summary.json")
    for path in (output_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    source_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    for input_path in args.input:
        for line in input_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                family = infer_family(record)
            except ValueError:
                source_counts["unsupported_family"] += 1
                continue
            key = (family, str(record.get("candidate_id")))
            if key in seen:
                source_counts["duplicate"] += 1
                continue
            seen.add(key)
            source_counts[str(input_path)] += 1
            source_records.append(record)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _worker, record, not args.skip_weight_four_screen
            )
            for record in source_records
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(futures):
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "total": len(futures),
                            "statuses": Counter(_status(item) for item in results),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(
        key=lambda record: (
            str(record.get("family")), str(record.get("source_candidate_id"))
        )
    )
    status_counts = Counter(_status(record) for record in results)
    family_counts = Counter(str(record.get("family")) for record in results)
    survivor_parameters = Counter(
        (
            record.get("n"),
            record.get("k"),
            record.get("maximum_check_weight"),
        )
        for record in results
        if record.get("accepted")
    )
    summary = {
        "inputs": [str(path) for path in args.input],
        "records": len(results),
        "source_counts": source_counts,
        "family_counts": family_counts,
        "status_counts": status_counts,
        "survivor_parameter_counts": {
            f"n={n},k={k},w={weight}": count
            for (n, k, weight), count in sorted(survivor_parameters.items())
        },
        "weight_four_screened": not args.skip_weight_four_screen,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(output_path)
    with tempfile.NamedTemporaryFile("w", dir=summary_path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    print(json.dumps({"output": str(output_path), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
