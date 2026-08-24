#!/usr/bin/env python3
"""Collect, rank, and summarize GALA polynomial-search results."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter
from typing import Any

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _support_key(record: dict[str, Any]) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    support = record.get("canonical_support")
    candidate = record.get("candidate", {})
    if support is None or "m" not in candidate:
        return None
    return int(candidate["m"]), tuple(tuple(term) for term in support)


def load_certifications(
    path: pathlib.Path | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[int, tuple[tuple[int, int], ...]], dict[str, Any]],
]:
    if path is None or not path.exists():
        return {}, {}
    records = load_jsonl(path)
    by_id = {record["candidate_id"]: record for record in records}
    by_support = {
        support: record
        for record in records
        if (support := _support_key(record)) is not None
    }
    return by_id, by_support


def ranking_key(record: dict[str, Any]) -> tuple[Any, ...]:
    certified = record.get("certified_distance")
    bound = record.get("distance_upper_bound")
    return (
        certified is not None,
        certified if certified is not None else -1,
        bound if bound is not None else -1,
        record.get("tanner_girth") or -1,
        -record.get("num_four_cycles", 10**12),
        -record["candidate"]["m"],
        record["candidate_id"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=pathlib.Path, default=DEFAULT_RESULTS_DIR / "weight8.jsonl"
    )
    parser.add_argument(
        "--certifications",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR / "certifications.jsonl",
    )
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_RESULTS_DIR / "weight8.csv")
    parser.add_argument(
        "--markdown", type=pathlib.Path, default=DEFAULT_RESULTS_DIR / "summary.md"
    )
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    certifications_by_id, certifications_by_support = load_certifications(args.certifications)
    for record in records:
        certification = certifications_by_id.get(record["candidate_id"])
        if certification is None:
            certification = certifications_by_support.get(_support_key(record))
        if certification:
            ilp_bound = certification.get("ilp_distance_upper_bound")
            if ilp_bound is None and certification.get("complete"):
                # Backward compatibility with certificates written before the
                # explicit ILP-upper-bound field was added.
                ilp_bound = certification.get("certified_distance")
            if ilp_bound is not None:
                record["ilp_distance_upper_bound"] = ilp_bound
                old_bound = record.get("distance_upper_bound")
                record["distance_upper_bound"] = (
                    ilp_bound if old_bound is None else min(old_bound, ilp_bound)
                )
            if certification.get("complete"):
                record["certified_distance"] = certification["certified_distance"]

    accepted = sorted(
        (record for record in records if record.get("accepted")),
        key=ranking_key,
        reverse=True,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "m",
        "r",
        "s",
        "t",
        "polynomial",
        "n",
        "k",
        "rank_h",
        "check_weight",
        "qubit_degree",
        "tanner_girth",
        "num_four_cycles",
        "distance_upper_bound",
        "decoder_distance_upper_bound",
        "distance_trials",
        "ilp_distance_upper_bound",
        "certified_distance",
        "total_seconds",
    ]
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in accepted:
            candidate = record["candidate"]
            writer.writerow(
                {
                    field: candidate.get(field, record.get(field, ""))
                    for field in fieldnames
                }
            )

    rejection_counts = Counter(
        reason
        for record in records
        if not record.get("accepted")
        for reason in record.get("rejection_reasons", ["unknown"])
    )
    lines = [
        "# Paired-polynomial GALA search",
        "",
        f"- Evaluated candidates: {len(records)}",
        f"- Algebraically accepted: {len(accepted)}",
        f"- Rejection counts: `{dict(rejection_counts)}`",
        "",
        "Distance values in `distance_upper_bound` are analytic, randomized, or ILP-witnessed "
        "upper bounds. "
        "Only `certified_distance` is exact.",
        "",
        "| Candidate | Polynomial | `[[n,k]]` | Girth | 4-cycles | Bound | Certified |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in accepted[: args.top]:
        lines.append(
            "| `{candidate_id}` | `{polynomial}` | `[[{n},{k}]]` | {girth} | {cycles} | "
            "{bound} | {certified} |".format(
                candidate_id=record["candidate_id"],
                polynomial=record["polynomial"],
                n=record["n"],
                k=record["k"],
                girth=record.get("tanner_girth") or "",
                cycles=record.get("num_four_cycles", ""),
                bound=record.get("distance_upper_bound", ""),
                certified=record.get("certified_distance", ""),
            )
        )
    args.markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(accepted)} accepted candidates to {args.csv}")
    print(f"Wrote ranked summary to {args.markdown}")


if __name__ == "__main__":
    main()
