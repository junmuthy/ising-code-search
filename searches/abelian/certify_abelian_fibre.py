#!/usr/bin/env python3
"""Certify saved structurally accepted order-seven-fibre candidates."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import Any

from gala_search.abelian_fibre_codes import FibreCodeCandidate, certify_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", type=int, default=None)
    parser.add_argument("--ansatz", choices=("weight8", "weight12"))
    parser.add_argument("--stop-after-seven", action="store_true")
    return parser.parse_args()


def _candidate_from_record(record: dict[str, Any]) -> FibreCodeCandidate:
    data = record["candidate"]
    return FibreCodeCandidate(
        logical_x_order=int(data["logical_x_order"]),
        ansatz=str(data["ansatz"]),
        parameters=tuple(map(int, data["parameters"])),
        support=tuple(tuple(map(int, term)) for term in data["support"]),
    )


def _write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    records = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    accepted = [
        record
        for record in records
        if record.get("accepted")
        and (
            args.ansatz is None
            or record.get("candidate", {}).get("ansatz") == args.ansatz
        )
    ]
    if args.maximum is not None:
        accepted = accepted[: args.maximum]
    certifications = []
    for index, record in enumerate(accepted, start=1):
        candidate = _candidate_from_record(record)
        try:
            certificate = certify_distance(candidate)
        except Exception as error:
            certificate = {
                "candidate": candidate.to_dict(),
                "candidate_id": candidate.candidate_id,
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
            }
        certifications.append(certificate)
        _write_jsonl(args.output, certifications)
        print(
            json.dumps(
                {
                    "completed": index,
                    "candidate_id": candidate.candidate_id,
                    "distance": certificate.get("certified_distance"),
                    "complete": certificate.get("complete"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.stop_after_seven and certificate.get("certified_distance") == 7:
            break


if __name__ == "__main__":
    main()
