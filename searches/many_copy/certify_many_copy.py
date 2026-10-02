#!/usr/bin/env python3
"""Certify the natural connected four-copy packed-code seed."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

from gala_search.many_copy import (
    analyze_many_copy_candidate,
    certify_many_copy_distance,
    many_copy_automorphism_checks,
    many_copy_cnot_schedule,
    natural_four_copy_candidate,
)

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="HIGHS")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_DIR
        / "results"
        / "many-copy"
        / "natural-seed-certificate.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = natural_four_copy_candidate()
    structure = analyze_many_copy_candidate(candidate)
    automorphisms = many_copy_automorphism_checks(candidate)
    schedule = many_copy_cnot_schedule(candidate)
    distance = certify_many_copy_distance(
        candidate, solver=args.solver, target_distance=7
    )
    if (
        not structure["accepted"]
        or not all(automorphisms.values())
        or not schedule["optimal"]
        or distance["certified_distance"] != 7
    ):
        raise RuntimeError("the natural four-copy seed failed recertification")
    record = {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "polynomial": candidate.polynomial,
        "parameters": {
            "n": structure["n"],
            "k": structure["k"],
            "d": distance["certified_distance"],
        },
        "structure": structure,
        "automorphisms": automorphisms,
        "cnot_schedule": schedule,
        "distance": distance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        output.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **record["parameters"]}, sort_keys=True))


if __name__ == "__main__":
    main()
