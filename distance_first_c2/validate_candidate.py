#!/usr/bin/env python3
"""Independently brute-force a saved small CSS candidate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


def independent_basis(rows: Iterable[int]) -> tuple[int, ...]:
    pivots: dict[int, int] = {}
    for original in rows:
        value = int(original)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return tuple(pivots.values())


def independent_span(rows: Iterable[int]) -> set[int]:
    output = {0}
    for row in independent_basis(rows):
        output |= {value ^ row for value in tuple(output)}
    return output


def exact_distance(
    commuting_checks: Iterable[int], stabilizers: Iterable[int], n: int
) -> tuple[int, int]:
    checks = independent_basis(commuting_checks)
    stabilizer_space = independent_span(stabilizers)
    for weight in range(1, n + 1):
        for operator in range(1, 1 << n):
            if operator.bit_count() != weight or operator in stabilizer_space:
                continue
            if all(not ((operator & check).bit_count() & 1) for check in checks):
                return weight, operator
    return n + 1, 0


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    candidate = record.get(
        "best", record.get("model_d5", record.get("survivor", record))
    )
    n = int(candidate["n"])
    basis_x = tuple(map(int, candidate["basis_x_masks"]))
    basis_z = tuple(map(int, candidate["basis_z_masks"]))
    rank_x = len(independent_basis(basis_x))
    rank_z = len(independent_basis(basis_z))
    distance_x, witness_x = exact_distance(basis_z, basis_x, n)
    distance_z, witness_z = exact_distance(basis_x, basis_z, n)
    result = {
        "source": str(args.input),
        "n": n,
        "k": n - rank_x - rank_z,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "css_orthogonal": all(
            not ((left & right).bit_count() & 1)
            for left in basis_x
            for right in basis_z
        ),
        "distance_x": distance_x,
        "distance_z": distance_z,
        "distance": min(distance_x, distance_z),
        "witness_x": witness_x,
        "witness_z": witness_z,
        "witness_x_support": [bit for bit in range(n) if witness_x >> bit & 1],
        "witness_z_support": [bit for bit in range(n) if witness_z >> bit & 1],
    }
    result["matches_search"] = (
        result["k"] == int(candidate["k"])
        and result["distance_x"] == int(candidate["distance_x"])
        and result["distance_z"] == int(candidate["distance_z"])
        and result["css_orthogonal"]
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["matches_search"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
