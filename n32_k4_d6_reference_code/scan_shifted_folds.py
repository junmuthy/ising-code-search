#!/usr/bin/env python3
"""Scan saved [[32,4,5]] seeds for thickness-dependent C4 shifts."""

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

from scan_frontier_folds import (
    ACTIVE_THICKNESS,
    LOGICAL_ORDER,
    NUM_QUBITS,
    THICKNESS,
    act_rows,
    gf2_rank,
    logical_columns,
    matrix_from_masks,
    pairing_is_permutation,
    tanner_connected,
)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, values: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def shifted_fold(shifts: tuple[int, ...], epsilon: int) -> np.ndarray:
    output = np.empty(NUM_QUBITS, dtype=int)
    for logical in range(LOGICAL_ORDER):
        for thickness in range(THICKNESS):
            source = logical * THICKNESS + thickness
            target = (
                (epsilon * logical + shifts[thickness]) % LOGICAL_ORDER
            ) * THICKNESS + thickness
            output[source] = target
    return output


def shifted_fold_catalog() -> list[dict[str, Any]]:
    logicals = logical_columns()
    output = []
    for epsilon, choices in ((1, (0, 2)), (-1, (0, 1, 2, 3))):
        for shifts in itertools.product(choices, repeat=THICKNESS):
            if epsilon == 1 and not any(shifts):
                continue
            permutation = shifted_fold(tuple(shifts), epsilon)
            if not np.array_equal(permutation[permutation], np.arange(NUM_QUBITS)):
                raise AssertionError("catalog contains a non-involution")
            pairing = (logicals @ act_rows(logicals, permutation).T) % 2
            if not pairing_is_permutation(pairing):
                continue
            output.append(
                {
                    "epsilon": epsilon,
                    "shifts": tuple(int(value) for value in shifts),
                    "permutation": permutation,
                    "pairing": pairing,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--maximum-starts", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    saved = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    saved = [item for item in saved if int(item["certified_distance"]) == 5]
    saved.sort(
        key=lambda item: (
            int(item["distance"]["weight_counts_through_seven"].get("5", 10**9)),
            int(item["maximum_check_weight"]),
        )
    )
    saved = saved[: args.maximum_starts]
    folds = shifted_fold_catalog()
    hits: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for start_index, source in enumerate(saved):
        hx = matrix_from_masks(source["stabilizer_masks"])
        rank_x = gf2_rank(hx)
        for fold_index, fold in enumerate(folds):
            counters["pairs_tested"] += 1
            hz = act_rows(hx, fold["permutation"])
            if np.any((hx @ hz.T) % 2):
                continue
            counters["css_folded"] += 1
            distinct = gf2_rank(np.vstack([hx, hz])) > rank_x
            if distinct:
                counters["distinct_check_spaces"] += 1
            connected = tanner_connected(hx, hz)
            if connected:
                counters["tanner_connected"] += 1
            hits.append(
                {
                    "start_index": start_index,
                    "fold_index": fold_index,
                    "epsilon": int(fold["epsilon"]),
                    "shifts": list(fold["shifts"]),
                    "pairing": fold["pairing"].astype(int).tolist(),
                    "distinct_check_spaces": distinct,
                    "tanner_connected": connected,
                    "n": 32,
                    "k": 4,
                    "distance_x": 5,
                    "distance_z": 5,
                    "maximum_check_weight": int(source["maximum_check_weight"]),
                    "source": source,
                }
            )
        atomic_json(
            args.output / "progress.json",
            {
                "status": "scanning",
                "completed_starts": start_index + 1,
                "total_starts": len(saved),
                "folds_per_start": len(folds),
                "hits": len(hits),
                "distinct_hits": counters["distinct_check_spaces"],
                "seconds": round(time.perf_counter() - started, 6),
            },
        )
    hits.sort(
        key=lambda item: (
            not item["distinct_check_spaces"],
            not item["tanner_connected"],
            item["maximum_check_weight"],
            item["start_index"],
        )
    )
    write_jsonl(args.output / "folded-seeds.jsonl", hits)
    summary = {
        "target": "thickness-dependent shifted ZX folds of saved [[32,4,5]] seeds",
        "input": str(args.input),
        "starts": len(saved),
        "folds_per_start": len(folds),
        "counters": dict(sorted(counters.items())),
        "hits": len(hits),
        "distinct_connected_hits": sum(
            hit["distinct_check_spaces"] and hit["tanner_connected"] for hit in hits
        ),
        "seconds": round(time.perf_counter() - started, 6),
        "scope": "identity permutation of thickness fibres; all involutive shifts",
    }
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", **summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
