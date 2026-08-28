#!/usr/bin/env python3
"""Scan saved [[32,4,5]] seeds for nontrivial physical ZX folds."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time
from collections import Counter, deque
from typing import Any, Iterable

import numpy as np


LOGICAL_ORDER = 4
THICKNESS = 8
ACTIVE_THICKNESS = 7
NUM_QUBITS = LOGICAL_ORDER * THICKNESS


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


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def involutions(items: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    if not items:
        yield ()
        return
    first = items[0]
    rest = items[1:]
    for tail in involutions(rest):
        mapping = {item: image for item, image in zip(rest, tail)}
        mapping[first] = first
        yield tuple(mapping[index] for index in items)
    for offset, partner in enumerate(rest):
        remaining = rest[:offset] + rest[offset + 1 :]
        for tail in involutions(remaining):
            mapping = {item: image for item, image in zip(remaining, tail)}
            mapping[first] = partner
            mapping[partner] = first
            yield tuple(mapping[index] for index in items)


def fold_permutation(thickness_images: tuple[int, ...], epsilon: int) -> np.ndarray:
    images = thickness_images + (ACTIVE_THICKNESS,)
    output = np.empty(NUM_QUBITS, dtype=int)
    for logical in range(LOGICAL_ORDER):
        for thickness in range(THICKNESS):
            source = logical * THICKNESS + thickness
            target = (epsilon * logical) % LOGICAL_ORDER * THICKNESS + images[thickness]
            output[source] = target
    return output


def act_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def logical_columns() -> np.ndarray:
    output = np.zeros((LOGICAL_ORDER, NUM_QUBITS), dtype=np.uint8)
    for logical in range(LOGICAL_ORDER):
        output[logical, logical * THICKNESS : logical * THICKNESS + ACTIVE_THICKNESS] = 1
    return output


def pairing_is_permutation(pairing: np.ndarray) -> bool:
    return bool(np.all(pairing.sum(axis=0) == 1) and np.all(pairing.sum(axis=1) == 1))


def tanner_connected(hx: np.ndarray, hz: np.ndarray) -> bool:
    checks = np.vstack([hx, hz])
    size = len(checks) + NUM_QUBITS
    adjacency = [set() for _ in range(size)]
    for row_index, row in enumerate(checks):
        for qubit in np.flatnonzero(row):
            qnode = len(checks) + int(qubit)
            adjacency[row_index].add(qnode)
            adjacency[qnode].add(row_index)
    reached = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return len(reached) == size


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray(
        [[int(mask) >> qubit & 1 for qubit in range(NUM_QUBITS)] for mask in masks],
        dtype=np.uint8,
    )


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
    thickness_folds = tuple(involutions(tuple(range(ACTIVE_THICKNESS))))
    folds = [
        {
            "epsilon": epsilon,
            "thickness_images": images,
            "permutation": fold_permutation(images, epsilon),
        }
        for epsilon in (1, -1)
        for images in thickness_folds
        if not (epsilon == 1 and images == tuple(range(ACTIVE_THICKNESS)))
    ]
    logicals = logical_columns()
    hits: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for start_index, source in enumerate(saved):
        hx = matrix_from_masks(source["stabilizer_masks"])
        for fold_index, fold in enumerate(folds):
            counters["pairs_tested"] += 1
            permutation = fold["permutation"]
            hz = act_rows(hx, permutation)
            if np.any((hx @ hz.T) % 2):
                continue
            counters["css_folded"] += 1
            x_logicals = act_rows(logicals, permutation)
            pairing = (logicals @ x_logicals.T) % 2
            if not pairing_is_permutation(pairing):
                continue
            counters["logical_pairing_permutation"] += 1
            stacked_rank = gf2_rank(np.vstack([hx, hz]))
            distinct_rowspaces = stacked_rank > gf2_rank(hx)
            if distinct_rowspaces:
                counters["distinct_check_spaces"] += 1
            connected = tanner_connected(hx, hz)
            if connected:
                counters["tanner_connected"] += 1
            hit = {
                "start_index": start_index,
                "fold_index": fold_index,
                "epsilon": int(fold["epsilon"]),
                "thickness_images": list(fold["thickness_images"]),
                "pairing": pairing.astype(int).tolist(),
                "distinct_check_spaces": distinct_rowspaces,
                "tanner_connected": connected,
                "n": 32,
                "k": 4,
                "distance_x": 5,
                "distance_z": 5,
                "maximum_check_weight": int(source["maximum_check_weight"]),
                "source": source,
            }
            hits.append(hit)
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
            item["fold_index"],
        )
    )
    write_jsonl(args.output / "folded-seeds.jsonl", hits)
    summary = {
        "target": "nontrivial permutation-ZX folds of saved [[32,4,5]] seeds",
        "input": str(args.input),
        "starts": len(saved),
        "involutions_on_active_thickness": len(thickness_folds),
        "folds_per_start": len(folds),
        "counters": dict(sorted(counters.items())),
        "hits": len(hits),
        "distinct_connected_hits": sum(
            hit["distinct_check_spaces"] and hit["tanner_connected"] for hit in hits
        ),
        "seconds": round(time.perf_counter() - started, 6),
        "scope": "zero internal logical-coordinate shifts; spectator fibre fixed",
    }
    atomic_json(args.output / "summary.json", summary)
    atomic_json(
        args.output / "progress.json", {"status": "complete", **summary}
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
