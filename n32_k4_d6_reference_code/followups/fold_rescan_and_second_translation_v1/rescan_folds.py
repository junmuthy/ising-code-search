#!/usr/bin/env python3
"""Exhaust the C4-normalizing affine ZX folds of the saved [[32,4,6]] H_X."""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOGICAL_ORDER = 4
THICKNESS = 8
NUM_QUBITS = LOGICAL_ORDER * THICKNESS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def masks_from_supports(supports: list[list[int]]) -> list[int]:
    return [sum(1 << int(qubit) for qubit in support) for support in supports]


def matrix_from_masks(masks: list[int]) -> np.ndarray:
    return np.asarray(
        [[(int(mask) >> qubit) & 1 for qubit in range(NUM_QUBITS)] for mask in masks],
        dtype=np.uint8,
    )


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    pivots: list[int] = []
    rank = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        pivots.append(column)
        rank += 1
        if rank == work.shape[0]:
            break
    return work[:rank], pivots


def nullspace_masks(matrix: np.ndarray) -> list[int]:
    rref, pivots = gf2_rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in pivots]
    output = []
    for column in free:
        vector = 1 << column
        for row, pivot in enumerate(pivots):
            if rref[row, column]:
                vector |= 1 << pivot
        output.append(vector)
    return output


def span_masks(basis: list[int]) -> list[int]:
    output = [0]
    for vector in basis:
        output.extend(value ^ vector for value in tuple(output))
    return output


def independent_rows(masks: list[int]) -> list[int]:
    basis: dict[int, int] = {}
    for value in masks:
        reduced = int(value)
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot in basis:
                reduced ^= basis[pivot]
            else:
                basis[pivot] = reduced
                break
    return list(basis.values())


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


def shift_vectors(thickness_permutation: tuple[int, ...], epsilon: int) -> Iterable[tuple[int, ...]]:
    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for first in range(THICKNESS):
        if first in visited:
            continue
        second = thickness_permutation[first]
        if second == first:
            components.append((first,))
            visited.add(first)
        else:
            components.append((first, second))
            visited.update((first, second))
    choices = [
        ((0, 2) if epsilon == 1 else (0, 1, 2, 3))
        if len(component) == 1
        else (0, 1, 2, 3)
        for component in components
    ]
    for values in itertools.product(*choices):
        shifts = [0] * THICKNESS
        for component, value in zip(components, values, strict=True):
            shifts[component[0]] = value
            if len(component) == 2:
                shifts[component[1]] = (-value if epsilon == 1 else value) % LOGICAL_ORDER
        yield tuple(shifts)


def affine_permutation(
    thickness_permutation: tuple[int, ...], shifts: tuple[int, ...], epsilon: int
) -> list[int]:
    return [
        ((epsilon * logical + shifts[thickness]) % LOGICAL_ORDER) * THICKNESS
        + thickness_permutation[thickness]
        for logical in range(LOGICAL_ORDER)
        for thickness in range(THICKNESS)
    ]


def transform_mask(mask: int, permutation: list[int]) -> int:
    output = 0
    remaining = int(mask)
    while remaining:
        bit = remaining & -remaining
        output |= 1 << permutation[bit.bit_length() - 1]
        remaining ^= bit
    return output


def transform_masks(masks: list[int], permutation: list[int]) -> list[int]:
    return [transform_mask(mask, permutation) for mask in masks]


def pairing_matrix(left: list[int], right: list[int]) -> list[list[int]]:
    return [[(a & b).bit_count() % 2 for b in right] for a in left]


def pairing_is_permutation(pairing: list[list[int]]) -> bool:
    return all(sum(row) == 1 for row in pairing) and all(
        sum(pairing[row][column] for row in range(LOGICAL_ORDER)) == 1
        for column in range(LOGICAL_ORDER)
    )


def folded_css(hx_basis: list[int], hz_basis: list[int]) -> bool:
    return all(not ((left & right).bit_count() % 2) for left in hx_basis for right in hz_basis)


def exact_distance_through_seven(
    permutation: list[int],
    low_kernel_vectors: list[tuple[int, int]],
    stabilizer_span: set[int],
) -> int | None:
    for weight, vector in low_kernel_vectors:
        if transform_mask(vector, permutation) not in stabilizer_span:
            return weight
    return None


def connected(checks_x: list[int], checks_z: list[int]) -> bool:
    checks = checks_x + checks_z
    adjacency = [set() for _ in range(len(checks) + NUM_QUBITS)]
    for row, mask in enumerate(checks):
        for data in range(NUM_QUBITS):
            if mask >> data & 1:
                adjacency[row].add(len(checks) + data)
                adjacency[len(checks) + data].add(row)
    reached = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return len(reached) == len(adjacency)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    source = json.loads(args.input.read_text())["result"]["best"]
    presentation = json.loads(args.presentation.read_text())
    hx_presentation = masks_from_supports(presentation["checks_x"])
    hx_basis = independent_rows([int(mask) for mask in source["stabilizer_masks_x"]])
    if len(hx_basis) != 14:
        raise RuntimeError("saved H_X does not have rank 14")
    stabilizer_span = set(span_masks(hx_basis))
    kernel_basis = nullspace_masks(matrix_from_masks(hx_basis))
    kernel = span_masks(kernel_basis)
    low_kernel_vectors = sorted(
        (vector.bit_count(), vector)
        for vector in kernel
        if vector and vector.bit_count() <= 7
    )
    logical_z = masks_from_supports(source["logical_supports_z"])
    degree_x = [sum(mask >> data & 1 for mask in hx_presentation) for data in range(NUM_QUBITS)]

    counters: Counter[str] = Counter()
    distance_histogram: Counter[int | str] = Counter()
    degree_histogram: Counter[int] = Counter()
    hits: list[dict[str, Any]] = []
    started = time.perf_counter()
    next_checkpoint = args.checkpoint_every
    for epsilon in (1, -1):
        for thickness_permutation in involutions(tuple(range(THICKNESS))):
            counters["thickness_involutions"] += 1
            for shifts in shift_vectors(thickness_permutation, epsilon):
                counters["affine_involutive_folds_tested"] += 1
                if args.checkpoint_every and counters["affine_involutive_folds_tested"] >= next_checkpoint:
                    elapsed = time.perf_counter() - started
                    checkpoint = {
                        "status": "scanning",
                        "tested": counters["affine_involutive_folds_tested"],
                        "folded_css": counters["folded_css"],
                        "distance_at_least_six_connected": counters["tanner_connected_distance_at_least_six"],
                        "seconds": round(elapsed, 6),
                    }
                    atomic_json(args.output / "progress.json", checkpoint)
                    print("checkpoint " + json.dumps(checkpoint, sort_keys=True), flush=True)
                    next_checkpoint += args.checkpoint_every
                permutation = affine_permutation(thickness_permutation, shifts, epsilon)
                logical_x = transform_masks(logical_z, permutation)
                pairing = pairing_matrix(logical_z, logical_x)
                if not pairing_is_permutation(pairing):
                    continue
                counters["logical_pairing_permutation"] += 1
                hz_basis = transform_masks(hx_basis, permutation)
                if not folded_css(hx_basis, hz_basis):
                    continue
                counters["folded_css"] += 1
                distance = exact_distance_through_seven(
                    permutation, low_kernel_vectors, stabilizer_span
                )
                distance_label: int | str = distance if distance is not None else ">=8"
                distance_histogram[distance_label] += 1
                if distance is None or distance >= 6:
                    counters["distance_at_least_six"] += 1
                else:
                    continue
                hz_presentation = transform_masks(hx_presentation, permutation)
                if not connected(hx_presentation, hz_presentation):
                    continue
                counters["tanner_connected_distance_at_least_six"] += 1
                combined_degrees = [degree_x[data] + degree_x[permutation[data]] for data in range(NUM_QUBITS)]
                combined_maximum = max(combined_degrees)
                degree_histogram[combined_maximum] += 1
                record = {
                    "epsilon": epsilon,
                    "thickness_permutation": list(thickness_permutation),
                    "shifts": list(shifts),
                    "permutation": permutation,
                    "pairing": pairing,
                    "distance": distance_label,
                    "distinct_check_spaces": any(mask not in stabilizer_span for mask in hz_basis),
                    "tanner_connected": True,
                    "combined_data_degrees": combined_degrees,
                    "combined_data_degree_histogram": dict(sorted(Counter(combined_degrees).items())),
                    "combined_collision_lower_bound": max(8, combined_maximum),
                }
                hits.append(record)


    hits.sort(
        key=lambda item: (
            item["combined_collision_lower_bound"],
            not item["distinct_check_spaces"],
            item["epsilon"],
            item["thickness_permutation"],
            item["shifts"],
        )
    )
    with (args.output / "distance-six-folds.jsonl").open("w") as handle:
        for hit in hits:
            handle.write(json.dumps(hit, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "scope": (
            "all involutive affine folds P(a,t)=(epsilon*a+s_t,pi(t)) "
            "normalizing the physical C4 translation, with arbitrary involutive pi on all eight thickness fibres"
        ),
        "input": str(args.input.resolve()),
        "presentation": str(args.presentation.resolve()),
        "kernel_dimension": len(kernel_basis),
        "low_kernel_vector_count_through_weight_seven": len(low_kernel_vectors),
        "counters": dict(sorted(counters.items())),
        "distance_histogram_among_folded_css": dict(
            sorted(distance_histogram.items(), key=lambda item: str(item[0]))
        ),
        "combined_collision_lower_bound_histogram_for_connected_distance_ge_six": dict(sorted(degree_histogram.items())),
        "best_combined_collision_lower_bound": min(
            (item["combined_collision_lower_bound"] for item in hits), default=None
        ),
        "distance_at_least_six_connected_hits": len(hits),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", **summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
