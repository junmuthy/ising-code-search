#!/usr/bin/env python3
"""Exhaustively validate the saved ``[[36,4,6]]`` coset-2BGA presentation."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np


PRESENTATION = pathlib.Path(__file__).with_name("presentation.json")


def support_matrix(supports: Sequence[Sequence[int]], width: int) -> np.ndarray:
    matrix = np.zeros((len(supports), width), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, list(support)] = 1
    return matrix


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


def row_masks(matrix: np.ndarray) -> list[int]:
    return [
        sum(int(value) << column for column, value in enumerate(row))
        for row in np.asarray(matrix, dtype=np.uint8)
    ]


def span_masks(generators: Sequence[int]) -> list[int]:
    span = [0]
    for generator in generators:
        span += [word ^ int(generator) for word in span]
    return span


def integer_rank(vectors: Sequence[int]) -> int:
    basis: dict[int, int] = {}
    for vector in vectors:
        work = int(vector)
        while work:
            pivot = work.bit_length() - 1
            if pivot in basis:
                work ^= basis[pivot]
            else:
                basis[pivot] = work
                break
    return len(basis)


def permute_matrix(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, list(permutation)] = matrix
    return output


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    rank_left = gf2_rank(left)
    rank_right = gf2_rank(right)
    return rank_left == rank_right == gf2_rank(np.vstack([left, right]))


def disjoint_rows(matrix: np.ndarray) -> bool:
    overlap = matrix @ matrix.T
    return bool(np.array_equal(overlap, np.diag(np.diag(overlap))))


def permutation_order(permutation: Sequence[int]) -> int:
    seen: set[int] = set()
    order = 1
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = int(permutation[current])
            length += 1
        order = math.lcm(order, length)
    return order


def logical_coset_data(
    stabilizers: np.ndarray,
    logicals: np.ndarray,
) -> tuple[int, dict[int, int], int, int]:
    """Return exact distance and weight-six coset statistics."""

    stabilizer_span = span_masks(row_masks(stabilizers))
    logical_masks = row_masks(logicals)
    minima: dict[int, int] = {}
    weight_six_counts: dict[int, int] = {}
    for label in range(1, 1 << len(logical_masks)):
        representative = 0
        for index, logical in enumerate(logical_masks):
            if label >> index & 1:
                representative ^= logical
        minimum = logicals.shape[1] + 1
        count_six = 0
        for stabilizer in stabilizer_span:
            weight = (representative ^ stabilizer).bit_count()
            minimum = min(minimum, weight)
            count_six += weight == 6
        minima[label] = minimum
        if count_six:
            weight_six_counts[label] = count_six
    return (
        min(minima.values()),
        minima,
        sum(weight_six_counts.values()),
        integer_rank(list(weight_six_counts)),
    )


def validate() -> dict[str, object]:
    record = json.loads(PRESENTATION.read_text())
    n = int(record["n"])
    checks_x = support_matrix(record["checks_x"], n)
    checks_z = support_matrix(record["checks_z"], n)
    logical_x = support_matrix(record["logical_supports_x"], n)
    logical_z = support_matrix(record["logical_supports_z"], n)
    p = tuple(int(value) for value in record["c4_permutation"])
    q = tuple(int(value) for value in record["zx_permutation"])

    rank_x = gf2_rank(checks_x)
    rank_z = gf2_rank(checks_z)
    distance_x, minima_x, weight_six_x, weight_six_span_x = logical_coset_data(
        checks_x, logical_x
    )
    distance_z, minima_z, weight_six_z, weight_six_span_z = logical_coset_data(
        checks_z, logical_z
    )

    p_x = permute_matrix(logical_x, p)
    p_z = permute_matrix(logical_z, p)
    expected_next_x = np.roll(logical_x, -1, axis=0)
    expected_next_z = np.roll(logical_z, -1, axis=0)
    z_stabilizer_span = set(span_masks(row_masks(checks_z)))
    p_cycles_z_logically = all(
        (left ^ right) in z_stabilizer_span
        for left, right in zip(
            row_masks(p_z), row_masks(expected_next_z), strict=True
        )
    )

    result: dict[str, object] = {
        "parameters": {
            "n": n,
            "k": n - rank_x - rank_z,
            "distance": min(distance_x, distance_z),
            "distance_x": distance_x,
            "distance_z": distance_z,
            "rank_x": rank_x,
            "rank_z": rank_z,
        },
        "checks": {
            "css_orthogonal": bool(not np.any((checks_x @ checks_z.T) % 2)),
            "maximum_weight_x": int(checks_x.sum(axis=1).max()),
            "maximum_weight_z": int(checks_z.sum(axis=1).max()),
        },
        "logicals": {
            "pairing": ((logical_x @ logical_z.T) % 2).astype(int).tolist(),
            "x_weights": logical_x.sum(axis=1).astype(int).tolist(),
            "z_weights": logical_z.sum(axis=1).astype(int).tolist(),
            "x_disjoint": disjoint_rows(logical_x),
            "z_disjoint": disjoint_rows(logical_z),
            "x_commutes_with_z_checks": bool(
                not np.any((logical_x @ checks_z.T) % 2)
            ),
            "z_commutes_with_x_checks": bool(
                not np.any((logical_z @ checks_x.T) % 2)
            ),
            "x_coset_minima": minima_x,
            "z_coset_minima": minima_z,
            "weight_six_representatives_x": weight_six_x,
            "weight_six_representatives_z": weight_six_z,
            "weight_six_coset_span_dimension_x": weight_six_span_x,
            "weight_six_coset_span_dimension_z": weight_six_span_z,
            "four_independent_weight_six_x_impossible": weight_six_span_x < 4,
            "four_independent_weight_six_z_impossible": weight_six_span_z < 4,
            "minimum_common_cyclic_support_weight": 7,
        },
        "c4": {
            "physical_order": permutation_order(p),
            "logical_order": 4,
            "preserves_x_stabilizer_rowspace": same_rowspace(
                permute_matrix(checks_x, p), checks_x
            ),
            "preserves_z_stabilizer_rowspace": same_rowspace(
                permute_matrix(checks_z, p), checks_z
            ),
            "cycles_x_supports_exactly": bool(
                np.array_equal(p_x, expected_next_x)
            ),
            "cycles_z_logicals_mod_stabilizers": p_cycles_z_logically,
        },
        "zx_hadamard": {
            "physical_order": permutation_order(q),
            "maps_x_checks_to_z_rowspace": same_rowspace(
                permute_matrix(checks_x, q), checks_z
            ),
            "maps_z_checks_to_x_rowspace": same_rowspace(
                permute_matrix(checks_z, q), checks_x
            ),
            "maps_x_logicals_to_z_exactly": bool(
                np.array_equal(permute_matrix(logical_x, q), logical_z)
            ),
            "maps_z_logicals_to_x_exactly": bool(
                np.array_equal(permute_matrix(logical_z, q), logical_x)
            ),
            "gate": "P_q H^tensor36",
            "logical_action": "H^tensor4",
        },
    }

    required = [
        result["parameters"] == {
            "n": 36,
            "k": 4,
            "distance": 6,
            "distance_x": 6,
            "distance_z": 6,
            "rank_x": 16,
            "rank_z": 16,
        },
        result["checks"]["css_orthogonal"],
        result["checks"]["maximum_weight_x"] == 8,
        result["checks"]["maximum_weight_z"] == 8,
        result["logicals"]["pairing"] == np.eye(4, dtype=int).tolist(),
        result["logicals"]["x_weights"] == [7, 7, 7, 7],
        result["logicals"]["z_weights"] == [7, 7, 7, 7],
        result["logicals"]["x_disjoint"],
        result["logicals"]["z_disjoint"],
        result["logicals"]["x_commutes_with_z_checks"],
        result["logicals"]["z_commutes_with_x_checks"],
        result["logicals"]["weight_six_representatives_x"] == 64,
        result["logicals"]["weight_six_coset_span_dimension_x"] == 3,
        result["logicals"]["four_independent_weight_six_x_impossible"],
        result["logicals"]["four_independent_weight_six_z_impossible"],
        result["c4"]["physical_order"] == 8,
        result["c4"]["logical_order"] == 4,
        result["c4"]["preserves_x_stabilizer_rowspace"],
        result["c4"]["preserves_z_stabilizer_rowspace"],
        result["c4"]["cycles_x_supports_exactly"],
        result["c4"]["cycles_z_logicals_mod_stabilizers"],
        result["zx_hadamard"]["physical_order"] == 4,
        result["zx_hadamard"]["maps_x_checks_to_z_rowspace"],
        result["zx_hadamard"]["maps_z_checks_to_x_rowspace"],
        result["zx_hadamard"]["maps_x_logicals_to_z_exactly"],
        result["zx_hadamard"]["maps_z_logicals_to_x_exactly"],
    ]
    result["all_required_checks_pass"] = all(required)
    if not result["all_required_checks_pass"]:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
