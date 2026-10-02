#!/usr/bin/env python3
"""Derive and exhaust folded gauge fixes of the C2 quotient of [[32,4,6]]."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESENTATION = (
    ROOT
    / "codes/n32_k4_d6_reference_code"
    / "schedule"
    / "all_weight8_translation_symmetric_v1"
    / "presentation.json"
)
DEFAULT_CANDIDATE = (
    ROOT
    / "codes/n32_k4_d6_reference_code"
    / "results"
    / "folded-local-20x250-260828-v1"
    / "start-results"
    / "start-010.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "inverse-c2-minimum"
    / "n32-c4-to-c2-quotient-260830-v1"
    / "analysis.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def support_to_mask(support: Iterable[int]) -> int:
    return sum(1 << int(index) for index in support)


def mask_to_support(mask: int, length: int) -> list[int]:
    return [index for index in range(length) if mask >> index & 1]


def matrix_to_masks(matrix: np.ndarray) -> list[int]:
    return [support_to_mask(np.flatnonzero(row)) for row in matrix]


def independent_masks(rows: Iterable[int]) -> list[int]:
    pivots: dict[int, int] = {}
    for row in rows:
        reduced = int(row)
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot in pivots:
                reduced ^= pivots[pivot]
            else:
                pivots[pivot] = reduced
                break
    return list(pivots.values())


def span_masks(basis: Iterable[int]) -> list[int]:
    span = [0]
    for vector in basis:
        span.extend(value ^ int(vector) for value in tuple(span))
    return span


def rowspace(matrix: np.ndarray) -> set[int]:
    return set(span_masks(independent_masks(matrix_to_masks(matrix))))


def rank(matrix: np.ndarray) -> int:
    return len(independent_masks(matrix_to_masks(matrix)))


def permutation_orbits(permutation: list[int]) -> list[list[int]]:
    unseen = set(range(len(permutation)))
    output: list[list[int]] = []
    while unseen:
        start = min(unseen)
        orbit: list[int] = []
        item = start
        while item not in orbit:
            orbit.append(item)
            unseen.remove(item)
            item = permutation[item]
        output.append(orbit)
    return output


def square(permutation: list[int]) -> list[int]:
    return [permutation[permutation[index]] for index in range(len(permutation))]


def quotient_support(support: Iterable[int], orbit_index: dict[int, int]) -> int:
    output = 0
    for index in support:
        output ^= 1 << orbit_index[int(index)]
    return output


def permute_mask(mask: int, permutation: list[int]) -> int:
    return sum(
        ((mask >> index) & 1) << permutation[index]
        for index in range(len(permutation))
    )


def commutes(mask: int, checks: Iterable[int]) -> bool:
    return all(not ((mask & check).bit_count() & 1) for check in checks)


def exact_css_distance(
    commuting_checks: Iterable[int], stabilizers: Iterable[int], length: int
) -> tuple[int, int]:
    stabilizer_space = set(span_masks(independent_masks(stabilizers)))
    check_basis = independent_masks(commuting_checks)
    best = length + 1
    operator = 0
    for candidate in range(1, 1 << length):
        weight = candidate.bit_count()
        if weight >= best or candidate in stabilizer_space:
            continue
        if commutes(candidate, check_basis):
            best = weight
            operator = candidate
    return best, operator


def logical_coset_distance(logicals: list[int], stabilizers: set[int]) -> int:
    combinations: list[int] = []
    for label in range(1, 1 << len(logicals)):
        logical = 0
        for index, vector in enumerate(logicals):
            if label >> index & 1:
                logical ^= vector
        combinations.append(logical)
    return min(
        (logical ^ stabilizer).bit_count()
        for logical in combinations
        for stabilizer in stabilizers
    )


def main() -> None:
    arguments = parse_args()
    presentation = json.loads(arguments.presentation.read_text())
    candidate = json.loads(arguments.candidate.read_text())["result"]["best"]

    physical_translation = [(index + 8) % 32 for index in range(32)]
    data_orbits = permutation_orbits(square(physical_translation))
    if any(len(orbit) != 2 for orbit in data_orbits):
        raise ValueError("the square of the C4 data translation must act freely")
    orbit_index = {
        physical: quotient for quotient, orbit in enumerate(data_orbits) for physical in orbit
    }
    quotient_length = len(data_orbits)

    quotient_checks: dict[str, list[int]] = {}
    check_orbits: dict[str, list[list[int]]] = {}
    for pauli in ("x", "z"):
        action = presentation[f"translation_action_on_{pauli}_checks"]
        orbits = permutation_orbits(square(action))
        check_orbits[pauli] = orbits
        quotient_checks[pauli] = [
            quotient_support(presentation[f"checks_{pauli}"][orbit[0]], orbit_index)
            for orbit in orbits
        ]

    hx = independent_masks(quotient_checks["x"])
    hz = independent_masks(quotient_checks["z"])
    sx = set(span_masks(hx))
    sz = set(span_masks(hz))

    physical_fold = presentation["fold_permutation"]
    quotient_fold: list[int] = []
    for orbit in data_orbits:
        images = {orbit_index[physical_fold[index]] for index in orbit}
        if len(images) != 1:
            raise ValueError("the saved fold does not descend through the C2 quotient")
        quotient_fold.append(images.pop())
    quotient_translation = [orbit_index[physical_translation[orbit[0]]] for orbit in data_orbits]

    logical_z = [
        quotient_support(candidate["logical_supports_z"][index], orbit_index)
        for index in (0, 1)
    ]
    logical_x = [permute_mask(mask, quotient_fold) for mask in logical_z]

    quotient_dx, quotient_x_operator = exact_css_distance(hz, hx, quotient_length)
    quotient_dz, quotient_z_operator = exact_css_distance(hx, hz, quotient_length)

    # The quotient has rank six per CSS type. Exhaust every distinct rank-seven
    # folded extension H_X+<v>, H_Z+<P(v)> which preserves the intended logicals
    # and the physical C2 translation. Coset representatives v+h define the
    # same extension, so at most 2^(16-rank(H_X)) cases need detailed checks.
    extensions: list[dict[str, Any]] = []
    seen_cosets: set[frozenset[int]] = set()
    for vector in range(1, 1 << quotient_length):
        if vector in sx or not commutes(vector, hz):
            continue
        if any((vector & logical).bit_count() & 1 for logical in logical_z):
            continue
        coset = frozenset(vector ^ stabilizer for stabilizer in sx)
        if coset in seen_cosets:
            continue
        seen_cosets.add(coset)

        folded = permute_mask(vector, quotient_fold)
        extended_x = sx | set(coset)
        folded_coset = {folded ^ stabilizer for stabilizer in sz}
        extended_z = sz | folded_coset
        if folded in sz or not commutes(folded, hx):
            continue
        if (vector & folded).bit_count() & 1:
            continue
        if permute_mask(vector, quotient_translation) not in extended_x:
            continue
        if permute_mask(folded, quotient_translation) not in extended_z:
            continue
        if any(logical in extended_z for logical in logical_z):
            continue
        if any(logical in extended_x for logical in logical_x):
            continue

        extended_hx = independent_masks([*hx, vector])
        extended_hz = independent_masks([*hz, folded])
        distance_x, operator_x = exact_css_distance(
            extended_hz, extended_hx, quotient_length
        )
        distance_z, operator_z = exact_css_distance(
            extended_hx, extended_hz, quotient_length
        )
        extensions.append(
            {
                "distance": min(distance_x, distance_z),
                "distance_x": distance_x,
                "distance_z": distance_z,
                "extension_x": mask_to_support(vector, quotient_length),
                "extension_z": mask_to_support(folded, quotient_length),
                "minimum_x_operator": mask_to_support(operator_x, quotient_length),
                "minimum_z_operator": mask_to_support(operator_z, quotient_length),
                "logical_x_coset_distance": logical_coset_distance(
                    logical_x, set(span_masks(extended_hx))
                ),
                "logical_z_coset_distance": logical_coset_distance(
                    logical_z, set(span_masks(extended_hz))
                ),
                "maximum_displayed_check_weight": max(
                    [mask.bit_count() for mask in extended_hx + extended_hz]
                ),
                "checks_x": [
                    mask_to_support(mask, quotient_length) for mask in extended_hx
                ],
                "checks_z": [
                    mask_to_support(mask, quotient_length) for mask in extended_hz
                ],
            }
        )

    result = {
        "schema_version": 1,
        "source_code": "[[32,4,6]]",
        "construction": "quotient by the square of the physical C4 translation",
        "data_orbits": data_orbits,
        "check_orbits_under_translation_squared": check_orbits,
        "quotient": {
            "n": quotient_length,
            "rank_x": len(hx),
            "rank_z": len(hz),
            "k": quotient_length - len(hx) - len(hz),
            "distance": min(quotient_dx, quotient_dz),
            "distance_x": quotient_dx,
            "distance_z": quotient_dz,
            "minimum_x_operator": mask_to_support(quotient_x_operator, quotient_length),
            "minimum_z_operator": mask_to_support(quotient_z_operator, quotient_length),
            "css_orthogonal": commutes(0, [])
            and all(not ((x & z).bit_count() & 1) for x in hx for z in hz),
            "self_dual_check_space": sx == sz,
            "check_weights_x": [mask.bit_count() for mask in quotient_checks["x"]],
            "check_weights_z": [mask.bit_count() for mask in quotient_checks["z"]],
            "checks_x": [
                mask_to_support(mask, quotient_length) for mask in quotient_checks["x"]
            ],
            "checks_z": [
                mask_to_support(mask, quotient_length) for mask in quotient_checks["z"]
            ],
            "logical_z": [mask_to_support(mask, quotient_length) for mask in logical_z],
            "logical_x": [mask_to_support(mask, quotient_length) for mask in logical_x],
            "logical_pairing": [
                [(z & x).bit_count() & 1 for x in logical_x] for z in logical_z
            ],
            "translation": quotient_translation,
            "fold": quotient_fold,
        },
        "folded_rank_seven_extensions": {
            "distinct_extensions": len(extensions),
            "maximum_distance": max(
                (extension["distance"] for extension in extensions), default=None
            ),
            "extensions": extensions,
        },
        "conclusion": (
            "The direct descendant is [[16,4,2]]. The unique compatible folded "
            "rank-seven gauge fixing is [[16,2,3]], so this quotient route cannot "
            "inherit a [[16,2,6]] code from the saved [[32,4,6]] instance."
        ),
    }
    atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
