"""Construct immutable Stim code inputs for an enumerated fold and schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from codes.n32_k4_d6_reference_code.stim_fault_distance.model import (
    CodeData,
    ScheduledGate,
    load_code_data,
)


def transform_supports(
    supports: Iterable[Iterable[int]], permutation: Iterable[int]
) -> tuple[tuple[int, ...], ...]:
    mapping = tuple(int(value) for value in permutation)
    return tuple(
        tuple(sorted(mapping[int(qubit)] for qubit in support)) for support in supports
    )


def _commutes(left: Iterable[Iterable[int]], right: Iterable[Iterable[int]]) -> bool:
    right_sets = [set(row) for row in right]
    return all(not (len(set(row) & other) % 2) for row in left for other in right_sets)


def code_for_schedule(schedule: dict[str, Any]) -> CodeData:
    """Transport the saved logical Z grid through a selected ZX fold."""

    canonical = load_code_data()
    permutation = tuple(int(value) for value in schedule["fold"]["permutation"])
    checks_z = transform_supports(canonical.checks_x, permutation)
    logicals_z = canonical.logicals_z
    logicals_x = transform_supports(logicals_z, permutation)
    layers = tuple(
        tuple(
            ScheduledGate(
                kind=str(gate["type"]),
                check=int(gate["check"]),
                data=int(gate["data"]),
            )
            for gate in layer["gates"]
        )
        for layer in schedule["layers"]
    )

    if not _commutes(canonical.checks_x, checks_z):
        raise ValueError("candidate fold is not CSS orthogonal")
    if not _commutes(logicals_z, canonical.checks_x):
        raise ValueError("saved logical Z grid does not commute with H_X")
    if not _commutes(logicals_x, checks_z):
        raise ValueError("folded logical X grid does not commute with H_Z")
    pairing = [
        [len(set(logical_z) & set(logical_x)) % 2 for logical_x in logicals_x]
        for logical_z in logicals_z
    ]
    if not (
        all(sum(row) == 1 for row in pairing)
        and all(sum(pairing[row][column] for row in range(4)) == 1 for column in range(4))
    ):
        raise ValueError("fold does not give a permutation logical ZX pairing")
    if len(layers) != 12 or sum(map(len, layers)) != 256:
        raise ValueError("candidate is not a complete depth-12 schedule")

    identifier = str(schedule.get("schedule_id", "saved-reference"))
    return CodeData(
        checks_x=canonical.checks_x,
        checks_z=checks_z,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        redundant_relations=canonical.redundant_relations,
        fold_permutation=permutation,
        schedule=layers,
        presentation_path=canonical.presentation_path,
        schedule_path=Path(f"<in-memory:{identifier}>") ,
        source_path=canonical.source_path,
    )
