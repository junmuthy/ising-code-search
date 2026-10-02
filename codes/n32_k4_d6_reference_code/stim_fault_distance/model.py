"""Load and validate the saved code presentation and syndrome schedule."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUM_DATA_QUBITS = 32
NUM_CHECKS_PER_TYPE = 16
STATIC_DISTANCE = 6

MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR.parent
PRESENTATION_PATH = (
    REFERENCE_DIR
    / "schedule"
    / "all_weight8_translation_symmetric_v1"
    / "presentation.json"
)
SCHEDULE_PATH = (
    REFERENCE_DIR
    / "followups"
    / "fold_rescan_and_second_translation_v1"
    / "optimal-clean-joint-12-layer.json"
)
SOURCE_PATH = (
    REFERENCE_DIR
    / "results"
    / "folded-local-20x250-260828-v1"
    / "start-results"
    / "start-010.json"
)


@dataclass(frozen=True)
class ScheduledGate:
    """One scheduled check-data CNOT."""

    kind: str
    check: int
    data: int


@dataclass(frozen=True)
class CodeData:
    """All immutable inputs needed to build the Stim circuits."""

    checks_x: tuple[tuple[int, ...], ...]
    checks_z: tuple[tuple[int, ...], ...]
    logicals_x: tuple[tuple[int, ...], ...]
    logicals_z: tuple[tuple[int, ...], ...]
    redundant_relations: tuple[tuple[int, ...], ...]
    fold_permutation: tuple[int, ...]
    schedule: tuple[tuple[ScheduledGate, ...], ...]
    presentation_path: Path
    schedule_path: Path
    source_path: Path

    @property
    def num_logicals(self) -> int:
        return len(self.logicals_x)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tuple_rows(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in rows)


def _xor_support(rows: tuple[tuple[int, ...], ...], indices: tuple[int, ...]) -> set[int]:
    support: set[int] = set()
    for index in indices:
        support.symmetric_difference_update(rows[index])
    return support


def load_code_data() -> CodeData:
    """Load the canonical all-weight-eight presentation and depth-12 schedule."""

    presentation = _load_json(PRESENTATION_PATH)
    schedule_record = _load_json(SCHEDULE_PATH)
    source = _load_json(SOURCE_PATH)["result"]["best"]

    checks_x = _tuple_rows(presentation["checks_x"])
    checks_z = _tuple_rows(presentation["checks_z"])
    logicals_x = _tuple_rows(source["logical_supports_x"])
    logicals_z = _tuple_rows(source["logical_supports_z"])
    relations = _tuple_rows(presentation["redundant_relations"])
    fold = tuple(int(value) for value in presentation["fold_permutation"])
    layers = tuple(
        tuple(
            ScheduledGate(
                kind=str(gate["type"]),
                check=int(gate["check"]),
                data=int(gate["data"]),
            )
            for gate in layer["gates"]
        )
        for layer in schedule_record["layers"]
    )

    if presentation["code"] != "[[32,4,6]]":
        raise ValueError(f"unexpected code: {presentation['code']}")
    if len(checks_x) != NUM_CHECKS_PER_TYPE or len(checks_z) != NUM_CHECKS_PER_TYPE:
        raise ValueError("the saved presentation must contain 16 checks of each type")
    if any(len(row) != 8 for row in checks_x + checks_z):
        raise ValueError("the saved presentation is not uniformly weight eight")
    if len(logicals_x) != 4 or len(logicals_z) != 4:
        raise ValueError("the saved source must contain four logicals of each type")
    if any(len(row) != 7 for row in logicals_x + logicals_z):
        raise ValueError("the saved logical representatives are not weight seven")
    if len(fold) != NUM_DATA_QUBITS or sorted(fold) != list(range(NUM_DATA_QUBITS)):
        raise ValueError("invalid fold permutation")
    if len(layers) != 12:
        raise ValueError("the saved simultaneous schedule must have depth 12")
    if sum(len(layer) for layer in layers) != 256:
        raise ValueError("the saved schedule must contain 256 CNOTs")
    for relation in relations:
        if _xor_support(checks_x, relation) or _xor_support(checks_z, relation):
            raise ValueError(f"invalid redundant-check relation: {relation}")

    expected_edges = {
        (kind, check, data)
        for kind, checks in (("X", checks_x), ("Z", checks_z))
        for check, row in enumerate(checks)
        for data in row
    }
    scheduled_edges = {
        (gate.kind, gate.check, gate.data)
        for layer in layers
        for gate in layer
    }
    if expected_edges != scheduled_edges:
        raise ValueError("the schedule does not cover the presentation exactly")
    for layer_index, layer in enumerate(layers):
        used_data: set[int] = set()
        used_ancillas: set[tuple[str, int]] = set()
        for gate in layer:
            ancilla = (gate.kind, gate.check)
            if gate.data in used_data or ancilla in used_ancillas:
                raise ValueError(f"collision in schedule layer {layer_index}")
            used_data.add(gate.data)
            used_ancillas.add(ancilla)

    return CodeData(
        checks_x=checks_x,
        checks_z=checks_z,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        redundant_relations=relations,
        fold_permutation=fold,
        schedule=layers,
        presentation_path=PRESENTATION_PATH,
        schedule_path=SCHEDULE_PATH,
        source_path=SOURCE_PATH,
    )
