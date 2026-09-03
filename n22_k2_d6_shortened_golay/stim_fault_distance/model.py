"""Load the shortened-Golay checks, logicals, and one syndrome schedule."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from n22_k2_d6_shortened_golay.code_data.code import (
    LOGICAL_X_SUPPORTS,
    LOGICAL_Z_SUPPORTS,
    NUM_CHECKS_PER_TYPE,
    NUM_DATA_QUBITS,
    STATIC_DISTANCE,
    build_checks,
    fold_permutation,
    gf2_rank,
)


MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR.parent
DEFAULT_SCHEDULE_PATH = REFERENCE_DIR / "schedule" / "baseline_schedule.json"


@dataclass(frozen=True)
class ScheduledGate:
    kind: str
    check: int
    data: int


@dataclass(frozen=True)
class CodeData:
    checks_x: tuple[tuple[int, ...], ...]
    checks_z: tuple[tuple[int, ...], ...]
    logicals_x: tuple[tuple[int, ...], ...]
    logicals_z: tuple[tuple[int, ...], ...]
    redundant_relations: tuple[tuple[int, ...], ...]
    fold_permutation: tuple[int, ...]
    schedule: tuple[tuple[ScheduledGate, ...], ...]
    schedule_path: Path

    @property
    def num_logicals(self) -> int:
        return len(self.logicals_x)


def _supports(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(np.flatnonzero(row).astype(int).tolist()) for row in matrix)


def load_code_data(schedule_path: Path | None = None) -> CodeData:
    path = DEFAULT_SCHEDULE_PATH if schedule_path is None else Path(schedule_path)
    schedule_record = json.loads(path.read_text(encoding="utf-8"))
    checks_x_array, checks_z_array = build_checks()
    checks_x = _supports(checks_x_array)
    checks_z = _supports(checks_z_array)
    layers = tuple(
        tuple(
            ScheduledGate(str(gate["type"]), int(gate["check"]), int(gate["data"]))
            for gate in layer["gates"]
        )
        for layer in schedule_record["layers"]
    )
    relation = tuple(range(NUM_CHECKS_PER_TYPE))
    if gf2_rank(checks_x_array) != 10 or gf2_rank(checks_z_array) != 10:
        raise ValueError("unexpected check rank")
    if np.any(np.bitwise_xor.reduce(checks_x_array, axis=0)):
        raise ValueError("X checks lack the all-row relation")
    if np.any(np.bitwise_xor.reduce(checks_z_array, axis=0)):
        raise ValueError("Z checks lack the all-row relation")
    if len(layers) != 8 or sum(map(len, layers)) != 176:
        raise ValueError("schedule is not a complete depth-eight extraction")
    expected = {
        (kind, check, data)
        for kind, checks in (("X", checks_x), ("Z", checks_z))
        for check, support in enumerate(checks)
        for data in support
    }
    observed = {(gate.kind, gate.check, gate.data) for layer in layers for gate in layer}
    if expected != observed:
        raise ValueError("schedule does not cover the Tanner graph exactly")
    for index, layer in enumerate(layers):
        if len(layer) != NUM_DATA_QUBITS:
            raise ValueError(f"layer {index} is not a perfect matching")
        if len({gate.data for gate in layer}) != NUM_DATA_QUBITS:
            raise ValueError(f"data collision in layer {index}")
        if len({(gate.kind, gate.check) for gate in layer}) != NUM_DATA_QUBITS:
            raise ValueError(f"ancilla collision in layer {index}")
    return CodeData(
        checks_x=checks_x,
        checks_z=checks_z,
        logicals_x=tuple(tuple(row) for row in LOGICAL_X_SUPPORTS),
        logicals_z=tuple(tuple(row) for row in LOGICAL_Z_SUPPORTS),
        redundant_relations=(relation,),
        fold_permutation=fold_permutation(),
        schedule=layers,
        schedule_path=path,
    )


__all__ = [
    "CodeData",
    "DEFAULT_SCHEDULE_PATH",
    "NUM_CHECKS_PER_TYPE",
    "NUM_DATA_QUBITS",
    "STATIC_DISTANCE",
    "ScheduledGate",
    "load_code_data",
]

