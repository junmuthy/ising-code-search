"""Load the all-weight-eight BB64 basis and a simultaneous syndrome schedule."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NUM_DATA_QUBITS = 64
NUM_CHECKS_PER_TYPE = 32
NUM_LOGICALS = 8
STATIC_DISTANCE = 8

MODULE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = MODULE_DIR.parent
BASIS_PATH = (
    REFERENCE_DIR
    / "results"
    / "run_003_low_weight"
    / "simultaneous_basis_weight64.npz"
)
DEFAULT_SCHEDULE_PATH = REFERENCE_DIR / "schedule_fault_search" / "baseline_schedule.json"


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
    schedule: tuple[tuple[ScheduledGate, ...], ...]
    basis_path: Path
    schedule_path: Path

    @property
    def num_logicals(self) -> int:
        return len(self.logicals_x)


def gf2_rank(matrix: np.ndarray) -> int:
    pivots: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Return a row basis for the right kernel of a binary matrix."""
    work = np.asarray(matrix, dtype=np.uint8).copy()
    rows, columns = work.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        work[[pivot_row, selected]] = work[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row, column]:
                work[row] ^= work[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = np.zeros(columns, dtype=np.uint8)
        vector[free] = 1
        for row, pivot in reversed(list(enumerate(pivot_columns))):
            vector[pivot] = int(np.dot(work[row], vector) % 2)
        basis.append(vector)
    return np.asarray(basis, dtype=np.uint8)


def independent_relations(check: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Choose four row relations, explicitly including the all-row parity."""
    kernel = gf2_nullspace(check.T)
    all_rows = np.ones(len(check), dtype=np.uint8)
    if np.any((all_rows @ check) % 2):
        raise ValueError("the displayed checks do not have the all-row parity relation")
    selected = [all_rows]
    rank = 1
    for relation in kernel:
        candidate = np.asarray([*selected, relation], dtype=np.uint8)
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            selected.append(relation)
            rank = candidate_rank
    expected = len(check) - gf2_rank(check)
    if rank != expected:
        raise ValueError(f"expected {expected} independent row relations, found {rank}")
    return tuple(
        tuple(np.flatnonzero(relation).astype(int).tolist()) for relation in selected
    )


def _supports(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in np.asarray(matrix, dtype=np.uint8)
    )


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _xor_support(rows: tuple[tuple[int, ...], ...], indices: tuple[int, ...]) -> set[int]:
    support: set[int] = set()
    for index in indices:
        support.symmetric_difference_update(rows[index])
    return support


def load_code_data(
    schedule_path: Path | None = None,
    *,
    basis_path: Path | None = None,
) -> CodeData:
    basis_path = BASIS_PATH if basis_path is None else Path(basis_path)
    schedule_path = DEFAULT_SCHEDULE_PATH if schedule_path is None else Path(schedule_path)
    archive = np.load(basis_path)
    matrix_x = np.asarray(archive["matrix_x"], dtype=np.uint8)
    matrix_z = np.asarray(archive["matrix_z"], dtype=np.uint8)
    logical_x = np.asarray(archive["logical_x"], dtype=np.uint8)
    logical_z = np.asarray(archive["logical_z"], dtype=np.uint8)
    checks_x = _supports(matrix_x)
    checks_z = _supports(matrix_z)
    logicals_x = _supports(logical_x)
    logicals_z = _supports(logical_z)
    relations = independent_relations(matrix_x)
    schedule_record = _load_json(schedule_path)
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

    if matrix_x.shape != (NUM_CHECKS_PER_TYPE, NUM_DATA_QUBITS):
        raise ValueError(f"unexpected X-check shape: {matrix_x.shape}")
    if not np.array_equal(matrix_x, matrix_z):
        raise ValueError("the saved code must have H_X=H_Z")
    if gf2_rank(matrix_x) != 28:
        raise ValueError("the displayed checks must have rank 28")
    if any(len(row) != 8 for row in checks_x + checks_z):
        raise ValueError("the presentation must be uniformly weight eight")
    if len(logicals_x) != NUM_LOGICALS or len(logicals_z) != NUM_LOGICALS:
        raise ValueError("the basis must contain eight canonical logical pairs")
    if any(len(row) != 8 for row in logicals_x + logicals_z):
        raise ValueError("every saved logical representative must have weight eight")
    if len(relations) != 4:
        raise ValueError("expected four independent redundant-check relations")
    for relation in relations:
        if _xor_support(checks_x, relation) or _xor_support(checks_z, relation):
            raise ValueError(f"invalid redundant-check relation: {relation}")
    if len(layers) != 8:
        raise ValueError("the selected schedule must have depth eight")
    expected_edges = {
        (kind, check, data)
        for kind, checks in (("X", checks_x), ("Z", checks_z))
        for check, row in enumerate(checks)
        for data in row
    }
    scheduled_edges = {
        (gate.kind, gate.check, gate.data) for layer in layers for gate in layer
    }
    if expected_edges != scheduled_edges:
        raise ValueError("the schedule does not cover the presentation exactly")
    for layer_index, layer in enumerate(layers):
        data_seen: set[int] = set()
        ancilla_seen: set[tuple[str, int]] = set()
        for gate in layer:
            ancilla = (gate.kind, gate.check)
            if gate.data in data_seen or ancilla in ancilla_seen:
                raise ValueError(f"collision in schedule layer {layer_index}")
            data_seen.add(gate.data)
            ancilla_seen.add(ancilla)
        if len(layer) != 64 or len(data_seen) != 64 or len(ancilla_seen) != 64:
            raise ValueError(f"schedule layer {layer_index} is not a perfect matching")

    times = {
        (gate.kind, gate.check, gate.data): layer_index
        for layer_index, layer in enumerate(layers)
        for gate in layer
    }
    for check_x in range(NUM_CHECKS_PER_TYPE):
        for check_z in range(NUM_CHECKS_PER_TYPE):
            overlap = set(checks_x[check_x]) & set(checks_z[check_z])
            inversions = sum(
                times[("X", check_x, data)] < times[("Z", check_z, data)]
                for data in overlap
            )
            if inversions % 2:
                raise ValueError("cross-ancilla backaction parity failure")

    return CodeData(
        checks_x=checks_x,
        checks_z=checks_z,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        redundant_relations=relations,
        schedule=layers,
        basis_path=basis_path,
        schedule_path=schedule_path,
    )
