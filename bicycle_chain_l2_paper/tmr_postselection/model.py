"""Load and validate the frozen paper ``[[28,4,5]]`` code artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


NUM_DATA = 28
NUM_CHECKS = 14
NUM_LOGICALS = 4
BATCHES = ((0, 1, 2, 3),)

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_ROOT = Path(os.environ.get("BICYCLE_CHAIN_L2_CODE_ROOT", PACKAGE_DIR.parent))
DEFAULT_BASIS = CODE_ROOT / "code_data" / "presentation.npz"
DEFAULT_SCHEDULE = CODE_ROOT / "schedule" / "paper_table_ii_schedule.json"
EXPECTED_BASIS_SHA256 = "8c5efff89b55d877501e7f60629b797e745197f8626c5041d0aad09bbb4c26b4"
EXPECTED_SCHEDULE_SHA256 = "bf8a53fcf43234cb4a8a7aef2000085fd21c0b50cb9219209fc20326f7f253cd"


@dataclass(frozen=True)
class ScheduledGate:
    kind: str
    check: int
    data: int


@dataclass(frozen=True)
class CodeArtifact:
    matrix_x: np.ndarray
    matrix_z: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    batches: tuple[tuple[int, ...], ...]
    schedule: tuple[tuple[ScheduledGate, ...], ...]
    basis_path: Path
    schedule_path: Path
    basis_sha256: str
    schedule_sha256: str

    @property
    def num_data(self) -> int:
        return int(self.matrix_x.shape[1])

    @property
    def num_checks(self) -> int:
        return int(self.matrix_x.shape[0])

    @property
    def num_logicals(self) -> int:
        return int(self.logical_z.shape[0])

    @property
    def checks_x(self) -> tuple[tuple[int, ...], ...]:
        return supports(self.matrix_x)

    @property
    def checks_z(self) -> tuple[tuple[int, ...], ...]:
        return supports(self.matrix_z)

    @property
    def logicals_x(self) -> tuple[tuple[int, ...], ...]:
        return supports(self.logical_x)

    @property
    def logicals_z(self) -> tuple[tuple[int, ...], ...]:
        return supports(self.logical_z)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def supports(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in np.asarray(matrix, dtype=np.uint8)
    )


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, columns = work.shape
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
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def independent_indices(rows: np.ndarray) -> tuple[int, ...]:
    """Return a deterministic maximal independent subset of matrix rows."""

    selected: list[int] = []
    rank = 0
    matrix = np.asarray(rows, dtype=np.uint8)
    for index in range(len(matrix)):
        candidate = matrix[[*selected, index]]
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            selected.append(index)
            rank = candidate_rank
    return tuple(selected)


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("GF(2) inverse requires a square matrix")
    size = matrix.shape[0]
    work = np.concatenate((matrix.copy(), np.eye(size, dtype=np.uint8)), axis=1)
    for column in range(size):
        candidates = np.flatnonzero(work[column:, column])
        if not len(candidates):
            raise ValueError("matrix is singular over GF(2)")
        selected = column + int(candidates[0])
        work[[column, selected]] = work[[selected, column]]
        for row in range(size):
            if row != column and work[row, column]:
                work[row] ^= work[column]
    return work[:, size:]


def initialization_correction_map(
    check_z: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Return independent rows, correction qubits, and a syndrome right inverse.

    If ``s`` contains the reported outcomes of the selected independent Z
    checks, the virtual X correction has bits ``B_inv @ s`` on
    ``correction_qubits``.  Thus the selected check syndrome is reset to zero.
    """

    check_z = np.asarray(check_z, dtype=np.uint8)
    row_indices = independent_indices(check_z)
    independent = check_z[list(row_indices)]
    correction_qubits = independent_indices(independent.T)
    square = independent[:, list(correction_qubits)]
    inverse = gf2_inverse(square)
    right_inverse = np.zeros((check_z.shape[1], len(row_indices)), dtype=np.uint8)
    right_inverse[list(correction_qubits)] = inverse
    if not np.array_equal((independent @ right_inverse) % 2, np.eye(len(row_indices), dtype=np.uint8)):
        raise ValueError("failed to construct a Z-syndrome right inverse")
    return row_indices, correction_qubits, right_inverse


def row_relations(check: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Return a basis of relations among the displayed check rows."""

    matrix = np.asarray(check, dtype=np.uint8).T
    work = matrix.copy()
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
    free = [column for column in range(columns) if column not in pivot_columns]
    basis: list[tuple[int, ...]] = []
    for free_column in free:
        vector = np.zeros(columns, dtype=np.uint8)
        vector[free_column] = 1
        for row, pivot in reversed(list(enumerate(pivot_columns))):
            vector[pivot] = int(np.dot(work[row], vector) % 2)
        basis.append(tuple(np.flatnonzero(vector).astype(int).tolist()))
    return tuple(basis)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _validate_schedule(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    schedule: tuple[tuple[ScheduledGate, ...], ...],
) -> None:
    expected = {
        (kind, check, int(data))
        for kind, matrix in (("X", matrix_x), ("Z", matrix_z))
        for check, row in enumerate(matrix)
        for data in np.flatnonzero(row)
    }
    actual = {
        (gate.kind, gate.check, gate.data) for layer in schedule for gate in layer
    }
    if expected != actual:
        raise ValueError("schedule does not contain exactly the Tanner edges")
    if len(schedule) != 8:
        raise ValueError("expected an eight-layer syndrome schedule")
    num_data = int(matrix_x.shape[1])
    for layer_index, layer in enumerate(schedule):
        data = [gate.data for gate in layer]
        ancillas = [(gate.kind, gate.check) for gate in layer]
        if (
            len(layer) != num_data
            or len(set(data)) != num_data
            or len(set(ancillas)) != num_data
        ):
            raise ValueError(f"schedule layer {layer_index} is not a perfect matching")


def load_code_artifact(
    basis_path: Path = DEFAULT_BASIS,
    schedule_path: Path = DEFAULT_SCHEDULE,
    *,
    require_frozen_hashes: bool = True,
) -> CodeArtifact:
    basis_path = Path(basis_path)
    schedule_path = Path(schedule_path)
    basis_hash = sha256(basis_path)
    schedule_hash = sha256(schedule_path)
    if require_frozen_hashes and basis_hash != EXPECTED_BASIS_SHA256:
        raise ValueError(f"unexpected basis hash: {basis_hash}")
    if require_frozen_hashes and schedule_hash != EXPECTED_SCHEDULE_SHA256:
        raise ValueError(f"unexpected schedule hash: {schedule_hash}")

    archive = np.load(basis_path)
    matrix_x = np.asarray(archive["matrix_x"], dtype=np.uint8)
    matrix_z = np.asarray(archive["matrix_z"], dtype=np.uint8)
    logical_x = np.asarray(archive["logical_x"], dtype=np.uint8)
    logical_z = np.asarray(archive["logical_z"], dtype=np.uint8)
    record = _load_json(schedule_path)
    schedule = tuple(
        tuple(
            ScheduledGate(str(gate["type"]), int(gate["check"]), int(gate["data"]))
            for gate in layer["gates"]
        )
        for layer in record["layers"]
    )

    if matrix_x.shape != (NUM_CHECKS, NUM_DATA):
        raise ValueError(f"unexpected check shape: {matrix_x.shape}")
    if gf2_rank(matrix_x) != 12 or gf2_rank(matrix_z) != 12:
        raise ValueError("expected X- and Z-check rank 12")
    if logical_x.shape != (NUM_LOGICALS, NUM_DATA) or logical_z.shape != logical_x.shape:
        raise ValueError("unexpected logical matrix shape")
    if np.any((matrix_x @ logical_z.T) % 2):
        raise ValueError("logical Z representatives do not commute with X checks")
    if np.any((matrix_z @ logical_x.T) % 2):
        raise ValueError("logical X representatives do not commute with Z checks")
    if not np.array_equal((logical_z @ logical_x.T) % 2, np.eye(NUM_LOGICALS, dtype=np.uint8)):
        raise ValueError("logical X/Z representatives are not canonical")
    if set(np.count_nonzero(matrix_x, axis=1).tolist()) != {8}:
        raise ValueError("checks are not uniformly weight eight")
    if set(np.count_nonzero(logical_z, axis=1).tolist()) != {7}:
        raise ValueError("logical Z representatives are not uniformly weight seven")
    union: set[int] = set()
    for logical in BATCHES[0]:
        support = set(np.flatnonzero(logical_z[logical]).astype(int).tolist())
        if union & support:
            raise ValueError("the logical Z representatives are not disjoint")
        union |= support
    if len(union) != NUM_DATA:
        raise ValueError("the four logical Z representatives must partition the data qubits")
    _validate_schedule(matrix_x, matrix_z, schedule)

    return CodeArtifact(
        matrix_x=matrix_x,
        matrix_z=matrix_z,
        logical_x=logical_x,
        logical_z=logical_z,
        batches=BATCHES,
        schedule=schedule,
        basis_path=basis_path,
        schedule_path=schedule_path,
        basis_sha256=basis_hash,
        schedule_sha256=schedule_hash,
    )


def vector_from_support(support: Iterable[int], length: int = NUM_DATA) -> np.ndarray:
    vector = np.zeros(length, dtype=np.uint8)
    vector[list(support)] = 1
    return vector


def support_to_pauli(axis: str, support: Sequence[int]) -> str:
    return "*".join(f"{axis}{qubit}" for qubit in support)
