"""Load and validate the frozen paper ``[[56,8,6]]`` artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bicycle_chain_l2_paper.tmr_postselection.model import (
    CodeArtifact,
    ScheduledGate,
    _validate_schedule,
    gf2_rank,
)


NUM_DATA = 56
NUM_CHECKS = 28
NUM_LOGICALS = 8
BATCHES = (tuple(range(NUM_LOGICALS)),)
PACKAGE_DIR = Path(__file__).resolve().parent
CODE_ROOT = PACKAGE_DIR.parent
DEFAULT_BASIS = CODE_ROOT / "code_data" / "presentation.npz"
DEFAULT_SCHEDULE = CODE_ROOT / "schedule" / "paper_table_ii_schedule.json"
EXPECTED_BASIS_SHA256 = "9f870ff5dae31ca0f5f077def02f1a19cc6b12693ca6f9f2e7c53969f334f9de"
EXPECTED_SCHEDULE_SHA256 = "097a4f43a93b16f6ff1198043c967399621beb35ff7eebc76a572ba33a2ae5e5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with schedule_path.open(encoding="utf-8") as source:
        record = json.load(source)
    schedule = tuple(
        tuple(
            ScheduledGate(str(gate["type"]), int(gate["check"]), int(gate["data"]))
            for gate in layer["gates"]
        )
        for layer in record["layers"]
    )
    if matrix_x.shape != (NUM_CHECKS, NUM_DATA) or matrix_z.shape != matrix_x.shape:
        raise ValueError("unexpected check shape")
    if (gf2_rank(matrix_x), gf2_rank(matrix_z)) != (24, 24):
        raise ValueError("expected X- and Z-check rank 24")
    if logical_x.shape != (NUM_LOGICALS, NUM_DATA) or logical_z.shape != logical_x.shape:
        raise ValueError("unexpected logical shape")
    if np.any((matrix_x @ logical_z.T) % 2) or np.any((matrix_z @ logical_x.T) % 2):
        raise ValueError("logical representative fails to commute")
    if not np.array_equal(
        (logical_z @ logical_x.T) % 2, np.eye(NUM_LOGICALS, dtype=np.uint8)
    ):
        raise ValueError("logical basis is not canonical")
    if set(np.count_nonzero(matrix_x, axis=1).tolist()) != {8}:
        raise ValueError("checks are not uniformly weight eight")
    if set(np.count_nonzero(logical_z, axis=1).tolist()) != {7}:
        raise ValueError("logicals are not uniformly weight seven")
    union: set[int] = set()
    for logical in BATCHES[0]:
        support = set(np.flatnonzero(logical_z[logical]).astype(int).tolist())
        if union.intersection(support):
            raise ValueError("logical Z supports overlap")
        union.update(support)
    if len(union) != NUM_DATA:
        raise ValueError("the eight logical supports must partition the data")
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
