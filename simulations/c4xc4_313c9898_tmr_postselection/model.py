"""Read-only, hash-pinned adapter for the saved [[64,16,6]] bundle."""

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from codes.bicycle_chain_l2_paper.tmr_postselection.model import (
    CodeArtifact, ScheduledGate, gf2_rank, sha256,
)

CODE_ID = "313c98984982b039f1068bfd6001a4ffceb370b2091c3edbc9d3f8b915aaa028"
CODE_ROOT = Path(os.environ.get(
    "GALA313_CODE_ROOT",
    Path(__file__).resolve().parents[2] / "codes/c4xc4_64_16_6_313c9898",
))
ACTIVE = (0, 2, 4, 6)
FROZEN = {
    "preferred": (
        "f92ea7ebe5578ed2d182f3282cab980bf847a6f594b068885da0f08698a53d94",
        "f4d45dff84d2655ca4287628db8539b3cb2702770cd47b3ddd0956e8e7dd0934",
    ),
    "original": (
        "08346f20b59e73a1c6d97bfe5607414fe510148bd7d9ec84865d7a9b85af7fb2",
        "f14daf180a306669119fe107e9d032866737facd52cbd8843623f074abdfd6a1",
    ),
}


def bit_matrix(rows):
    return np.array([[(int(row) >> q) & 1 for q in range(64)] for row in rows], dtype=np.uint8)


def validate_schedule(hx, hz, schedule):
    expected = Counter((kind, r, int(q)) for kind, matrix in (("X", hx), ("Z", hz))
                       for r, row in enumerate(matrix) for q in np.flatnonzero(row))
    actual = Counter((g.kind, g.check, g.data) for layer in schedule for g in layer)
    if actual != expected or len(schedule) != 12:
        raise ValueError("schedule must contain every Tanner edge exactly once at depth 12")
    for layer in schedule:
        if len({g.data for g in layer}) != len(layer):
            raise ValueError("schedule data collision")
        if len({(g.kind, g.check) for g in layer}) != len(layer):
            raise ValueError("schedule ancilla collision")
    # Commuting checks are not sufficient: audit the cross-ancilla propagation
    # induced by interleaving X/Z CNOTs. Every ordered overlap must be even.
    times = {(g.kind, g.check, g.data): t for t, layer in enumerate(schedule) for g in layer}
    for x, xr in enumerate(hx):
        for z, zr in enumerate(hz):
            before = sum(times["X", x, int(q)] < times["Z", z, int(q)]
                         for q in np.flatnonzero(xr & zr))
            if before % 2:
                raise ValueError("schedule has unwanted X-to-Z ancilla propagation")


def load_code_artifact(presentation="preferred", root=CODE_ROOT):
    if presentation not in FROZEN:
        raise ValueError("presentation must be preferred or original")
    basis_path = Path(root) / presentation / "code.json"
    schedule_path = Path(root) / presentation / "schedule.json"
    hashes = (sha256(basis_path), sha256(schedule_path))
    if hashes != FROZEN[presentation]:
        raise ValueError("saved code or schedule hash differs from the frozen input")
    record = json.loads(basis_path.read_text())
    sched = json.loads(schedule_path.read_text())
    if record["basis_id"] != CODE_ID or sched["basis_id"] != CODE_ID:
        raise ValueError("wrong code ID")
    hx, hz = (bit_matrix(record["checks"][axis]) for axis in ("X", "Z"))
    lx, lz = (bit_matrix(record["logicals"][axis]) for axis in ("X", "Z"))
    checks = 24 if presentation == "preferred" else 32
    if hx.shape != (checks, 64) or hz.shape != hx.shape:
        raise ValueError("wrong check shapes")
    if (gf2_rank(hx), gf2_rank(hz)) != (24, 24) or np.any((hx @ hz.T) % 2):
        raise ValueError("invalid CSS ranks/commutation")
    if lx.shape != (16, 64) or lz.shape != lx.shape:
        raise ValueError("all sixteen logicals must be retained in the encoder")
    if np.any((hx @ lz.T) % 2) or np.any((hz @ lx.T) % 2):
        raise ValueError("logical-check commutation failed")
    if not np.array_equal((lx @ lz.T) % 2, np.eye(16, dtype=np.uint8)):
        raise ValueError("noncanonical logical basis")
    if any(set(m.sum(axis=1).tolist()) != {w} for m, w in ((hx, 12), (hz, 12), (lx, 6), (lz, 6))):
        raise ValueError("wrong check/logical weights")
    if np.max(lz[list(ACTIVE)].sum(axis=0)) != 1:
        raise ValueError("selected logical supports must be disjoint")
    schedule = tuple(tuple(ScheduledGate(str(g["type"]), int(g["check"]), int(g["data"]))
                           for g in layer) for layer in sched["layers"])
    validate_schedule(hx, hz, schedule)
    return CodeArtifact(hx, hz, lx, lz, (ACTIVE,), schedule, basis_path, schedule_path, *hashes)
