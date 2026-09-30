#!/usr/bin/env python3
"""Independently verify the automorphism-dual [[56,8,6]] code with qLDPC.

Rebuild the matrices from the polynomials, without importing the search code
or loading its saved certificates. Compute BOTH CSS distances using qLDPC's
native exact enumeration, with cutoff=0 (no early termination). This can take
several minutes. No randomized distance bounds or promised distances are used.

This verifies stabilizer/code distance, NOT syndrome-circuit fault distance.
Requires numpy and qldpc; tested with qldpc 0.3.3.

Usage:
    python verify_with_qldpc.py
    python verify_with_qldpc.py --output verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import qldpc
from qldpc import codes
from qldpc.objects import Pauli


def require(condition: bool, message: str) -> None:
    """Fail explicitly, including when Python runs with assertions disabled."""
    if not condition:
        raise ValueError(message)


def regular_matrix(support: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return A(c)[g,h] = coefficient of h-g, ordered by g=7*x+y."""
    matrix = np.zeros((28, 28), dtype=np.uint8)
    for x in range(4):
        for y in range(7):
            for dx, dy in support:
                matrix[7 * x + y, 7 * ((x + dx) % 4) + (y + dy) % 7] ^= 1
    return matrix


def build_code() -> codes.CSSCode:
    """Build the discovered code, not the different BB56 from the STAR paper."""
    # a = 1 + y + x + x*y^2; b = 1 + y^-1 + x + x*y^-2.
    matrix_a = regular_matrix(((0, 0), (0, 1), (1, 0), (1, 2)))
    matrix_b = regular_matrix(((0, 0), (0, -1), (1, 0), (1, -2)))
    hx = np.hstack((matrix_a, matrix_b))
    hz = np.hstack((matrix_b.T, matrix_a.T))
    # Do not promise stabilizer status, equal X/Z distances, or known parameters.
    return codes.CSSCode(hx, hz, field=2)


def verify_structure(code: codes.CSSCode) -> dict:
    """Check stabilizers, ranks, and a full canonical logical basis over GF(2)."""
    hx, hz = code.matrix_x, code.matrix_z
    require(hx.shape == hz.shape == (28, 56), "Unexpected check matrix shapes")
    require(not np.any(hx @ hz.T), "X and Z checks do not commute")
    require(not code.is_subsystem_code, "This must be a stabilizer code, not a subsystem code")
    rank_x, rank_z = int(code.code_x.rank), int(code.code_z.rank)
    require((rank_x, rank_z) == (24, 24), "Expected binary check ranks 24 and 24")
    require(len(code) == 56 and code.dimension == 8, "Expected n=56 and k=8")
    weights_x = sorted(set(np.asarray(hx, dtype=int).sum(axis=1).tolist()))
    weights_z = sorted(set(np.asarray(hz, dtype=int).sum(axis=1).tolist()))
    require(weights_x == weights_z == [8], "Expected all displayed checks to have weight 8")

    logical_x = code.get_logical_ops(Pauli.X)
    logical_z = code.get_logical_ops(Pauli.Z)
    require(logical_x.shape == logical_z.shape == (8, 56), "Expected eight logical pairs")
    require(not np.any(hz @ logical_x.T), "Logical X operators have nonzero syndrome")
    require(not np.any(hx @ logical_z.T), "Logical Z operators have nonzero syndrome")
    require(
        np.array_equal(logical_x @ logical_z.T, np.eye(8, dtype=int)),
        "Logical X/Z operators do not have canonical pairing",
    )

    # Independently check the saved weight-six witnesses, but do NOT use them
    # to set qLDPC's distance cache or as an early-exit cutoff.
    witnesses = {"X": [2, 9, 10, 29, 35, 36], "Z": [0, 1, 7, 33, 34, 41]}
    for name, check, conjugate in (("X", hz, logical_z), ("Z", hx, logical_x)):
        word = code.field.Zeros(56)
        word[witnesses[name]] = 1
        require(np.count_nonzero(word) == 6, f"{name} witness does not have weight six")
        require(not np.any(check @ word), f"{name} witness has nonzero syndrome")
        require(np.any(conjugate @ word), f"{name} witness is a stabilizer, not a logical")

    return {
        "n": len(code),
        "k": int(code.dimension),
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weights_x": weights_x,
        "check_weights_z": weights_z,
        "css_commutes": True,
        "is_stabilizer_code": True,
        "canonical_logical_pairs_verified": 8,
        "weight_six_witnesses": witnesses,
    }


def verify() -> dict:
    """Compute exact d_X and d_Z independently, with no equality shortcut."""
    started = time.perf_counter()
    code = build_code()
    report = verify_structure(code)
    print(
        "Structure verified: n=56, k=8, ranks=24/24, check weights=8.",
        file=sys.stderr,
        flush=True,
    )
    distances = {}
    seconds = {}
    for name, pauli in (("X", Pauli.X), ("Z", Pauli.Z)):
        print(
            f"Computing exact d_{name} with qLDPC (full enumeration)...",
            file=sys.stderr,
            flush=True,
        )
        before = time.perf_counter()
        # cutoff=6 would only prove d<=6. cutoff=0 forces complete enumeration.
        result = code.get_distance_exact(pauli, cutoff=0)
        require(result == 6, f"Expected d_{name}=6, but qLDPC computed {result}")
        distances[name] = int(result)
        seconds[name] = round(time.perf_counter() - before, 6)
        print(
            f"Exact d_{name}={result} ({seconds[name]:.2f} seconds).",
            file=sys.stderr,
            flush=True,
        )

    report.update(
        family="C4xC7-automorphism-dual-BB",
        qubit_index="q(h,x,y)=28*h+7*x+y",
        polynomials={"a": "1+y+x+x*y^2", "b": "1+y^-1+x+x*y^-2"},
        qldpc_version=version("qldpc"),
        qldpc_source=str(Path(qldpc.__file__).resolve()),
        d_x=distances["X"],
        d_z=distances["Z"],
        d=min(distances.values()),
        distance_method="qldpc.codes.CSSCode.get_distance_exact, separately for X and Z, cutoff=0",
        distance_exact=True,
        distance_seconds=seconds,
        total_seconds=round(time.perf_counter() - started, 6),
        fault_distance=None,
        scope="Stabilizer code distance only; no circuit fault-distance claim or Stim simulation.",
        verified=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, help="Also save the JSON report to a new file; never overwrite"
    )
    args = parser.parse_args()
    if args.output is not None:
        if args.output.exists():
            parser.error(f"refusing to overwrite {args.output}")
        if not args.output.parent.is_dir():
            parser.error(f"output parent directory does not exist: {args.output.parent}")
    payload = json.dumps(verify(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
