"""Exhaust all normalized four-term paired BB polynomials over C2 x C7."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import tempfile
import time
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

from qldpc import abstract, codes

from gala_search.single_row import gf2_rank, gf2_rref

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "results" / "c2xc7-bb"


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def packed(vector: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(vector))


def exact_distance(check: np.ndarray, logicals: np.ndarray) -> dict[str, Any]:
    basis, _pivots = gf2_rref(np.asarray(check, dtype=np.uint8))
    stabilizers = [0]
    for row in basis:
        mask = packed(row)
        stabilizers.extend(value ^ mask for value in tuple(stabilizers))
    logical_masks = [packed(row) for row in logicals]
    best_weight = check.shape[1] + 1
    best_bits = 0
    best_operator = 0
    counts: Counter[int] = Counter()
    for logical_bits in range(1, 1 << len(logicals)):
        logical = 0
        for bit, mask in enumerate(logical_masks):
            if logical_bits >> bit & 1:
                logical ^= mask
        for stabilizer in stabilizers:
            operator = logical ^ stabilizer
            weight = operator.bit_count()
            if weight <= 7:
                counts[weight] += 1
            if weight < best_weight:
                best_weight = weight
                best_bits = logical_bits
                best_operator = operator
    return {
        "distance": best_weight,
        "logical_combination": best_bits,
        "operator_support": [
            qubit for qubit in range(check.shape[1]) if best_operator >> qubit & 1
        ],
        "weight_counts_through_seven": {
            str(weight): counts[weight] for weight in range(1, 8)
        },
        "stabilizers_per_coset": len(stabilizers),
        "cosets_checked": (1 << len(logicals)) - 1,
    }


def candidate_logicals(
    ring: abstract.GroupRing,
    xx: abstract.RingMember,
    yy: abstract.RingMember,
) -> np.ndarray:
    omega = sum((yy**power for power in range(7)), ring.zero)
    rows = []
    for half in range(2):
        for logical in range(2):
            vector = np.asarray((xx**logical * omega).to_vector(), dtype=np.uint8)
            zeros = np.zeros(14, dtype=np.uint8)
            rows.append(
                np.hstack([vector, zeros])
                if half == 0
                else np.hstack([zeros, vector])
            )
    return np.asarray(rows, dtype=np.uint8)


def analyze(
    r: int,
    y_one: int,
    y_two: int,
    ring: abstract.GroupRing,
    xx: abstract.RingMember,
    yy: abstract.RingMember,
) -> dict[str, Any]:
    polynomial = ring.one + yy**r + xx * yy**y_one + xx * yy**y_two
    code = codes.GALACode(
        generators_f=[polynomial],
        generators_g=[polynomial.T],
        num_active_rows=1,
    )
    check = np.asarray(code.matrix_x, dtype=np.uint8)
    logicals = candidate_logicals(ring, xx, yy)
    field = code.field
    rank = gf2_rank(check)
    rank_gain = gf2_rank(np.vstack([check, logicals])) - rank
    pairing = np.asarray(field(logicals) @ field(logicals).T, dtype=int)
    checks = {
        "n_is_28": code.num_qubits == 28,
        "k_is_4": code.dimension == 4,
        "rank_is_12": rank == 12,
        "hx_equals_hz": bool(np.array_equal(code.matrix_x, code.matrix_z)),
        "css_orthogonal": bool(not np.any(code.matrix_x @ code.matrix_z.T)),
        "check_weight_is_8": code.get_weight() == 8,
        "logicals_commute": bool(not np.any(code.matrix_x @ field(logicals).T)),
        "logicals_complete": rank_gain == 4,
        "logical_pairing_identity": bool(np.array_equal(pairing, np.eye(4, dtype=int))),
        "logicals_disjoint": bool(np.all(np.count_nonzero(logicals, axis=0) <= 1)),
        "tanner_connected": nx.is_connected(code.code_x.graph.to_undirected()),
    }
    distance = exact_distance(check, logicals) if all(checks.values()) else None
    return {
        "support": [[0, 0], [0, r], [1, y_one], [1, y_two]],
        "polynomial": f"1 + y^{r} + x y^{y_one} + x y^{y_two}",
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_h": rank,
        "check_weight": code.get_weight(),
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "distance": distance,
        "certified_distance": distance["distance"] if distance else None,
        "logical_supports": [np.flatnonzero(row).astype(int).tolist() for row in logicals],
        "zx_pairing": pairing.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="paired-weight8-exhaustive-v1")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = RESULTS_ROOT / args.run_name
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 7))
    xx, yy = ring.generators
    supports = [
        (r, y_one, y_two)
        for r in range(1, 7)
        for y_one, y_two in itertools.combinations(range(7), 2)
    ]
    records = []
    counters: Counter[str] = Counter()
    started = time.perf_counter()
    for index, (r, y_one, y_two) in enumerate(supports, start=1):
        record = analyze(r, y_one, y_two, ring, xx, yy)
        records.append(record)
        counters["candidates"] += 1
        if record["accepted_structurally"]:
            counters["structurally_accepted"] += 1
            counters[f"distance_{record['certified_distance']}"] += 1
        if index % args.checkpoint_every == 0 or index == len(supports):
            progress = {
                "status": "searching" if index < len(supports) else "complete",
                "completed": index,
                "total": len(supports),
                "counters": dict(sorted(counters.items())),
                "seconds": round(time.perf_counter() - started, 6),
            }
            atomic_json(output / "progress.json", progress)
            write_jsonl(output / "candidates.jsonl", records)
            print(json.dumps(progress, sort_keys=True), flush=True)
    survivors = [
        record for record in records if (record["certified_distance"] or 0) >= 6
    ]
    write_jsonl(output / "survivors.jsonl", survivors)
    best_distance = max([record["certified_distance"] or 0 for record in records] + [0])
    best = [record for record in records if record["certified_distance"] == best_distance]
    write_jsonl(output / "best-candidates.jsonl", best)
    summary = {
        "target": "[[28,4,>=6]] BB code with two independent C2 logical cycles",
        "family": "all normalized paired four-term polynomials over C2 x C7",
        "candidates": len(supports),
        "counters": dict(sorted(counters.items())),
        "best_distance": best_distance,
        "survivors": len(survivors),
        "seconds": round(time.perf_counter() - started, 6),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
