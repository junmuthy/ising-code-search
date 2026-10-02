#!/usr/bin/env python3
"""Refine the {2,3,5} disjoint batch in the BB64 simultaneous basis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import z3


TARGET_BATCH = (2, 3, 5)
WEIGHT_PATTERNS = (
    (8, 8, 8),
    (12, 8, 8),
    (8, 12, 8),
    (8, 8, 12),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--success-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds-per-query", type=float, default=60)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def emit(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def independent_rows(matrix: np.ndarray) -> np.ndarray:
    selected: list[np.ndarray] = []
    rank = 0
    for row in np.asarray(matrix, dtype=np.uint8):
        proposed = np.asarray([*selected, row], dtype=np.uint8)
        proposed_rank = gf2_rank(proposed)
        if proposed_rank > rank:
            selected.append(row)
            rank = proposed_rank
    return np.asarray(selected, dtype=np.uint8)


def bases_from_json(path: Path) -> np.ndarray:
    record = json.loads(path.read_text())
    bases = np.zeros((8, 64), dtype=np.uint8)
    for logical, support in enumerate(record["base_supports"]):
        bases[logical, support] = 1
    return bases


def solve_pattern(
    bases: np.ndarray,
    stabilizers: np.ndarray,
    pattern: tuple[int, int, int],
    *,
    timeout_ms: int,
    random_seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=random_seed)
    coefficients = {
        logical: [z3.Bool(f"c_{logical}_{stab}") for stab in range(len(stabilizers))]
        for logical in TARGET_BATCH
    }
    representatives = {
        logical: [z3.Bool(f"r_{logical}_{qubit}") for qubit in range(64)]
        for logical in TARGET_BATCH
    }
    for logical, weight in zip(TARGET_BATCH, pattern):
        for qubit in range(64):
            parity = z3.BoolVal(bool(bases[logical, qubit]))
            for stab in np.flatnonzero(stabilizers[:, qubit]):
                parity = z3.Xor(parity, coefficients[logical][int(stab)])
            solver.add(representatives[logical][qubit] == parity)
        solver.add(z3.PbEq([(value, 1) for value in representatives[logical]], weight))
    for left_index, left in enumerate(TARGET_BATCH):
        for right in TARGET_BATCH[left_index + 1 :]:
            for qubit in range(64):
                solver.add(
                    z3.Or(
                        z3.Not(representatives[left][qubit]),
                        z3.Not(representatives[right][qubit]),
                    )
                )
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "pattern": list(pattern),
        "status": str(status),
        "seconds": round(elapsed, 6),
        "timeout_seconds": timeout_ms / 1000,
        "random_seed": random_seed,
    }
    if status == z3.unknown:
        result["reason_unknown"] = solver.reason_unknown()
        return result, None
    if status == z3.unsat:
        return result, None
    model = solver.model()
    proposed = np.zeros((8, 64), dtype=np.uint8)
    for logical in TARGET_BATCH:
        coefficient_values = np.asarray(
            [
                int(z3.is_true(model.eval(value, model_completion=True)))
                for value in coefficients[logical]
            ],
            dtype=np.uint8,
        )
        proposed[logical] = bases[logical] ^ ((coefficient_values @ stabilizers) % 2)
    result["supports"] = {
        str(logical): np.flatnonzero(proposed[logical]).astype(int).tolist()
        for logical in TARGET_BATCH
    }
    result["weights"] = {
        str(logical): int(proposed[logical].sum()) for logical in TARGET_BATCH
    }
    return result, proposed


def solve_individual(
    base: np.ndarray, stabilizers: np.ndarray, logical: int, timeout_ms: int
) -> dict[str, Any]:
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    coefficients = [z3.Bool(f"single_c_{logical}_{stab}") for stab in range(len(stabilizers))]
    representatives = [z3.Bool(f"single_r_{logical}_{qubit}") for qubit in range(64)]
    for qubit in range(64):
        parity = z3.BoolVal(bool(base[qubit]))
        for stab in np.flatnonzero(stabilizers[:, qubit]):
            parity = z3.Xor(parity, coefficients[int(stab)])
        solver.add(representatives[qubit] == parity)
    solver.add(z3.PbEq([(value, 1) for value in representatives], 8))
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "logical": logical,
        "weight": 8,
        "status": str(status),
        "seconds": round(elapsed, 6),
    }
    if status == z3.unknown:
        result["reason_unknown"] = solver.reason_unknown()
    elif status == z3.sat:
        model = solver.model()
        result["support"] = [
            qubit
            for qubit, value in enumerate(representatives)
            if z3.is_true(model.eval(value, model_completion=True))
        ]
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source = np.load(args.source_npz)
    check = np.asarray(source["matrix_z"], dtype=np.uint8)
    stabilizers = independent_rows(check)
    bases = bases_from_json(args.success_json)
    timeout_ms = max(1, round(1000 * args.seconds_per_query))
    started = time.perf_counter()
    individual_results = []
    for logical in TARGET_BATCH:
        emit("individual_start", logical=logical, target_weight=8)
        result = solve_individual(bases[logical], stabilizers, logical, timeout_ms)
        individual_results.append(result)
        emit("individual_complete", **result)
        atomic_json(args.output_dir / "checkpoint.json", {
            "phase": "individual", "individual_results": individual_results
        })
    atomic_json(args.output_dir / "individual_weight8.json", individual_results)

    pattern_results = []
    success_result = None
    success_representatives = None
    for pattern_index, pattern in enumerate(WEIGHT_PATTERNS):
        emit("pattern_start", pattern=list(pattern), pattern_index=pattern_index)
        result, proposed = solve_pattern(
            bases,
            stabilizers,
            pattern,
            timeout_ms=timeout_ms,
            random_seed=args.random_seed + pattern_index,
        )
        pattern_results.append(result)
        emit("pattern_complete", pattern_index=pattern_index, **result)
        atomic_json(args.output_dir / "checkpoint.json", {
            "phase": "patterns", "individual_results": individual_results,
            "pattern_results": pattern_results,
        })
        if proposed is not None:
            success_result = result
            success_representatives = proposed
            break
    atomic_json(args.output_dir / "pattern_results.json", pattern_results)
    if success_representatives is not None:
        original = np.asarray(source["logical_z"], dtype=np.uint8).copy()
        original[list(TARGET_BATCH)] = success_representatives[list(TARGET_BATCH)]
        np.savez_compressed(
            args.output_dir / "improved_batch.npz",
            logical_z=original,
            target_batch=np.asarray(TARGET_BATCH, dtype=np.int64),
            target_weights=np.asarray(success_result["pattern"], dtype=np.int64),
        )
        atomic_json(args.output_dir / "success.json", success_result)
    summary = {
        "target_batch": list(TARGET_BATCH),
        "individual_results": individual_results,
        "pattern_results": pattern_results,
        "improvement_found": success_result is not None,
        "success": success_result,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    emit("complete", improvement_found=success_result is not None,
         elapsed_seconds=summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
