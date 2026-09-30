#!/usr/bin/env python3
"""Search the eight BB64 logical-class candidates for fewer STAR batches.

This is a deliberately separate follow-up to ``../search.py``.  The original
run stopped after its first three-batch solution; here every saved candidate is
tested first for one batch and then for two batches.  Stabilizer dressing never
changes the logical classes, the C4 x C2 action, or the exact transversal-H
pairing.

By default every representative is constrained to weight eight.  Since the
code distance is eight, this keeps every saved logical minimum weight.  In the
one-batch case, eight disjoint weight-eight supports must partition all 64 data
qubits, so the solver uses that stronger exact-cover formulation.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import z3

from search_helpers import (
    batch_disjointness,
    independent_rows,
    permutation_matrix_test,
    permute_rows,
)


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_BASIS = DIRECTORY / "search_input_basis.npz"
DEFAULT_CANDIDATES = DIRECTORY / "class_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-npz", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds-per-case", type=float, default=60.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--weight", type=int, default=8)
    parser.add_argument(
        "--batch-counts",
        type=int,
        nargs="+",
        choices=(1, 2),
        default=(1, 2),
        help="Counts to search, in the supplied order (default: 1 2).",
    )
    parser.add_argument(
        "--candidate-indices",
        type=int,
        nargs="*",
        help="Optional subset of the eight saved candidate indices.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run, retaining completed case records.",
    )
    parser.add_argument(
        "--retry-unknown",
        action="store_true",
        help="On resume, rerun cases that previously timed out.",
    )
    return parser.parse_args()


def emit(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def base_from_record(record: dict[str, Any]) -> np.ndarray:
    base = np.zeros((8, 64), dtype=np.uint8)
    for logical, support in enumerate(record["base_supports"]):
        base[logical, np.asarray(support, dtype=np.int64)] = 1
    return base


def partitions_from_colors(colors: Sequence[int], num_batches: int) -> list[list[int]]:
    return [
        [logical for logical, color in enumerate(colors) if color == batch]
        for batch in range(num_batches)
    ]


def x_partition_from_z(
    z_partition: Sequence[Sequence[int]], hadamard_permutation: np.ndarray
) -> list[list[int]]:
    """Map Z batches to X batches when X_i has support Z_{p(i)}."""
    inverse = np.argsort(np.asarray(hadamard_permutation, dtype=np.int64))
    return [sorted(int(inverse[index]) for index in batch) for batch in z_partition]


def certify_basis(
    check: np.ndarray,
    logical_z: np.ndarray,
    pairing: np.ndarray,
    hadamard_permutation: np.ndarray,
    z_partition: Sequence[Sequence[int]],
    physical_x: np.ndarray,
    physical_y: np.ndarray,
) -> dict[str, Any]:
    """Verify the properties that must survive a new stabilizer dressing."""
    logical_x = logical_z[hadamard_permutation]
    x_partition = x_partition_from_z(z_partition, hadamard_permutation)
    identity = np.eye(8, dtype=np.uint8)
    if not np.array_equal((logical_z @ logical_z.T) % 2, pairing):
        raise ValueError("dressing changed the saved logical pairing")
    if not np.array_equal((logical_z @ logical_x.T) % 2, identity):
        raise ValueError("Z/X representatives are not a canonical logical basis")
    if not batch_disjointness(logical_z, z_partition):
        raise ValueError("Z partition is not an exact cover by disjoint batches")
    if not batch_disjointness(logical_x, x_partition):
        raise ValueError("H-permuted X partition is not disjoint")
    if np.any((check @ logical_z.T) % 2) or np.any((check @ logical_x.T) % 2):
        raise ValueError("logical representative fails a stabilizer commutation check")

    actions: dict[str, np.ndarray] = {}
    for axis, physical in (("x", physical_x), ("y", physical_y)):
        actions[f"grid_{axis}_z_logical_action"] = (
            permute_rows(logical_z, physical) @ logical_x.T
        ) % 2
        actions[f"grid_{axis}_x_logical_action"] = (
            permute_rows(logical_x, physical) @ logical_z.T
        ) % 2
        if not permutation_matrix_test(actions[f"grid_{axis}_z_logical_action"]):
            raise ValueError(f"grid {axis} is not a Z-logical permutation")
        if not permutation_matrix_test(actions[f"grid_{axis}_x_logical_action"]):
            raise ValueError(f"grid {axis} is not an X-logical permutation")

    return {
        "logical_x": logical_x,
        "x_partition": x_partition,
        **actions,
    }


def _representative_variables(
    solver: z3.Solver,
    base: np.ndarray,
    stabilizers: np.ndarray,
    *,
    prefix: str,
    target_weight: int | None,
) -> tuple[list[list[z3.BoolRef]], list[list[z3.BoolRef]]]:
    coefficients = [
        [z3.Bool(f"{prefix}_c_{logical}_{stab}") for stab in range(len(stabilizers))]
        for logical in range(8)
    ]
    representatives = [
        [z3.Bool(f"{prefix}_r_{logical}_{qubit}") for qubit in range(64)]
        for logical in range(8)
    ]
    for logical in range(8):
        for qubit in range(64):
            parity: z3.BoolRef = z3.BoolVal(bool(base[logical, qubit]))
            for stab in np.flatnonzero(stabilizers[:, qubit]):
                # The project's installed Z3 accepts binary rather than fully
                # variadic Xor calls, so retain the established nested form.
                parity = z3.Xor(parity, coefficients[logical][int(stab)])
            solver.add(representatives[logical][qubit] == parity)
        if target_weight is not None:
            solver.add(
                z3.PbEq([(value, 1) for value in representatives[logical]], target_weight)
            )
    return coefficients, representatives


def build_solver(
    base: np.ndarray,
    stabilizers: np.ndarray,
    num_batches: int,
    *,
    target_weight: int | None,
    timeout_seconds: float,
    random_seed: int,
    prefix: str,
) -> tuple[z3.Solver, list[list[z3.BoolRef]], list[list[z3.BoolRef]], list[z3.BoolRef]]:
    solver = z3.Solver()
    solver.set(
        timeout=max(1, round(1000 * timeout_seconds)),
        random_seed=random_seed,
    )
    coefficients, representatives = _representative_variables(
        solver,
        base,
        stabilizers,
        prefix=prefix,
        target_weight=target_weight,
    )
    colors: list[z3.BoolRef] = []
    if num_batches == 1:
        if target_weight == 8:
            # Eight disjoint minimum-weight logicals on 64 qubits form an exact
            # partition.  This is stronger and cheaper than pairwise clauses.
            for qubit in range(64):
                solver.add(
                    z3.PbEq(
                        [(representatives[logical][qubit], 1) for logical in range(8)],
                        1,
                    )
                )
        else:
            for left in range(8):
                for right in range(left + 1, 8):
                    for qubit in range(64):
                        solver.add(
                            z3.Or(
                                z3.Not(representatives[left][qubit]),
                                z3.Not(representatives[right][qubit]),
                            )
                        )
    elif num_batches == 2:
        colors = [z3.Bool(f"{prefix}_batch_{logical}") for logical in range(8)]
        solver.add(z3.Not(colors[0]))
        solver.add(z3.PbGe([(z3.Not(color), 1) for color in colors], 2))
        solver.add(z3.PbGe([(color, 1) for color in colors], 2))
        for left in range(8):
            for right in range(left + 1, 8):
                separated = z3.Xor(colors[left], colors[right])
                for qubit in range(64):
                    solver.add(
                        z3.Or(
                            separated,
                            z3.Not(representatives[left][qubit]),
                            z3.Not(representatives[right][qubit]),
                        )
                    )
    else:
        raise ValueError("this focused search supports only one or two batches")
    return solver, coefficients, representatives, colors


def solve_case(
    base: np.ndarray,
    stabilizers: np.ndarray,
    num_batches: int,
    *,
    target_weight: int | None,
    timeout_seconds: float,
    random_seed: int,
    heartbeat_seconds: float,
    candidate_index: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    build_started = time.perf_counter()
    solver, coefficients, representatives, colors = build_solver(
        base,
        stabilizers,
        num_batches,
        target_weight=target_weight,
        timeout_seconds=timeout_seconds,
        random_seed=random_seed,
        prefix=f"candidate{candidate_index}_batches{num_batches}",
    )
    build_seconds = time.perf_counter() - build_started
    stop_heartbeat = threading.Event()
    solve_started = time.perf_counter()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_seconds):
            emit(
                "solver_heartbeat",
                candidate_index=candidate_index,
                num_batches=num_batches,
                elapsed_seconds=round(time.perf_counter() - solve_started, 3),
                timeout_seconds=timeout_seconds,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        status = solver.check()
    finally:
        stop_heartbeat.set()
        thread.join()
    solve_seconds = time.perf_counter() - solve_started
    result: dict[str, Any] = {
        "candidate_index": candidate_index,
        "num_batches": num_batches,
        "target_weight": target_weight,
        "status": str(status),
        "feasible": status == z3.sat if status != z3.unknown else None,
        "build_seconds": round(build_seconds, 6),
        "solve_seconds": round(solve_seconds, 6),
        "timeout_seconds": timeout_seconds,
        "random_seed": random_seed,
        "formulation": (
            "64-qubit exact partition" if num_batches == 1 and target_weight == 8
            else "conditional pairwise disjointness"
        ),
    }
    if status == z3.unknown:
        result["reason_unknown"] = solver.reason_unknown()
        return result, None
    if status == z3.unsat:
        return result, None

    model = solver.model()
    coefficient_values = np.asarray(
        [
            [int(z3.is_true(model.eval(value, model_completion=True))) for value in row]
            for row in coefficients
        ],
        dtype=np.uint8,
    )
    dressed = base ^ ((coefficient_values @ stabilizers) % 2)
    if num_batches == 1:
        partition = [list(range(8))]
    else:
        color_values = [
            int(z3.is_true(model.eval(color, model_completion=True))) for color in colors
        ]
        partition = partitions_from_colors(color_values, num_batches)
    if not batch_disjointness(dressed, partition):
        raise AssertionError("Z3 model failed post-solve disjointness verification")
    result.update(
        {
            "partition": partition,
            "batch_sizes": [len(batch) for batch in partition],
            "weights": np.count_nonzero(dressed, axis=1).astype(int).tolist(),
            "supports": [np.flatnonzero(row).astype(int).tolist() for row in dressed],
            "stabilizer_dressing_coefficients": coefficient_values.astype(int).tolist(),
        }
    )
    return result, dressed


def config_from_args(args: argparse.Namespace, candidate_indices: Sequence[int]) -> dict[str, Any]:
    return {
        "basis_npz": str(args.basis_npz.resolve()),
        "candidates_json": str(args.candidates_json.resolve()),
        "seconds_per_case": args.seconds_per_case,
        "heartbeat_seconds": args.heartbeat_seconds,
        "random_seed": args.random_seed,
        "target_weight": args.weight if args.weight > 0 else None,
        "batch_counts": list(args.batch_counts),
        "candidate_indices": list(candidate_indices),
    }


def result_key(num_batches: int, candidate_index: int) -> str:
    return f"batches{num_batches}:candidate{candidate_index}"


def save_solution(
    path: Path,
    source: Any,
    logical_z: np.ndarray,
    pairing: np.ndarray,
    hadamard_permutation: np.ndarray,
    z_partition: Sequence[Sequence[int]],
    certificate: dict[str, Any],
) -> None:
    z_batch = np.empty(8, dtype=np.int64)
    for batch_index, batch in enumerate(z_partition):
        z_batch[np.asarray(batch, dtype=np.int64)] = batch_index
    x_partition = certificate["x_partition"]
    x_batch = np.empty(8, dtype=np.int64)
    for batch_index, batch in enumerate(x_partition):
        x_batch[np.asarray(batch, dtype=np.int64)] = batch_index
    np.savez_compressed(
        path,
        matrix_x=np.asarray(source["matrix_x"], dtype=np.uint8),
        matrix_z=np.asarray(source["matrix_z"], dtype=np.uint8),
        logical_z=logical_z,
        logical_x=certificate["logical_x"],
        zx_pairing=pairing,
        hadamard_permutation=hadamard_permutation,
        grid_x_physical_permutation=np.asarray(
            source["grid_x_physical_permutation"], dtype=np.int64
        ),
        grid_y_physical_permutation=np.asarray(
            source["grid_y_physical_permutation"], dtype=np.int64
        ),
        grid_x_z_logical_action=certificate["grid_x_z_logical_action"],
        grid_y_z_logical_action=certificate["grid_y_z_logical_action"],
        grid_x_x_logical_action=certificate["grid_x_x_logical_action"],
        grid_y_x_logical_action=certificate["grid_y_x_logical_action"],
        disjoint_batch_of_z_logical=z_batch,
        disjoint_batch_of_x_logical=x_batch,
    )


def main() -> None:
    args = parse_args()
    if args.seconds_per_case <= 0 or args.heartbeat_seconds <= 0:
        raise SystemExit("solver and heartbeat times must be positive")
    candidates = json.loads(args.candidates_json.read_text())
    candidate_indices = (
        list(range(len(candidates)))
        if args.candidate_indices is None
        else list(args.candidate_indices)
    )
    if not candidate_indices or any(index not in range(len(candidates)) for index in candidate_indices):
        raise SystemExit(f"candidate indices must be in 0..{len(candidates) - 1}")
    config = config_from_args(args, candidate_indices)
    state_path = args.output_dir / "checkpoint.json"
    if args.output_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config"] != config:
            raise SystemExit("resume configuration differs from checkpoint configuration")
    else:
        state = {
            "config": config,
            "complete": False,
            "active_case": None,
            "results": {},
            "solutions": [],
        }
        atomic_json(state_path, state)

    source = np.load(args.basis_npz)
    check = np.asarray(source["matrix_z"], dtype=np.uint8)
    stabilizers = independent_rows(check)
    physical_x = np.asarray(source["grid_x_physical_permutation"], dtype=np.int64)
    physical_y = np.asarray(source["grid_y_physical_permutation"], dtype=np.int64)
    target_weight = args.weight if args.weight > 0 else None
    run_started = time.perf_counter()
    emit(
        "search_start",
        candidates=len(candidate_indices),
        batch_counts=list(args.batch_counts),
        target_weight=target_weight,
        seconds_per_case=args.seconds_per_case,
        completed_cases=len(state["results"]),
    )
    for num_batches in args.batch_counts:
        for candidate_index in candidate_indices:
            key = result_key(num_batches, candidate_index)
            previous = state["results"].get(key)
            if previous is not None and not (
                args.retry_unknown and previous["status"] == "unknown"
            ):
                emit("case_skip", case=key, status=previous["status"])
                continue
            state["active_case"] = {
                "key": key,
                "started_unix": time.time(),
            }
            atomic_json(state_path, state)
            emit(
                "case_start",
                case=key,
                candidate_index=candidate_index,
                num_batches=num_batches,
            )
            record = candidates[candidate_index]
            base = base_from_record(record)
            result, dressed = solve_case(
                base,
                stabilizers,
                num_batches,
                target_weight=target_weight,
                timeout_seconds=args.seconds_per_case,
                random_seed=args.random_seed + 1009 * num_batches + candidate_index,
                heartbeat_seconds=args.heartbeat_seconds,
                candidate_index=candidate_index,
            )
            if dressed is not None:
                pairing = np.asarray(record["pairing"], dtype=np.uint8)
                hadamard_permutation = np.asarray(
                    record["hadamard_permutation"], dtype=np.int64
                )
                certificate = certify_basis(
                    check,
                    dressed,
                    pairing,
                    hadamard_permutation,
                    result["partition"],
                    physical_x,
                    physical_y,
                )
                result["x_partition"] = certificate["x_partition"]
                artifact_name = f"basis_batches{num_batches}_candidate{candidate_index}.npz"
                save_solution(
                    args.output_dir / artifact_name,
                    source,
                    dressed,
                    pairing,
                    hadamard_permutation,
                    result["partition"],
                    certificate,
                )
                result["artifact"] = artifact_name
                state["solutions"].append(key)
            state["results"][key] = result
            state["active_case"] = None
            atomic_json(state_path, state)
            atomic_json(args.output_dir / "results.json", state["results"])
            emit(
                "case_complete",
                case=key,
                status=result["status"],
                feasible=result["feasible"],
                solve_seconds=result["solve_seconds"],
                partition=result.get("partition"),
            )

    records = list(state["results"].values())
    summary = {
        "config": config,
        "code": "self-dual [[64,8,8]] Liang-Chen BB code",
        "logical_map_preserved": "physical H maps Z_i to X_{p(i)} exactly",
        "hadamard_permutation": candidates[0]["hadamard_permutation"],
        "candidate_count": len(candidate_indices),
        "case_count": len(records),
        "status_histogram": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "solutions": state["solutions"],
        "one_batch_found": any(
            record["num_batches"] == 1 and record["status"] == "sat"
            for record in records
        ),
        "two_batch_found": any(
            record["num_batches"] == 2 and record["status"] == "sat"
            for record in records
        ),
        "interpretation": (
            "sat gives an exact witness; unsat is exact for the configured weight; "
            "unknown is unresolved because the per-case timeout expired"
        ),
        "elapsed_this_invocation_seconds": round(time.perf_counter() - run_started, 6),
    }
    state["complete"] = True
    state["summary"] = summary
    atomic_json(state_path, state)
    atomic_json(args.output_dir / "summary.json", summary)
    emit("search_complete", **summary)


if __name__ == "__main__":
    main()
