#!/usr/bin/env python3
"""Solve the exact BB64 circuit fault distance using Stim's MaxSAT encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import stim
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    FaultEffect,
    atomic_json,
    detector_error_model,
    explain_effect,
    utc_now,
)

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--experiment", choices=("bulk", "memory"), default="memory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--fault-model", choices=("cnot", "no-idle", "full"), default="full")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def noise_from_args(name: str, probability: float) -> NoiseModel:
    if name == "cnot":
        return NoiseModel.cnot_only(probability)
    if name == "no-idle":
        return NoiseModel.without_idles(probability)
    return NoiseModel(probability)


def solve_maxsat(problem: str, output: mp.Queue[Any]) -> None:
    try:
        formula = WCNF(from_string=problem)
        started = time.monotonic()
        with RC2(formula, solver="g3", adapt=True, exhaust=True, minz=True) as solver:
            model = solver.compute()
            cost = solver.cost
        output.put(
            {
                "status": "optimal" if model is not None else "unsat",
                "cost": None if cost is None else int(cost),
                "model": model,
                "solver_seconds": round(time.monotonic() - started, 3),
            }
        )
    except BaseException:  # pragma: no cover - reports child failures to the parent
        output.put({"status": "error", "traceback": traceback.format_exc()})


def clause_satisfied(clause: list[int], positive: set[int]) -> bool:
    return any(
        (literal > 0 and literal in positive)
        or (literal < 0 and -literal not in positive)
        for literal in clause
    )


def raw_dem_effects(dem: stim.DetectorErrorModel) -> tuple[FaultEffect, ...]:
    effects = []
    for instruction_index, instruction in enumerate(dem):
        if instruction.type != "error":
            continue
        detector_mask = 0
        observable_mask = 0
        for target in instruction.targets_copy():
            if target.is_separator():
                raise ValueError("unexpected decomposed DEM separator")
            if target.is_relative_detector_id():
                detector_mask ^= 1 << int(target.val)
            elif target.is_logical_observable_id():
                observable_mask ^= 1 << int(target.val)
            else:
                raise ValueError(f"unexpected DEM target: {target}")
        effects.append(
            FaultEffect(
                detector_mask=detector_mask,
                observable_mask=observable_mask,
                probability=float(instruction.args_copy()[0]),
                dem_instruction_index=instruction_index,
            )
        )
    return tuple(effects)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output}")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    configuration = {
        "schema_version": 1,
        "schedule": str(args.schedule.resolve()),
        "experiment": args.experiment,
        "rounds": args.rounds,
        "bases": list(bases),
        "fault_model": args.fault_model,
        "probability": args.probability,
        "solver": "PySAT RC2 over Stim shortest_error_sat_problem WDIMACS",
    }
    checkpoint = args.output.with_suffix(".checkpoint.json")
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["configuration"] != configuration:
            raise SystemExit("resume configuration mismatch")
        state["status"] = "running"
    else:
        state: dict[str, Any] = {
            "schema_version": 1,
            "started": utc_now(),
            "updated": utc_now(),
            "status": "running",
            "configuration": configuration,
            "results": {},
            "active": None,
        }
    atomic_json(checkpoint, state)
    code = load_code_data(args.schedule)
    noise = noise_from_args(args.fault_model, args.probability)
    total_started = time.monotonic()
    for basis in bases:
        if basis in state["results"]:
            continue
        circuit = (
            build_bulk_circuit(basis, noise=noise, code=code)
            if args.experiment == "bulk"
            else build_memory_circuit(basis, args.rounds, noise=noise, code=code)
        )
        problem_started = time.monotonic()
        print(
            f"checkpoint {utc_now()} basis={basis} phase=maxsat-encode-start "
            f"detectors={circuit.num_detectors}",
            flush=True,
        )
        problem = circuit.shortest_error_sat_problem(format="WDIMACS")
        formula = WCNF(from_string=problem)
        problem_digest = hashlib.sha256(problem.encode()).hexdigest()
        encoding = {
            "sha256": problem_digest,
            "bytes": len(problem.encode()),
            "variables": formula.nv,
            "hard_clauses": len(formula.hard),
            "soft_clauses": len(formula.soft),
            "soft_weight_sum": sum(int(weight) for weight in formula.wght),
            "encoding_seconds": round(time.monotonic() - problem_started, 3),
        }
        state["active"] = {"basis": basis, "phase": "maxsat-solve", "encoding": encoding}
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=maxsat-solve-start "
            f"variables={formula.nv} hard={len(formula.hard)} soft={len(formula.soft)}",
            flush=True,
        )
        context = mp.get_context("fork")
        output: mp.Queue[Any] = context.Queue()
        process = context.Process(target=solve_maxsat, args=(problem, output))
        solve_started = time.monotonic()
        process.start()
        while process.is_alive():
            process.join(timeout=args.heartbeat_seconds)
            if process.is_alive():
                elapsed = round(time.monotonic() - solve_started, 3)
                state["active"] = {
                    "basis": basis,
                    "phase": "maxsat-solve",
                    "solver_elapsed_seconds": elapsed,
                    "encoding": encoding,
                }
                state["updated"] = utc_now()
                atomic_json(checkpoint, state)
                print(
                    f"checkpoint {utc_now()} basis={basis} phase=maxsat-solve-heartbeat "
                    f"solver_elapsed_seconds={elapsed}",
                    flush=True,
                )
        try:
            solved = output.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError(f"MaxSAT child exited with code {process.exitcode} without a result") from error
        if solved["status"] != "optimal":
            raise RuntimeError(f"MaxSAT failed: {solved}")
        model = [int(value) for value in solved.pop("model")]
        positive = {value for value in model if value > 0}
        unsatisfied = [
            index
            for index, clause in enumerate(formula.soft)
            if not clause_satisfied(clause, positive)
        ]
        unsatisfied_cost = sum(int(formula.wght[index]) for index in unsatisfied)
        if unsatisfied_cost != int(solved["cost"]):
            raise RuntimeError("recomputed MaxSAT model cost does not match RC2")

        dem = detector_error_model(circuit)
        effects = raw_dem_effects(dem)
        if len(effects) != len(formula.soft):
            raise RuntimeError(
                f"Stim soft-clause count {len(formula.soft)} does not match DEM errors {len(effects)}"
            )
        selected = [effects[index] for index in unsatisfied]
        detector_mask = 0
        observable_mask = 0
        for effect in selected:
            detector_mask ^= effect.detector_mask
            observable_mask ^= effect.observable_mask
        witness_valid = detector_mask == 0 and observable_mask != 0
        if not witness_valid:
            raise RuntimeError("MaxSAT soft-clause order did not map to a valid DEM witness")
        result = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "distance": int(solved["cost"]),
            "exact": True,
            "solver": {key: value for key, value in solved.items() if key != "cost"},
            "encoding": encoding,
            "num_qubits": circuit.num_qubits,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "dem_error_count": dem.num_errors,
            "witness": {
                "fault_count": len(selected),
                "soft_clause_indices": unsatisfied,
                "observable_mask": observable_mask,
                "observable_indices": [
                    index
                    for index in range(circuit.num_observables)
                    if observable_mask >> index & 1
                ],
                "detector_mask": detector_mask,
                "effects": [
                    {
                        "dem_instruction_index": effect.dem_instruction_index,
                        "detector_mask": effect.detector_mask,
                        "observable_mask": effect.observable_mask,
                        "representative_circuit_error": explain_effect(circuit, effect),
                    }
                    for effect in selected
                ],
                "validated_zero_detector_nonzero_observable": witness_valid,
            },
            "elapsed_seconds": round(time.monotonic() - problem_started, 3),
        }
        state["results"][basis] = result
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=maxsat-solve-finish "
            f"distance={result['distance']} solver_seconds={solved['solver_seconds']} "
            f"witness_valid={witness_valid}",
            flush=True,
        )
    state.update(
        {
            "status": "complete",
            "completed": utc_now(),
            "updated": utc_now(),
            "active": None,
            "elapsed_seconds_this_invocation": round(time.monotonic() - total_started, 3),
            "distance_by_basis": {
                basis: int(state["results"][basis]["distance"]) for basis in bases
            },
        }
    )
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
