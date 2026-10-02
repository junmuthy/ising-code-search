#!/usr/bin/env python3
"""Decide whether an exact BB64 logical mechanism exists at five faults."""

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
from pysat.card import CardEnc, EncType
from pysat.formula import WCNF
from pysat.solvers import Solver

from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
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
    parser.add_argument("--fault-bound", type=int, default=5)
    parser.add_argument(
        "--fallback-bound",
        type=int,
        help="largest bound to test; defaults to --fault-bound",
    )
    parser.add_argument("--solver", default="cadical195")
    parser.add_argument(
        "--cardinality-encoding",
        choices=("sequential", "native"),
        default="sequential",
    )
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


def solve_bound(
    hard: list[list[int]],
    fault_variables: list[int],
    top_variable: int,
    bound: int,
    solver_name: str,
    cardinality_encoding: str,
    output: mp.Queue[Any],
) -> None:
    try:
        started = time.monotonic()
        if cardinality_encoding == "native":
            cardinality_clauses: list[list[int]] = []
            cardinality_auxiliary_variables = 0
            solver = Solver(name=solver_name, bootstrap_with=hard)
            solver.add_atmost(fault_variables, bound)
            solver.add_atmost([-variable for variable in fault_variables], len(fault_variables) - bound)
        else:
            cardinality = CardEnc.equals(
                lits=fault_variables,
                bound=bound,
                top_id=top_variable,
                encoding=EncType.seqcounter,
            )
            cardinality_clauses = cardinality.clauses
            cardinality_auxiliary_variables = cardinality.nv - top_variable
            solver = Solver(name=solver_name, bootstrap_with=hard + cardinality_clauses)
        with solver:
            satisfiable = solver.solve()
            model = solver.get_model() if satisfiable else None
            statistics = solver.accum_stats()
        output.put(
            {
                "status": "sat" if satisfiable else "unsat",
                "model": model,
                "solver_seconds": round(time.monotonic() - started, 3),
                "solver_statistics": statistics,
                "cardinality_auxiliary_variables": cardinality_auxiliary_variables,
                "cardinality_clauses": len(cardinality_clauses),
                "cardinality_encoding": cardinality_encoding,
            }
        )
    except BaseException:  # pragma: no cover - reports child failures to the parent
        output.put({"status": "error", "traceback": traceback.format_exc()})


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


def run_decision(
    *,
    hard: list[list[int]],
    fault_variables: list[int],
    top_variable: int,
    bound: int,
    solver_name: str,
    cardinality_encoding: str,
    heartbeat_seconds: float,
    basis: str,
    state: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    context = mp.get_context("fork")
    output: mp.Queue[Any] = context.Queue()
    process = context.Process(
        target=solve_bound,
        args=(
            hard,
            fault_variables,
            top_variable,
            bound,
            solver_name,
            cardinality_encoding,
            output,
        ),
    )
    started = time.monotonic()
    process.start()
    while process.is_alive():
        process.join(timeout=heartbeat_seconds)
        if process.is_alive():
            elapsed = round(time.monotonic() - started, 3)
            state["active"] = {
                "basis": basis,
                "phase": "bounded-sat",
                "fault_bound": bound,
                "solver_elapsed_seconds": elapsed,
            }
            state["updated"] = utc_now()
            atomic_json(checkpoint, state)
            print(
                f"checkpoint {utc_now()} basis={basis} phase=bounded-sat-heartbeat "
                f"fault_bound={bound} solver_elapsed_seconds={elapsed}",
                flush=True,
            )
    try:
        result = output.get(timeout=5)
    except queue.Empty as error:
        raise RuntimeError(f"SAT child exited with code {process.exitcode} without a result") from error
    if result["status"] == "error":
        raise RuntimeError(result["traceback"])
    result["fault_bound"] = bound
    return result


def validated_witness(
    circuit: stim.Circuit,
    effects: tuple[FaultEffect, ...],
    fault_variables: list[int],
    model: list[int],
) -> dict[str, Any]:
    positive = {int(value) for value in model if value > 0}
    selected_indices = [
        index for index, variable in enumerate(fault_variables) if variable in positive
    ]
    selected = [effects[index] for index in selected_indices]
    detector_mask = 0
    observable_mask = 0
    for effect in selected:
        detector_mask ^= effect.detector_mask
        observable_mask ^= effect.observable_mask
    valid = detector_mask == 0 and observable_mask != 0
    if not valid:
        raise RuntimeError("bounded-SAT model does not map to a valid DEM witness")
    return {
        "fault_count": len(selected),
        "soft_clause_indices": selected_indices,
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
        "validated_zero_detector_nonzero_observable": valid,
    }


def main() -> None:
    args = parse_args()
    fallback_bound = args.fault_bound if args.fallback_bound is None else args.fallback_bound
    if args.fault_bound < 1 or fallback_bound < args.fault_bound:
        raise SystemExit("invalid fault bounds")
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
        "fault_bound": args.fault_bound,
        "fallback_bound": fallback_bound,
        "solver": args.solver,
        "cardinality_encoding": args.cardinality_encoding,
        "encoding": (
            "Stim WDIMACS hard clauses plus exact-cardinality-k; exactness is valid "
            "because the separately saved meet-in-the-middle certificate excludes "
            "all smaller cardinalities"
        ),
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
        basis_started = time.monotonic()
        problem = circuit.shortest_error_sat_problem(format="WDIMACS")
        formula = WCNF(from_string=problem)
        if any(weight != 1 for weight in formula.wght):
            raise RuntimeError("expected unit-weight fault clauses")
        if any(len(clause) != 1 or clause[0] >= 0 for clause in formula.soft):
            raise RuntimeError("expected each soft clause to be a negative unit fault literal")
        fault_variables = [-int(clause[0]) for clause in formula.soft]
        if len(set(fault_variables)) != len(fault_variables):
            raise RuntimeError("fault variables are not unique")
        dem = detector_error_model(circuit)
        effects = raw_dem_effects(dem)
        if len(effects) != len(fault_variables):
            raise RuntimeError("Stim soft clauses do not correspond one-to-one with DEM errors")
        encoding = {
            "sha256": hashlib.sha256(problem.encode()).hexdigest(),
            "bytes": len(problem.encode()),
            "variables": formula.nv,
            "hard_clauses": len(formula.hard),
            "fault_variables": len(fault_variables),
        }
        decisions = []
        witness = None
        distance = None
        for bound in range(args.fault_bound, fallback_bound + 1):
            print(
                f"checkpoint {utc_now()} basis={basis} phase=bounded-sat-start "
                f"fault_bound={bound} variables={formula.nv} "
                f"hard={len(formula.hard)} faults={len(fault_variables)}",
                flush=True,
            )
            decision = run_decision(
                hard=formula.hard,
                fault_variables=fault_variables,
                top_variable=formula.nv,
                bound=bound,
                solver_name=args.solver,
                cardinality_encoding=args.cardinality_encoding,
                heartbeat_seconds=args.heartbeat_seconds,
                basis=basis,
                state=state,
                checkpoint=checkpoint,
            )
            model = decision.pop("model")
            decisions.append(decision)
            print(
                f"checkpoint {utc_now()} basis={basis} phase=bounded-sat-finish "
                f"fault_bound={bound} status={decision['status']} "
                f"solver_seconds={decision['solver_seconds']}",
                flush=True,
            )
            if decision["status"] == "sat":
                if model is None:
                    raise RuntimeError("SAT decision omitted its model")
                witness = validated_witness(circuit, effects, fault_variables, model)
                distance = int(witness["fault_count"])
                break
        result = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "encoding": encoding,
            "decisions": decisions,
            "distance": distance,
            "exact_given_prior_lower_bound": distance is not None,
            "lower_bound": distance if distance is not None else fallback_bound + 1,
            "required_prior_lower_bound": args.fault_bound,
            "witness": witness,
            "elapsed_seconds": round(time.monotonic() - basis_started, 3),
        }
        state["results"][basis] = result
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=bounded-sat-basis-finish "
            f"distance={distance}",
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
                basis: state["results"][basis]["distance"] for basis in bases
            },
        }
    )
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
