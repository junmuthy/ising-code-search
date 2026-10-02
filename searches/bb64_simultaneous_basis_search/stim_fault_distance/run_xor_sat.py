#!/usr/bin/env python3
"""Solve the exact-five BB64 fault question using native XOR constraints."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import pycryptosat
import stim
from pysat.card import CardEnc, EncType

from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    FaultEffect,
    atomic_json,
    detector_error_model,
    explain_effect,
    utc_now,
)

from .circuit import NoiseModel, build_bulk_circuit, build_memory_circuit
from .effects import raw_dem_effects, translation_anchor_variables
from .model import load_code_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--experiment", choices=("bulk", "memory"), default="memory")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--fault-model", choices=("cnot", "no-idle", "full"), default="full")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--fault-count", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
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


def solve_xor_problem(
    detector_supports: list[list[int]],
    observable_supports: list[list[int]],
    translation_anchor_variables: list[int],
    fault_count: int,
    threads: int,
    output: mp.Queue[Any],
) -> None:
    try:
        started = time.monotonic()
        num_faults = max(
            variable
            for support in detector_supports + observable_supports
            for variable in support
        )
        logical_variables = [
            num_faults + index + 1 for index in range(len(observable_supports))
        ]
        solver = pycryptosat.Solver(threads=threads)
        for support in detector_supports:
            if support:
                solver.add_xor_clause(support, False)
        for support, logical in zip(
            observable_supports, logical_variables, strict=True
        ):
            solver.add_xor_clause([*support, logical], False)
        solver.add_clause(logical_variables)
        cardinality = CardEnc.equals(
            lits=list(range(1, num_faults + 1)),
            bound=fault_count,
            top_id=logical_variables[-1],
            encoding=EncType.seqcounter,
        )
        for clause in cardinality.clauses:
            solver.add_clause(clause)
        satisfiable = False
        model = None
        anchors_tested = 0
        for anchor_index, anchor in enumerate(translation_anchor_variables):
            satisfiable, model = solver.solve(assumptions=[anchor])
            anchors_tested = anchor_index + 1
            if satisfiable:
                break
            if anchors_tested % 5 == 0:
                output.put(
                    {
                        "event": "anchor-progress",
                        "anchors_tested": anchors_tested,
                        "anchors_total": len(translation_anchor_variables),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
        selected = (
            None
            if not satisfiable
            else [
                variable - 1
                for variable in range(1, num_faults + 1)
                if bool(model[variable])
            ]
        )
        output.put(
            {
                "status": "sat" if satisfiable else "unsat",
                "selected_effect_indices": selected,
                "solver_seconds": round(time.monotonic() - started, 3),
                "fault_variables": num_faults,
                "logical_variables": len(logical_variables),
                "detector_xor_clauses": len(detector_supports),
                "observable_xor_clauses": len(observable_supports),
                "translation_anchor_variables": len(translation_anchor_variables),
                "translation_anchors_tested": anchors_tested,
                "cardinality_auxiliary_variables": cardinality.nv - logical_variables[-1],
                "cardinality_clauses": len(cardinality.clauses),
                "threads": threads,
            }
        )
    except BaseException:  # pragma: no cover - reports child failures to the parent
        output.put({"status": "error", "traceback": traceback.format_exc()})


def supports(
    effects: tuple[FaultEffect, ...], num_detectors: int, num_observables: int
) -> tuple[list[list[int]], list[list[int]]]:
    detector_supports = [[] for _ in range(num_detectors)]
    observable_supports = [[] for _ in range(num_observables)]
    for effect_index, effect in enumerate(effects):
        variable = effect_index + 1
        for detector in range(effect.detector_mask.bit_length()):
            if effect.detector_mask >> detector & 1:
                detector_supports[detector].append(variable)
        for observable in range(effect.observable_mask.bit_length()):
            if effect.observable_mask >> observable & 1:
                observable_supports[observable].append(variable)
    return detector_supports, observable_supports


def witness(
    circuit: stim.Circuit,
    effects: tuple[FaultEffect, ...],
    selected_indices: list[int],
) -> dict[str, Any]:
    selected = [effects[index] for index in selected_indices]
    detector_mask = 0
    observable_mask = 0
    for effect in selected:
        detector_mask ^= effect.detector_mask
        observable_mask ^= effect.observable_mask
    valid = detector_mask == 0 and observable_mask != 0
    if not valid:
        raise RuntimeError("native-XOR SAT model did not produce a valid witness")
    return {
        "fault_count": len(selected),
        "effect_indices": selected_indices,
        "detector_mask": detector_mask,
        "observable_mask": observable_mask,
        "observable_indices": [
            index
            for index in range(circuit.num_observables)
            if observable_mask >> index & 1
        ],
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
    if args.fault_count < 1:
        raise SystemExit("--fault-count must be positive")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
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
        "fault_count": args.fault_count,
        "threads": args.threads,
        "solver": "CryptoMiniSat native detector/observable XOR plus exact cardinality",
        "required_prior_lower_bound": args.fault_count,
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
        dem = detector_error_model(circuit)
        effects = raw_dem_effects(dem)
        detector_supports, observable_supports = supports(
            effects, circuit.num_detectors, circuit.num_observables
        )
        anchor_variables = translation_anchor_variables(circuit, effects)
        context = mp.get_context("fork")
        output: mp.Queue[Any] = context.Queue()
        process = context.Process(
            target=solve_xor_problem,
            args=(
                detector_supports,
                observable_supports,
                anchor_variables,
                args.fault_count,
                args.threads,
                output,
            ),
        )
        started = time.monotonic()
        print(
            f"checkpoint {utc_now()} basis={basis} phase=xor-sat-start "
            f"fault_count={args.fault_count} effects={len(effects)} "
            f"detectors={circuit.num_detectors} observables={circuit.num_observables} "
            f"translation_anchor_variables={len(anchor_variables)} threads={args.threads}",
            flush=True,
        )
        process.start()
        solved: dict[str, Any] | None = None
        while process.is_alive() and solved is None:
            try:
                message = output.get(timeout=args.heartbeat_seconds)
            except queue.Empty:
                elapsed = round(time.monotonic() - started, 3)
                state["active"] = {
                    "basis": basis,
                    "phase": "xor-sat",
                    "fault_count": args.fault_count,
                    "solver_elapsed_seconds": elapsed,
                }
                state["updated"] = utc_now()
                atomic_json(checkpoint, state)
                print(
                    f"checkpoint {utc_now()} basis={basis} phase=xor-sat-heartbeat "
                    f"fault_count={args.fault_count} solver_elapsed_seconds={elapsed}",
                    flush=True,
                )
            else:
                if "event" in message:
                    state["active"] = {"basis": basis, "phase": "xor-sat", **message}
                    state["updated"] = utc_now()
                    atomic_json(checkpoint, state)
                    print(
                        f"checkpoint {utc_now()} basis={basis} phase=xor-sat-anchor "
                        f"anchors_tested={message['anchors_tested']}/"
                        f"{message['anchors_total']} "
                        f"solver_elapsed_seconds={message['elapsed_seconds']}",
                        flush=True,
                    )
                else:
                    solved = message
        process.join()
        while solved is None:
            try:
                message = output.get(timeout=5)
            except queue.Empty as error:
                raise RuntimeError(
                    f"native-XOR child exited with code {process.exitcode} without a result"
                ) from error
            if "event" not in message:
                solved = message
        if solved["status"] == "error":
            raise RuntimeError(solved["traceback"])
        selected = solved.pop("selected_effect_indices")
        record = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "status": solved["status"],
            "tested_fault_count": args.fault_count,
            "lower_bound": args.fault_count + 1 if solved["status"] == "unsat" else args.fault_count,
            "distance": args.fault_count if solved["status"] == "sat" else None,
            "exact_given_prior_lower_bound": solved["status"] == "sat",
            "solver": solved,
            "witness": None
            if selected is None
            else witness(circuit, effects, [int(index) for index in selected]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        state["results"][basis] = record
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=xor-sat-finish "
            f"status={record['status']} lower_bound={record['lower_bound']} "
            f"distance={record['distance']} solver_seconds={solved['solver_seconds']}",
            flush=True,
        )
    state.update(
        {
            "status": "complete",
            "completed": utc_now(),
            "updated": utc_now(),
            "active": None,
            "elapsed_seconds_this_invocation": round(time.monotonic() - total_started, 3),
            "conclusion_by_basis": {
                basis: {
                    "distance": state["results"][basis]["distance"],
                    "lower_bound": state["results"][basis]["lower_bound"],
                }
                for basis in bases
            },
        }
    )
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
