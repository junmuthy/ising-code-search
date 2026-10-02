"""Exact fault-distance certification from Stim detector signatures."""

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import stim
import z3


ProgressCallback = Callable[[dict[str, Any]], None]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically so interrupted searches retain valid checkpoints."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class FaultEffect:
    """One distinct detector/observable signature of an elementary fault."""

    detector_mask: int
    observable_mask: int
    probability: float
    dem_instruction_index: int

    @property
    def detector_weight(self) -> int:
        return self.detector_mask.bit_count()

    @property
    def observable_weight(self) -> int:
        return self.observable_mask.bit_count()


def detector_error_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Generate the undecomposed hypergraph detector error model."""

    return circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
        allow_gauge_detectors=False,
        approximate_disjoint_errors=True,
    )


def extract_fault_effects(
    dem: stim.DetectorErrorModel,
) -> tuple[FaultEffect, ...]:
    """Extract and deduplicate elementary fault signatures from a DEM.

    Stim can merge circuit faults having identical effects. Multiplicity is
    irrelevant to minimum fault cardinality: selecting the same signature
    twice cancels, and two distinct Pauli outcomes at one depolarizing channel
    multiply to another allowed one-fault Pauli outcome.
    """

    effects: dict[tuple[int, int], FaultEffect] = {}
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
        if detector_mask == 0 and observable_mask == 0:
            continue
        probability = float(instruction.args_copy()[0])
        key = (detector_mask, observable_mask)
        previous = effects.get(key)
        if previous is None or probability > previous.probability:
            effects[key] = FaultEffect(
                detector_mask=detector_mask,
                observable_mask=observable_mask,
                probability=probability,
                dem_instruction_index=instruction_index,
            )
    return tuple(
        sorted(
            effects.values(),
            key=lambda effect: (
                effect.detector_weight,
                effect.observable_weight,
                effect.detector_mask,
                effect.observable_mask,
            ),
        )
    )


def _parity(variables: list[z3.BoolRef]) -> z3.BoolRef:
    if not variables:
        return z3.BoolVal(False)
    if len(variables) == 1:
        return variables[0]
    level = list(variables)
    while len(level) > 1:
        next_level: list[z3.BoolRef] = []
        for index in range(0, len(level) - 1, 2):
            next_level.append(z3.Xor(level[index], level[index + 1]))
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _wait_with_heartbeats(
    function: Callable[[], Any],
    heartbeat_seconds: float,
    callback: ProgressCallback,
    context: dict[str, Any],
) -> Any:
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function)
        while True:
            try:
                return future.result(timeout=heartbeat_seconds)
            except concurrent.futures.TimeoutError:
                callback(
                    {
                        **context,
                        "event": context.get("heartbeat_event", "solver-heartbeat"),
                        "timestamp": utc_now(),
                        "solver_elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )


def _find_low_cardinality_witness(
    effects: tuple[FaultEffect, ...],
    fault_count: int,
    *,
    heartbeat_seconds: float,
    callback: ProgressCallback,
) -> list[int] | None:
    """Exactly search fault cardinalities one through three."""

    if fault_count == 1:
        for index, effect in enumerate(effects):
            if effect.detector_mask == 0 and effect.observable_mask:
                return [index]
        return None

    by_detector: dict[int, list[int]] = {}
    for index, effect in enumerate(effects):
        by_detector.setdefault(effect.detector_mask, []).append(index)

    if fault_count == 2:
        for indices in by_detector.values():
            for offset, left in enumerate(indices):
                for right in indices[offset + 1 :]:
                    if effects[left].observable_mask != effects[right].observable_mask:
                        return [left, right]
        return None

    if fault_count != 3:
        raise ValueError("direct exact enumeration supports only one to three faults")

    pair_total = len(effects) * (len(effects) - 1) // 2
    pair_count = 0
    last_checkpoint = time.monotonic()
    for left in range(len(effects)):
        effect_left = effects[left]
        for right in range(left + 1, len(effects)):
            effect_right = effects[right]
            target_detector = effect_left.detector_mask ^ effect_right.detector_mask
            for third in by_detector.get(target_detector, ()):
                if third == left or third == right:
                    continue
                observable_mask = (
                    effect_left.observable_mask
                    ^ effect_right.observable_mask
                    ^ effects[third].observable_mask
                )
                if observable_mask:
                    return [left, right, third]
            pair_count += 1
        now = time.monotonic()
        if now - last_checkpoint >= heartbeat_seconds:
            callback(
                {
                    "event": "exact-enumeration-heartbeat",
                    "timestamp": utc_now(),
                    "fault_bound": fault_count,
                    "pairs_tested": pair_count,
                    "pairs_total": pair_total,
                    "fraction_complete": round(pair_count / pair_total, 6),
                }
            )
            last_checkpoint = now
    return None


def _xor_masks(effects: Iterable[FaultEffect]) -> tuple[int, int]:
    detector_mask = 0
    observable_mask = 0
    for effect in effects:
        detector_mask ^= effect.detector_mask
        observable_mask ^= effect.observable_mask
    return detector_mask, observable_mask


def explain_effect(
    circuit: stim.Circuit,
    effect: FaultEffect,
) -> str | None:
    """Return one concrete circuit mechanism producing an effect, if available."""

    targets: list[str] = []
    for detector in range(effect.detector_mask.bit_length()):
        if effect.detector_mask >> detector & 1:
            targets.append(f"D{detector}")
    for observable in range(effect.observable_mask.bit_length()):
        if effect.observable_mask >> observable & 1:
            targets.append(f"L{observable}")
    filtered = stim.DetectorErrorModel(f"error(1) {' '.join(targets)}")
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=filtered,
        reduce_to_one_representative_error=True,
    )
    if not explained or not explained[0].circuit_error_locations:
        return None
    return str(explained[0].circuit_error_locations[0])


def certify_fault_distance(
    circuit: stim.Circuit,
    *,
    max_faults: int = 6,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Exactly minimize elementary fault count with zero detector syndrome."""

    if max_faults < 1:
        raise ValueError("max_faults must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    callback = progress or (lambda _event: None)
    started = time.monotonic()
    dem = detector_error_model(circuit)
    effects = extract_fault_effects(dem)
    callback(
        {
            "event": "dem-ready",
            "timestamp": utc_now(),
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "dem_errors": dem.num_errors,
            "distinct_fault_effects": len(effects),
        }
    )
    if not effects:
        raise ValueError("the noisy circuit has no observable fault effects")

    heuristic_context = {
        "event": "heuristic-start",
        "timestamp": utc_now(),
        "distinct_fault_effects": len(effects),
    }
    callback(heuristic_context)
    heuristic_started = time.monotonic()
    heuristic_errors = _wait_with_heartbeats(
        lambda: circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=8,
            dont_explore_edges_with_degree_above=12,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        ),
        heartbeat_seconds,
        callback,
        {
            "heartbeat_event": "heuristic-heartbeat",
            "distinct_fault_effects": len(effects),
        },
    )
    heuristic_elapsed = time.monotonic() - heuristic_started
    heuristic_upper_bound = len(heuristic_errors) if heuristic_errors else None
    callback(
        {
            "event": "heuristic-finish",
            "timestamp": utc_now(),
            "upper_bound": heuristic_upper_bound,
            "elapsed_seconds": round(heuristic_elapsed, 3),
        }
    )

    attempts: list[dict[str, Any]] = []
    witness: dict[str, Any] | None = None
    distance: int | None = None
    directly_tested = min(3, max_faults)
    for fault_bound in range(1, directly_tested + 1):
        callback(
            {
                "event": "exact-enumeration-start",
                "timestamp": utc_now(),
                "fault_bound": fault_bound,
                "distinct_fault_effects": len(effects),
            }
        )
        attempt_started = time.monotonic()
        selected_indices = _find_low_cardinality_witness(
            effects,
            fault_bound,
            heartbeat_seconds=heartbeat_seconds,
            callback=callback,
        )
        elapsed = time.monotonic() - attempt_started
        status = "sat" if selected_indices is not None else "unsat"
        attempts.append(
            {
                "fault_bound": fault_bound,
                "method": "exact-direct-enumeration",
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        callback(
            {
                "event": "exact-enumeration-finish",
                "timestamp": utc_now(),
                "fault_bound": fault_bound,
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        if selected_indices is not None:
            selected = [effects[index] for index in selected_indices]
            detector_mask, observable_mask = _xor_masks(selected)
            if detector_mask != 0 or observable_mask == 0:
                raise RuntimeError("direct enumeration returned an invalid witness")
            distance = len(selected)
            witness = {
                "source": "exact-direct-enumeration",
                "fault_count": distance,
                "observable_mask": observable_mask,
                "observable_indices": [
                    index
                    for index in range(circuit.num_observables)
                    if observable_mask >> index & 1
                ],
                "effects": [
                    {
                        **asdict(effect),
                        "representative_circuit_error": explain_effect(circuit, effect),
                    }
                    for effect in selected
                ],
            }
            break

    if (
        distance is None
        and heuristic_upper_bound is not None
        and heuristic_upper_bound <= max_faults
        and heuristic_upper_bound == directly_tested + 1
        and all(attempt["status"] == "unsat" for attempt in attempts)
    ):
        # The direct searches provide the exact lower bound. Stim's returned
        # ExplainedError list is a concrete circuit-level upper-bound witness.
        distance = heuristic_upper_bound
        witness = {
            "source": "stim-hypergraph-search",
            "fault_count": heuristic_upper_bound,
            "explained_errors": [str(error) for error in heuristic_errors],
        }

    if distance is not None:
        return {
            "schema_version": 1,
            "timestamp": utc_now(),
            "num_qubits": circuit.num_qubits,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "num_measurements": circuit.num_measurements,
            "dem_error_count": dem.num_errors,
            "distinct_fault_effect_count": len(effects),
            "heuristic_upper_bound": heuristic_upper_bound,
            "heuristic_elapsed_seconds": round(heuristic_elapsed, 3),
            "max_faults_tested": max_faults,
            "distance": distance,
            "lower_bound": distance,
            "exact_within_tested_range": True,
            "attempts": attempts,
            "witness": witness,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    variables = [z3.Bool(f"fault_{index}") for index in range(len(effects))]
    detector_supports: list[list[z3.BoolRef]] = [
        [] for _ in range(circuit.num_detectors)
    ]
    observable_supports: list[list[z3.BoolRef]] = [
        [] for _ in range(circuit.num_observables)
    ]
    for variable, effect in zip(variables, effects, strict=True):
        for detector in range(effect.detector_mask.bit_length()):
            if effect.detector_mask >> detector & 1:
                detector_supports[detector].append(variable)
        for observable in range(effect.observable_mask.bit_length()):
            if effect.observable_mask >> observable & 1:
                observable_supports[observable].append(variable)

    solver = z3.Solver()
    for support in detector_supports:
        solver.add(z3.Not(_parity(support)))
    logical_parities = [_parity(support) for support in observable_supports]
    solver.add(z3.Or(*logical_parities))
    cardinality_terms = [(variable, 1) for variable in variables]

    for fault_bound in range(directly_tested + 1, max_faults + 1):
        context = {
            "fault_bound": fault_bound,
            "distinct_fault_effects": len(effects),
        }
        callback({**context, "event": "solver-start", "timestamp": utc_now()})
        solver.push()
        solver.add(z3.PbLe(cardinality_terms, fault_bound))
        attempt_started = time.monotonic()
        status = _wait_with_heartbeats(
            solver.check,
            heartbeat_seconds,
            callback,
            context,
        )
        elapsed = time.monotonic() - attempt_started
        attempt: dict[str, Any] = {
            "fault_bound": fault_bound,
            "status": str(status),
            "elapsed_seconds": round(elapsed, 3),
        }
        callback(
            {
                **context,
                "event": "solver-finish",
                "timestamp": utc_now(),
                "status": str(status),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        if status == z3.sat:
            model = solver.model()
            selected_indices = [
                index
                for index, variable in enumerate(variables)
                if z3.is_true(model.eval(variable, model_completion=True))
            ]
            selected = [effects[index] for index in selected_indices]
            detector_mask, observable_mask = _xor_masks(selected)
            if detector_mask != 0 or observable_mask == 0:
                raise RuntimeError("solver returned an invalid logical witness")
            distance = len(selected)
            witness = {
                "fault_count": distance,
                "observable_mask": observable_mask,
                "observable_indices": [
                    index
                    for index in range(circuit.num_observables)
                    if observable_mask >> index & 1
                ],
                "effects": [
                    {
                        **asdict(effect),
                        "detector_indices": [
                            index
                            for index in range(circuit.num_detectors)
                            if effect.detector_mask >> index & 1
                        ],
                        "observable_indices": [
                            index
                            for index in range(circuit.num_observables)
                            if effect.observable_mask >> index & 1
                        ],
                        "representative_circuit_error": explain_effect(circuit, effect),
                    }
                    for effect in selected
                ],
            }
            attempt["witness_fault_count"] = distance
            attempts.append(attempt)
            solver.pop()
            break
        if status == z3.unknown:
            attempt["reason_unknown"] = solver.reason_unknown()
            attempts.append(attempt)
            solver.pop()
            break
        attempts.append(attempt)
        solver.pop()

    return {
        "schema_version": 1,
        "timestamp": utc_now(),
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_measurements": circuit.num_measurements,
        "dem_error_count": dem.num_errors,
        "distinct_fault_effect_count": len(effects),
        "heuristic_upper_bound": heuristic_upper_bound,
        "heuristic_elapsed_seconds": round(heuristic_elapsed, 3),
        "max_faults_tested": max_faults,
        "distance": distance,
        "lower_bound": distance if distance is not None else max_faults + 1,
        "exact_within_tested_range": all(
            attempt["status"] in ("unsat", "sat") for attempt in attempts
        ),
        "attempts": attempts,
        "witness": witness,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
