"""Fast heuristic screening and exact four-fault meet-in-the-middle checks."""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import asdict
from typing import Any, Callable, Iterable

import numpy as np
import stim

from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    FaultEffect,
    _find_low_cardinality_witness,
    _xor_masks,
    detector_error_model,
    explain_effect,
    extract_fault_effects,
    utc_now,
)


ProgressCallback = Callable[[dict[str, Any]], None]
MASK64 = (1 << 64) - 1


def heuristic_screen(
    circuit: stim.Circuit,
    *,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Stim's hypergraph search as a concrete, non-certifying upper bound."""

    callback = progress or (lambda _event: None)
    started = time.monotonic()

    def run() -> list[stim.ExplainedError]:
        return circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=8,
            dont_explore_edges_with_degree_above=12,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run)
        while True:
            try:
                errors = future.result(timeout=heartbeat_seconds)
                break
            except concurrent.futures.TimeoutError:
                callback(
                    {
                        "event": "heuristic-heartbeat",
                        "timestamp": utc_now(),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
    return {
        "upper_bound": len(errors) if errors else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "explained_errors": [str(error) for error in errors],
    }


def classify_explained_errors(
    explained_errors: Iterable[str], schedule: dict[str, Any]
) -> list[dict[str, Any]]:
    """Map Stim witness text back to schedule layers and Tanner edges."""

    pair_to_gate: dict[tuple[int, int], dict[str, int | str]] = {}
    for layer_index, layer in enumerate(schedule["layers"]):
        for gate in layer["gates"]:
            kind = str(gate["type"])
            check = int(gate["check"])
            data = int(gate["data"])
            pair = (32 + check, data) if kind == "X" else (data, 48 + check)
            pair_to_gate[pair] = {
                "kind": kind,
                "check": check,
                "data": data,
                "layer_zero_based": layer_index,
            }

    output: list[dict[str, Any]] = []
    for error in explained_errors:
        tick_match = re.search(r"after (\d+) TICKs", error)
        pauli_match = re.search(r"flipped_pauli_product:\s*([^\n]+)", error)
        resolving = next(
            (line.strip() for line in error.splitlines() if "resolving to DEPOLARIZE2" in line),
            None,
        )
        pair: tuple[int, int] | None = None
        if resolving is not None:
            without_coordinates = re.sub(r"\[[^\]]*\]", "", resolving)
            numbers = [int(value) for value in re.findall(r"(?<![.\d])\d+(?![.\d])", without_coordinates)]
            if len(numbers) >= 2:
                pair = (numbers[-2], numbers[-1])
        record: dict[str, Any] = {
            "tick_count_before_location": None if tick_match is None else int(tick_match.group(1)),
            "flipped_pauli_product": None if pauli_match is None else pauli_match.group(1),
            "cnot_pair": None if pair is None else list(pair),
        }
        if pair in pair_to_gate:
            record.update(pair_to_gate[pair])
        output.append(record)
    return output


def _four_fault_dtype(detector_chunks: int) -> np.dtype[Any]:
    fields: list[tuple[str, Any]] = [
        (f"detector_{chunk}", np.uint64) for chunk in range(detector_chunks)
    ]
    fields.extend(
        [
            ("observable", np.uint64),
            ("left", np.uint16),
            ("right", np.uint16),
        ]
    )
    return np.dtype(fields)


def estimate_four_fault_table(
    circuit: stim.Circuit,
) -> dict[str, int]:
    """Estimate the exact pair-table allocation before starting certification."""

    effects = extract_fault_effects(detector_error_model(circuit))
    detector_chunks = max(1, (circuit.num_detectors + 63) // 64)
    pair_count = len(effects) * (len(effects) - 1) // 2
    bytes_per_pair = _four_fault_dtype(detector_chunks).itemsize
    return {
        "distinct_fault_effects": len(effects),
        "detector_chunks": detector_chunks,
        "pair_count": pair_count,
        "bytes_per_pair": bytes_per_pair,
        "table_bytes": pair_count * bytes_per_pair,
    }


def find_four_fault_witness(
    effects: tuple[FaultEffect, ...],
    *,
    num_detectors: int,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> list[int] | None:
    """Exactly find four distinct signatures using sorted two-fault signatures.

    A four-fault mechanism is split into two disjoint pairs. The two pairs must
    have equal detector masks and unequal observable masks. Sorting compact
    NumPy records avoids the large Python-object overhead of a two-million-entry
    dictionary for the current 1,984-effect bulk circuit.
    """

    callback = progress or (lambda _event: None)
    count = len(effects)
    if count > np.iinfo(np.uint16).max:
        raise ValueError("too many distinct effects for uint16 pair indices")
    if any(effect.observable_mask.bit_length() > 64 for effect in effects):
        raise ValueError("more than 64 observables are not supported")
    detector_chunks = max(1, (num_detectors + 63) // 64)
    pair_total = count * (count - 1) // 2
    pairs = np.empty(pair_total, dtype=_four_fault_dtype(detector_chunks))
    position = 0
    last_checkpoint = time.monotonic()
    started = last_checkpoint
    for left in range(count):
        amount = count - left - 1
        if not amount:
            continue
        stop = position + amount
        left_effect = effects[left]
        right_effects = effects[left + 1 :]
        for chunk in range(detector_chunks):
            shift = 64 * chunk
            pairs[f"detector_{chunk}"][position:stop] = np.fromiter(
                (
                    ((left_effect.detector_mask ^ right.detector_mask) >> shift) & MASK64
                    for right in right_effects
                ),
                dtype=np.uint64,
                count=amount,
            )
        pairs["observable"][position:stop] = np.fromiter(
            (left_effect.observable_mask ^ right.observable_mask for right in right_effects),
            dtype=np.uint64,
            count=amount,
        )
        pairs["left"][position:stop] = left
        pairs["right"][position:stop] = np.arange(left + 1, count, dtype=np.uint16)
        position = stop
        now = time.monotonic()
        if now - last_checkpoint >= heartbeat_seconds:
            callback(
                {
                    "event": "exact-four-build-heartbeat",
                    "timestamp": utc_now(),
                    "pairs_built": position,
                    "pairs_total": pair_total,
                    "fraction_complete": round(position / pair_total, 6),
                }
            )
            last_checkpoint = now
    if position != pair_total:
        raise RuntimeError("pair table construction count mismatch")

    detector_fields = [f"detector_{chunk}" for chunk in range(detector_chunks)]
    callback(
        {
            "event": "exact-four-sort-start",
            "timestamp": utc_now(),
            "pairs_total": pair_total,
            "table_bytes": int(pairs.nbytes),
        }
    )
    pairs.sort(order=detector_fields)
    callback(
        {
            "event": "exact-four-sort-finish",
            "timestamp": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    )

    same_as_previous = np.ones(pair_total, dtype=np.bool_)
    same_as_previous[0] = False
    for field in detector_fields:
        same_as_previous[1:] &= pairs[field][1:] == pairs[field][:-1]
    group_starts = np.flatnonzero(~same_as_previous)
    group_stops = np.append(group_starts[1:], pair_total)
    last_checkpoint = time.monotonic()
    for group_number, (start_value, stop_value) in enumerate(
        zip(group_starts, group_stops, strict=True)
    ):
        start = int(start_value)
        stop = int(stop_value)
        if stop - start >= 2:
            block = pairs[start:stop]
            labels = np.unique(block["observable"])
            labelled = {
                int(label): block[block["observable"] == label] for label in labels
            }
            for left_label_index, left_label in enumerate(labels):
                left_block = labelled[int(left_label)]
                for right_label in labels[left_label_index + 1 :]:
                    right_block = labelled[int(right_label)]
                    # For an edge (u,v), the number of right-labelled edges
                    # touching it is degree(u)+degree(v). The (u,v) edge itself
                    # cannot occur with both labels, so there is no double-count
                    # correction. A strict deficit therefore proves and locates
                    # a disjoint partner in linear time.
                    degrees = np.bincount(
                        np.concatenate((right_block["left"], right_block["right"])),
                        minlength=count,
                    )
                    has_disjoint = len(right_block) > (
                        degrees[left_block["left"]] + degrees[left_block["right"]]
                    )
                    matches = np.flatnonzero(has_disjoint)
                    if not len(matches):
                        continue
                    first = left_block[int(matches[0])]
                    left = int(first["left"])
                    right = int(first["right"])
                    valid = (
                        (right_block["left"] != left)
                        & (right_block["right"] != left)
                        & (right_block["left"] != right)
                        & (right_block["right"] != right)
                    )
                    other_matches = np.flatnonzero(valid)
                    if not len(other_matches):
                        raise RuntimeError("degree test predicted a missing disjoint pair")
                    other = right_block[int(other_matches[0])]
                    return [left, right, int(other["left"]), int(other["right"])]
        now = time.monotonic()
        if now - last_checkpoint >= heartbeat_seconds:
            callback(
                {
                    "event": "exact-four-scan-heartbeat",
                    "timestamp": utc_now(),
                    "groups_scanned": group_number + 1,
                    "groups_total": len(group_starts),
                    "fraction_complete": round((group_number + 1) / len(group_starts), 6),
                }
            )
            last_checkpoint = now
    return None


def _witness_record(
    circuit: stim.Circuit,
    effects: tuple[FaultEffect, ...],
    indices: list[int],
    source: str,
) -> dict[str, Any]:
    selected = [effects[index] for index in indices]
    detector_mask, observable_mask = _xor_masks(selected)
    if detector_mask or not observable_mask:
        raise RuntimeError("invalid exact fault witness")
    return {
        "source": source,
        "fault_count": len(indices),
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


def certify_through_three(
    circuit: stim.Circuit,
    *,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Exactly find a logical mechanism of cardinality at most three."""

    callback = progress or (lambda _event: None)
    started = time.monotonic()
    effects = extract_fault_effects(detector_error_model(circuit))
    attempts: list[dict[str, Any]] = []
    for fault_count in range(1, 4):
        attempt_started = time.monotonic()
        indices = _find_low_cardinality_witness(
            effects,
            fault_count,
            heartbeat_seconds=heartbeat_seconds,
            callback=callback,
        )
        attempts.append(
            {
                "fault_count": fault_count,
                "method": "exact-direct-enumeration",
                "status": "sat" if indices is not None else "unsat",
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            }
        )
        if indices is not None:
            return {
                "lower_bound": fault_count,
                "distance": fault_count,
                "distinct_fault_effect_count": len(effects),
                "attempts": attempts,
                "witness": _witness_record(circuit, effects, indices, "exact-direct-enumeration"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
    return {
        "lower_bound": 4,
        "distance": None,
        "distinct_fault_effect_count": len(effects),
        "attempts": attempts,
        "witness": None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def certify_effect_signatures_through_four(
    effects: tuple[FaultEffect, ...],
    *,
    num_detectors: int,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Exactly search abstract detector/observable signatures through four."""

    callback = progress or (lambda _event: None)
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    for fault_count in range(1, 4):
        attempt_started = time.monotonic()
        indices = _find_low_cardinality_witness(
            effects,
            fault_count,
            heartbeat_seconds=heartbeat_seconds,
            callback=callback,
        )
        attempts.append(
            {
                "fault_count": fault_count,
                "method": "exact-direct-enumeration",
                "status": "sat" if indices is not None else "unsat",
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            }
        )
        if indices is not None:
            selected = [effects[index] for index in indices]
            detector_mask, observable_mask = _xor_masks(selected)
            return {
                "lower_bound": fault_count,
                "distance": fault_count,
                "attempts": attempts,
                "witness_indices": indices,
                "witness_observable_mask": observable_mask,
                "witness_detector_mask": detector_mask,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    attempt_started = time.monotonic()
    indices = find_four_fault_witness(
        effects,
        num_detectors=num_detectors,
        heartbeat_seconds=heartbeat_seconds,
        progress=callback,
    )
    attempts.append(
        {
            "fault_count": 4,
            "method": "exact-pair-meet-in-the-middle",
            "status": "sat" if indices is not None else "unsat",
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
        }
    )
    observable_mask = 0
    detector_mask = 0
    if indices is not None:
        detector_mask, observable_mask = _xor_masks(effects[index] for index in indices)
    return {
        "lower_bound": 4 if indices is not None else 5,
        "distance": 4 if indices is not None else None,
        "attempts": attempts,
        "witness_indices": indices,
        "witness_observable_mask": observable_mask if indices is not None else None,
        "witness_detector_mask": detector_mask if indices is not None else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def certify_through_four(
    circuit: stim.Circuit,
    *,
    heartbeat_seconds: float = 30.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Exactly find a logical mechanism of cardinality at most four."""

    callback = progress or (lambda _event: None)
    started = time.monotonic()
    dem = detector_error_model(circuit)
    effects = extract_fault_effects(dem)
    attempts: list[dict[str, Any]] = []
    for fault_count in range(1, 4):
        attempt_started = time.monotonic()
        indices = _find_low_cardinality_witness(
            effects,
            fault_count,
            heartbeat_seconds=heartbeat_seconds,
            callback=callback,
        )
        attempts.append(
            {
                "fault_count": fault_count,
                "method": "exact-direct-enumeration",
                "status": "sat" if indices is not None else "unsat",
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            }
        )
        if indices is not None:
            return {
                "lower_bound": fault_count,
                "distance": fault_count,
                "distinct_fault_effect_count": len(effects),
                "attempts": attempts,
                "witness": _witness_record(circuit, effects, indices, "exact-direct-enumeration"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    attempt_started = time.monotonic()
    indices = find_four_fault_witness(
        effects,
        num_detectors=circuit.num_detectors,
        heartbeat_seconds=heartbeat_seconds,
        progress=callback,
    )
    attempts.append(
        {
            "fault_count": 4,
            "method": "exact-pair-meet-in-the-middle",
            "status": "sat" if indices is not None else "unsat",
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
        }
    )
    return {
        "lower_bound": 4 if indices is not None else 5,
        "distance": 4 if indices is not None else None,
        "distinct_fault_effect_count": len(effects),
        "attempts": attempts,
        "witness": None
        if indices is None
        else _witness_record(circuit, effects, indices, "exact-pair-meet-in-the-middle"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
