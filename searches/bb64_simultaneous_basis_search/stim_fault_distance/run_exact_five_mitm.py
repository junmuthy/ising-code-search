#!/usr/bin/env python3
"""Exact anchored pair-plus-pair meet-in-the-middle test for five BB64 faults."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from codes.n32_k4_d6_reference_code.schedule_fault_search.screening import MASK64
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


Progress = Callable[[dict[str, Any]], None]
_WORKER_STATE: dict[str, Any] | None = None
_WORKER_STOP: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--experiment", choices=("bulk", "memory"), default="bulk")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--fault-model", choices=("cnot", "no-idle", "full"), default="full")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--batch-groups", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=1)
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


def pair_dtype(detector_chunks: int) -> np.dtype[Any]:
    return np.dtype(
        [
            (f"detector_{chunk}", np.uint64) for chunk in range(detector_chunks)
        ]
        + [("observable", np.uint64), ("left", np.uint16), ("right", np.uint16)]
    )


def key_dtype(detector_chunks: int) -> np.dtype[Any]:
    return np.dtype(
        [(f"detector_{chunk}", np.uint64) for chunk in range(detector_chunks)]
    )


def build_pair_table(
    effects: tuple[FaultEffect, ...],
    num_detectors: int,
    *,
    heartbeat_seconds: float,
    progress: Progress,
) -> tuple[np.ndarray[Any, Any], list[str]]:
    count = len(effects)
    if count > np.iinfo(np.uint16).max:
        raise ValueError("too many effects for uint16 pair indices")
    detector_chunks = max(1, (num_detectors + 63) // 64)
    fields = [f"detector_{chunk}" for chunk in range(detector_chunks)]
    pair_total = count * (count - 1) // 2
    pairs = np.empty(pair_total, dtype=pair_dtype(detector_chunks))
    position = 0
    last_checkpoint = time.monotonic()
    for left in range(count):
        amount = count - left - 1
        if not amount:
            continue
        stop = position + amount
        left_effect = effects[left]
        right_effects = effects[left + 1 :]
        for chunk, field in enumerate(fields):
            shift = 64 * chunk
            pairs[field][position:stop] = np.fromiter(
                (
                    ((left_effect.detector_mask ^ right.detector_mask) >> shift)
                    & MASK64
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
            progress(
                {
                    "event": "pair-build-heartbeat",
                    "pairs_built": position,
                    "pairs_total": pair_total,
                    "fraction_complete": round(position / pair_total, 6),
                }
            )
            last_checkpoint = now
    progress(
        {
            "event": "pair-sort-start",
            "pairs_total": pair_total,
            "table_bytes": int(pairs.nbytes),
        }
    )
    pairs.sort(order=fields)
    progress({"event": "pair-sort-finish", "pairs_total": pair_total})
    return pairs, fields


def key_equal(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any], fields: list[str]
) -> np.ndarray[Any, Any]:
    equal = np.ones(len(left), dtype=np.bool_)
    for field in fields:
        equal &= left[field] == right[field]
    return equal


def distinct_five(anchor: int, left: Any, right: Any) -> bool:
    return len(
        {
            anchor,
            int(left["left"]),
            int(left["right"]),
            int(right["left"]),
            int(right["right"]),
        }
    ) == 5


def search_group_blocks(
    pairs: np.ndarray[Any, Any],
    left_start: int,
    left_stop: int,
    right_start: int,
    right_stop: int,
    *,
    anchor: int,
    anchor_observable: int,
) -> list[int] | None:
    for left in pairs[left_start:left_stop]:
        for right in pairs[right_start:right_stop]:
            if (
                int(left["observable"])
                ^ int(right["observable"])
                ^ anchor_observable
            ) == 0:
                continue
            if distinct_five(anchor, left, right):
                return [
                    anchor,
                    int(left["left"]),
                    int(left["right"]),
                    int(right["left"]),
                    int(right["right"]),
                ]
    return None


def search_one_anchor(
    anchor: int,
    effect: FaultEffect,
    *,
    pairs: np.ndarray[Any, Any],
    fields: list[str],
    starts: np.ndarray[Any, Any],
    stops: np.ndarray[Any, Any],
    keys: np.ndarray[Any, Any],
    singleton: np.ndarray[Any, Any],
    batch_groups: int,
    stop: threading.Event,
) -> tuple[list[int] | None, int]:
    batches_tested = 0
    for batch_start in range(0, len(starts), batch_groups):
        if stop.is_set():
            return None, batches_tested
        batch_stop = min(batch_start + batch_groups, len(starts))
        queries = keys[batch_start:batch_stop].copy()
        for chunk, field in enumerate(fields):
            queries[field] ^= np.uint64(
                (effect.detector_mask >> (64 * chunk)) & MASK64
            )
        locations = np.searchsorted(keys, queries)
        in_range = locations < len(keys)
        candidate_offsets = np.flatnonzero(in_range)
        if len(candidate_offsets):
            candidate_offsets = candidate_offsets[
                key_equal(
                    keys[locations[candidate_offsets]],
                    queries[candidate_offsets],
                    fields,
                )
            ]
        if len(candidate_offsets):
            left_groups = batch_start + candidate_offsets
            right_groups = locations[candidate_offsets]
            both_single = singleton[left_groups] & singleton[right_groups]
            if np.any(both_single):
                left_records = pairs[starts[left_groups[both_single]]]
                right_records = pairs[starts[right_groups[both_single]]]
                nonzero_logical = (
                    left_records["observable"]
                    ^ right_records["observable"]
                    ^ np.uint64(effect.observable_mask)
                ) != 0
                distinct = (
                    (left_records["left"] != anchor)
                    & (left_records["right"] != anchor)
                    & (right_records["left"] != anchor)
                    & (right_records["right"] != anchor)
                    & (left_records["left"] != right_records["left"])
                    & (left_records["left"] != right_records["right"])
                    & (left_records["right"] != right_records["left"])
                    & (left_records["right"] != right_records["right"])
                )
                matches = np.flatnonzero(nonzero_logical & distinct)
                if len(matches):
                    index = int(matches[0])
                    left = left_records[index]
                    right = right_records[index]
                    return [
                        anchor,
                        int(left["left"]),
                        int(left["right"]),
                        int(right["left"]),
                        int(right["right"]),
                    ], batches_tested + 1
            for left_group, right_group in zip(
                left_groups[~both_single], right_groups[~both_single], strict=True
            ):
                witness = search_group_blocks(
                    pairs,
                    int(starts[left_group]),
                    int(stops[left_group]),
                    int(starts[right_group]),
                    int(stops[right_group]),
                    anchor=anchor,
                    anchor_observable=effect.observable_mask,
                )
                if witness is not None:
                    return witness, batches_tested + 1
        batches_tested += 1
    return None, batches_tested


def search_anchor_worker(
    number_and_anchor: tuple[int, int]
) -> tuple[int, list[int] | None, int]:
    if _WORKER_STATE is None or _WORKER_STOP is None:
        raise RuntimeError("exact-five worker state was not initialized before fork")
    anchor_number, anchor = number_and_anchor
    effects = _WORKER_STATE["effects"]
    witness, batches = search_one_anchor(
        anchor,
        effects[anchor],
        pairs=_WORKER_STATE["pairs"],
        fields=_WORKER_STATE["fields"],
        starts=_WORKER_STATE["starts"],
        stops=_WORKER_STATE["stops"],
        keys=_WORKER_STATE["keys"],
        singleton=_WORKER_STATE["singleton"],
        batch_groups=_WORKER_STATE["batch_groups"],
        stop=_WORKER_STOP,
    )
    return anchor_number, witness, batches


def find_five_fault_witness(
    effects: tuple[FaultEffect, ...],
    anchors: list[int],
    *,
    num_detectors: int,
    batch_groups: int,
    heartbeat_seconds: float,
    progress: Progress,
    workers: int = 1,
) -> tuple[list[int] | None, dict[str, int]]:
    pairs, fields = build_pair_table(
        effects,
        num_detectors,
        heartbeat_seconds=heartbeat_seconds,
        progress=progress,
    )
    same_as_previous = np.ones(len(pairs), dtype=np.bool_)
    same_as_previous[0] = False
    for field in fields:
        same_as_previous[1:] &= pairs[field][1:] == pairs[field][:-1]
    starts = np.flatnonzero(~same_as_previous)
    stops = np.append(starts[1:], len(pairs))
    keys = np.empty(len(starts), dtype=key_dtype(len(fields)))
    for field in fields:
        keys[field] = pairs[field][starts]
    singleton = stops - starts == 1
    progress(
        {
            "event": "pair-groups-ready",
            "pair_groups": len(starts),
            "singleton_pair_groups": int(np.count_nonzero(singleton)),
            "translation_anchors": len(anchors),
        }
    )
    del heartbeat_seconds
    batches_tested = 0
    anchors_tested = 0
    if workers == 1:
        stop: Any = threading.Event()

        def search(
            number_and_anchor: tuple[int, int]
        ) -> tuple[int, list[int] | None, int]:
            anchor_number, anchor = number_and_anchor
            witness, batches = search_one_anchor(
                anchor,
                effects[anchor],
                pairs=pairs,
                fields=fields,
                starts=starts,
                stops=stops,
                keys=keys,
                singleton=singleton,
                batch_groups=batch_groups,
                stop=stop,
            )
            return anchor_number, witness, batches

        executor: concurrent.futures.Executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        submit = search
    else:
        context = mp.get_context("fork")
        stop = context.Event()
        global _WORKER_STATE, _WORKER_STOP
        _WORKER_STATE = {
            "effects": effects,
            "pairs": pairs,
            "fields": fields,
            "starts": starts,
            "stops": stops,
            "keys": keys,
            "singleton": singleton,
            "batch_groups": batch_groups,
        }
        _WORKER_STOP = stop
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        )
        submit = search_anchor_worker

    with executor:
        futures = [executor.submit(submit, item) for item in enumerate(anchors, start=1)]
        for future in concurrent.futures.as_completed(futures):
            anchor_number, witness, batches = future.result()
            anchors_tested += 1
            batches_tested += batches
            progress(
                {
                    "event": "five-anchor-finish",
                    "anchor": anchor_number,
                    "anchors_completed": anchors_tested,
                    "anchors_total": len(anchors),
                    "batches_tested": batches_tested,
                    "workers": workers,
                }
            )
            if witness is not None:
                stop.set()
                _WORKER_STATE = None
                _WORKER_STOP = None
                return witness, {
                    "pair_count": len(pairs),
                    "pair_groups": len(starts),
                    "anchors_tested": anchors_tested,
                    "batches_tested": batches_tested,
                }
    _WORKER_STATE = None
    _WORKER_STOP = None
    return None, {
        "pair_count": len(pairs),
        "pair_groups": len(starts),
        "anchors_tested": anchors_tested,
        "batches_tested": batches_tested,
    }


def validate_witness(
    circuit: Any, effects: tuple[FaultEffect, ...], indices: list[int]
) -> dict[str, Any]:
    detector_mask = 0
    observable_mask = 0
    for index in indices:
        detector_mask ^= effects[index].detector_mask
        observable_mask ^= effects[index].observable_mask
    if detector_mask or not observable_mask or len(set(indices)) != 5:
        raise RuntimeError("invalid five-fault meet-in-the-middle witness")
    return {
        "fault_count": 5,
        "effect_indices": indices,
        "detector_mask": detector_mask,
        "observable_mask": observable_mask,
        "observable_indices": [
            index
            for index in range(circuit.num_observables)
            if observable_mask >> index & 1
        ],
        "effects": [
            {
                "dem_instruction_index": effects[index].dem_instruction_index,
                "detector_mask": effects[index].detector_mask,
                "observable_mask": effects[index].observable_mask,
                "representative_circuit_error": explain_effect(circuit, effects[index]),
            }
            for index in indices
        ],
        "validated_zero_detector_nonzero_observable": True,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.batch_groups < 1:
        raise SystemExit("--batch-groups must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    configuration = {
        "schema_version": 1,
        "schedule": str(args.schedule.resolve()),
        "experiment": args.experiment,
        "rounds": args.rounds,
        "bases": list(bases),
        "fault_model": args.fault_model,
        "probability": args.probability,
        "batch_groups": args.batch_groups,
        "workers": args.workers,
        "method": "translation-anchored pair-plus-pair exact five-fault search",
        "required_prior_lower_bound": 5,
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
        effects = raw_dem_effects(detector_error_model(circuit))
        anchor_variables = translation_anchor_variables(circuit, effects)
        anchors = [variable - 1 for variable in anchor_variables]
        started = time.monotonic()

        def progress(event: dict[str, Any]) -> None:
            state["active"] = {"basis": basis, **event}
            state["updated"] = utc_now()
            atomic_json(checkpoint, state)
            fields = " ".join(f"{key}={value}" for key, value in event.items())
            print(f"checkpoint {utc_now()} basis={basis} {fields}", flush=True)

        print(
            f"checkpoint {utc_now()} basis={basis} phase=exact-five-start "
            f"effects={len(effects)} anchors={len(anchors)} detectors={circuit.num_detectors}",
            flush=True,
        )
        indices, statistics = find_five_fault_witness(
            effects,
            anchors,
            num_detectors=circuit.num_detectors,
            batch_groups=args.batch_groups,
            heartbeat_seconds=args.heartbeat_seconds,
            progress=progress,
            workers=args.workers,
        )
        record = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "status": "sat" if indices is not None else "unsat",
            "tested_fault_count": 5,
            "distance": 5 if indices is not None else None,
            "lower_bound": 5 if indices is not None else 6,
            "statistics": statistics,
            "witness": None
            if indices is None
            else validate_witness(circuit, effects, indices),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        state["results"][basis] = record
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)
        print(
            f"checkpoint {utc_now()} basis={basis} phase=exact-five-finish "
            f"status={record['status']} lower_bound={record['lower_bound']} "
            f"distance={record['distance']} elapsed_seconds={record['elapsed_seconds']}",
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
