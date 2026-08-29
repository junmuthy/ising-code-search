#!/usr/bin/env python3
"""Exact low-cardinality certificate for a long memory via temporal windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from n32_k4_d6_reference_code.schedule_fault_search.data import code_for_schedule
from n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_effect_signatures_through_four,
)
from n32_k4_d6_reference_code.stim_fault_distance.circuit import (
    NoiseModel,
    build_memory_circuit,
)
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    FaultEffect,
    atomic_json,
    detector_error_model,
    extract_fault_effects,
    utc_now,
)


NormalizedCoordinate = tuple[float | int, ...]
NormalizedSignature = tuple[tuple[NormalizedCoordinate, ...], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--three-round-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--basis", choices=("X", "Z", "both"), default="both")
    parser.add_argument("--probability", type=float, default=1e-3)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def detector_times(
    coordinates: dict[int, list[float]], num_detectors: int
) -> tuple[int, ...]:
    times: list[int] = []
    full_coordinates: list[tuple[float, ...]] = []
    for detector in range(num_detectors):
        coordinate = tuple(float(value) for value in coordinates[detector])
        if not coordinate:
            raise ValueError(f"detector {detector} has no temporal coordinate")
        rounded = round(coordinate[0])
        if not math.isclose(coordinate[0], rounded):
            raise ValueError(f"detector {detector} has nonintegral time {coordinate[0]}")
        times.append(int(rounded))
        full_coordinates.append(coordinate)
    if len(set(full_coordinates)) != len(full_coordinates):
        raise ValueError("detector coordinates are not unique")
    return tuple(times)


def effect_descriptors(
    effects: tuple[FaultEffect, ...],
    times: tuple[int, ...],
) -> tuple[tuple[FaultEffect, tuple[int, ...], int, int], ...]:
    output = []
    for effect in effects:
        detectors = tuple(
            detector
            for detector in range(len(times))
            if effect.detector_mask >> detector & 1
        )
        if not detectors:
            raise ValueError(
                "detectorless elementary logical effect prevents a temporal lower bound"
            )
        support_times = [times[detector] for detector in detectors]
        output.append((effect, detectors, min(support_times), max(support_times)))
    return tuple(output)


def normalized_window_signatures(
    descriptors: Iterable[tuple[FaultEffect, tuple[int, ...], int, int]],
    coordinates: dict[int, list[float]],
    *,
    start: int,
    width: int = 2,
) -> tuple[NormalizedSignature, ...]:
    signatures: set[NormalizedSignature] = set()
    for effect, detectors, minimum, maximum in descriptors:
        if minimum < start or maximum > start + width:
            continue
        normalized = tuple(
            sorted(
                (
                    int(round(coordinates[detector][0])) - start,
                    *(float(value) for value in coordinates[detector][1:]),
                )
                for detector in detectors
            )
        )
        signatures.add((normalized, int(effect.observable_mask)))
    return tuple(sorted(signatures))


def effects_from_normalized_signatures(
    signatures: tuple[NormalizedSignature, ...]
) -> tuple[tuple[FaultEffect, ...], int]:
    coordinates = sorted(
        {coordinate for detector_support, _observable in signatures for coordinate in detector_support}
    )
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    effects = []
    for index, (detector_support, observable_mask) in enumerate(signatures):
        detector_mask = sum(1 << coordinate_index[coordinate] for coordinate in detector_support)
        effects.append(FaultEffect(detector_mask, observable_mask, 1.0, index))
    return tuple(effects), len(coordinates)


def signature_digest(signatures: tuple[NormalizedSignature, ...]) -> str:
    return hashlib.sha256(repr(signatures).encode()).hexdigest()[:16]


def load_prefix_upper_bound(records_path: Path, schedule_id: str) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    record = next(
        (item for item in records if str(item.get("schedule_id")) == schedule_id),
        None,
    )
    if record is None:
        raise ValueError(f"schedule {schedule_id} is absent from {records_path}")
    bases: dict[str, Any] = {}
    for basis in ("X", "Z"):
        stage = record["fault_screen"][f"memory-r3-full-{basis}"]
        heuristic = stage["heuristic"]
        if int(heuristic["upper_bound"]) != 5:
            raise ValueError(f"three-round {basis} witness is not cardinality five")
        ticks = [
            int(location["tick_count_before_location"])
            for location in heuristic["fault_locations"]
            if location["tick_count_before_location"] is not None
        ]
        if len(ticks) != 5 or max(ticks) > 13:
            raise ValueError(
                f"three-round {basis} witness is not confined to preparation/round zero"
            )
        bases[basis] = {
            "fault_count": 5,
            "tick_counts": ticks,
            "all_faults_in_preparation_or_first_syndrome_round": True,
        }
    return {
        "source": str(records_path.resolve()),
        "reason_extension_is_valid": (
            "the concrete five-fault mechanism occurs entirely in preparation and the first "
            "syndrome round; after it produces a zero-syndrome logical Pauli, appending ideal "
            "syndrome rounds preserves zero detectors and the logical observable"
        ),
        "bases": bases,
    }


def main() -> None:
    args = parse_args()
    if args.rounds < 4:
        raise SystemExit("at least four rounds are required for the temporal certificate")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    checkpoint = args.output.with_suffix(".checkpoint.json")
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    schedule_id = str(schedule["schedule_id"])
    bases = ("X", "Z") if args.basis == "both" else (args.basis,)
    configuration = {
        "schedule": str(args.schedule.resolve()),
        "schedule_id": schedule_id,
        "three_round_records": str(args.three_round_records.resolve()),
        "rounds": args.rounds,
        "bases": list(bases),
        "probability": args.probability,
        "heartbeat_seconds": args.heartbeat_seconds,
        "maximum_fault_cardinality": 4,
        "temporal_window_width": 2,
    }
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["configuration"] != configuration:
            raise SystemExit("resume configuration mismatch")
        state["status"] = "running"
    else:
        state = {
            "schema_version": 1,
            "started": utc_now(),
            "updated": utc_now(),
            "status": "running",
            "configuration": configuration,
            "proof": {
                "individual_effect_temporal_span_bound": 1,
                "minimal_witness_connectedness": (
                    "a disconnected zero-detector witness has a connected component with "
                    "zero detectors and nonzero observable, contradicting minimality"
                ),
                "endpoint_parity_window_bound": (
                    "at least two effects touch each temporal endpoint; with at most four "
                    "effects and individual span at most one, endpoint separation of three "
                    "or more exhausts the effects into two disconnected endpoint groups"
                ),
                "conclusion": (
                    "every minimal logical witness of at most four faults is contained in "
                    "three adjacent detector-time slices"
                ),
            },
            "prefix_five_fault_upper_bound": load_prefix_upper_bound(
                args.three_round_records, schedule_id
            ),
            "results": {},
            "active": None,
        }
    atomic_json(checkpoint, state)
    code = code_for_schedule(schedule)
    noise = NoiseModel(args.probability)
    started = time.monotonic()

    for basis in bases:
        if basis in state["results"]:
            continue
        print(
            f"checkpoint {utc_now()} phase=temporal-build basis={basis} rounds={args.rounds}",
            flush=True,
        )
        circuit = build_memory_circuit(basis, args.rounds, noise=noise, code=code)
        coordinates = circuit.get_detector_coordinates()
        times = detector_times(coordinates, circuit.num_detectors)
        effects = extract_fault_effects(detector_error_model(circuit))
        descriptors = effect_descriptors(effects, times)
        maximum_span = max(maximum - minimum for _effect, _ids, minimum, maximum in descriptors)
        if maximum_span > 1:
            raise RuntimeError(f"individual effect temporal span is {maximum_span}, not at most one")

        classes: dict[tuple[NormalizedSignature, ...], list[int]] = defaultdict(list)
        for window_start in range(min(times), max(times) - 1):
            signatures = normalized_window_signatures(
                descriptors,
                coordinates,
                start=window_start,
                width=2,
            )
            classes[signatures].append(window_start)
        ordered_classes = sorted(classes.items(), key=lambda item: item[1][0])
        basis_result: dict[str, Any] = {
            "num_detectors": circuit.num_detectors,
            "distinct_fault_effects": len(effects),
            "maximum_individual_temporal_span": maximum_span,
            "window_count": sum(len(windows) for _signatures, windows in ordered_classes),
            "window_class_count": len(ordered_classes),
            "classes": [],
        }
        for class_index, (signatures, windows) in enumerate(ordered_classes):
            abstract_effects, detector_count = effects_from_normalized_signatures(signatures)
            digest = signature_digest(signatures)

            def progress(event: dict[str, Any]) -> None:
                state["updated"] = utc_now()
                state["active"] = {
                    "basis": basis,
                    "class_index": class_index,
                    "class_count": len(ordered_classes),
                    "windows": windows,
                    "signature_digest": digest,
                    **event,
                }
                atomic_json(checkpoint, state)
                fields = " ".join(
                    f"{key}={value}" for key, value in event.items() if key != "timestamp"
                )
                print(
                    f"checkpoint {event.get('timestamp', utc_now())} phase=temporal-exact "
                    f"basis={basis} class={class_index + 1}/{len(ordered_classes)} "
                    f"windows={windows} {fields}",
                    flush=True,
                )

            print(
                f"checkpoint {utc_now()} phase=temporal-class basis={basis} "
                f"class={class_index + 1}/{len(ordered_classes)} windows={windows} "
                f"effects={len(abstract_effects)} detectors={detector_count}",
                flush=True,
            )
            certificate = certify_effect_signatures_through_four(
                abstract_effects,
                num_detectors=detector_count,
                heartbeat_seconds=args.heartbeat_seconds,
                progress=progress,
            )
            class_result = {
                "class_index": class_index,
                "windows": windows,
                "signature_digest": digest,
                "distinct_normalized_effects": len(abstract_effects),
                "local_detector_count": detector_count,
                "certificate": certificate,
            }
            basis_result["classes"].append(class_result)
            state["active"] = None
            state["updated"] = utc_now()
            state["partial_basis"] = {basis: basis_result}
            atomic_json(checkpoint, state)
            if int(certificate["lower_bound"]) < 5:
                break
        basis_result["all_window_classes_exclude_four_fault_logicals"] = all(
            int(item["certificate"]["lower_bound"]) >= 5
            for item in basis_result["classes"]
        ) and len(basis_result["classes"]) == len(ordered_classes)
        basis_result["lower_bound"] = (
            5 if basis_result["all_window_classes_exclude_four_fault_logicals"] else None
        )
        basis_result["upper_bound"] = 5
        basis_result["distance"] = (
            5 if basis_result["all_window_classes_exclude_four_fault_logicals"] else None
        )
        state["results"][basis] = basis_result
        state.pop("partial_basis", None)
        state["active"] = None
        state["updated"] = utc_now()
        atomic_json(checkpoint, state)

    state["status"] = "complete"
    state["completed"] = utc_now()
    state["updated"] = utc_now()
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    state["all_required_checks_pass"] = all(
        state["results"].get(basis, {}).get("distance") == 5 for basis in bases
    )
    state["active"] = None
    atomic_json(args.output, state)
    atomic_json(checkpoint, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    if not state["all_required_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
