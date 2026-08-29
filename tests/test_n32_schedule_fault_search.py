"""Tests for the alternative 12-layer schedule fault search."""

from __future__ import annotations

import json
import itertools
import random
from pathlib import Path

import numpy as np

from n32_k4_d6_reference_code.schedule_fault_search.data import code_for_schedule
from n32_k4_d6_reference_code.schedule_fault_search.materialize_survivor import (
    DEFAULT_RECORD,
)
from n32_k4_d6_reference_code.schedule_fault_search.scheduler import (
    ScheduleEnumerator,
    build_schedule_record,
    matrix_from_supports,
    reverse_colors,
)
from n32_k4_d6_reference_code.schedule_fault_search.screening import (
    find_four_fault_witness,
)
from n32_k4_d6_reference_code.stim_fault_distance.circuit import build_memory_circuit
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import FaultEffect


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "n32_k4_d6_reference_code"
PRESENTATION = (
    REFERENCE / "schedule" / "all_weight8_translation_symmetric_v1" / "presentation.json"
)
FOLLOWUPS = REFERENCE / "followups" / "fold_rescan_and_second_translation_v1"


def test_versioned_survivor_materializes_the_saved_schedule() -> None:
    compact = json.loads(DEFAULT_RECORD.read_text(encoding="utf-8"))
    presentation, _folds, _reference = load_inputs()
    checks_x = matrix_from_supports(presentation["checks_x"])
    enumerator = ScheduleEnumerator(
        checks_x,
        compact["fold"]["permutation"],
        presentation["translation_action_on_x_checks"],
    )
    fold = dict(compact["fold"])
    fold["pairing"] = fold.pop("logical_pairing")
    schedule = build_schedule_record(
        checks_x,
        fold["permutation"],
        enumerator.edge_orbit,
        compact["edge_orbit_colors_zero_based"],
        depth=compact["cnot_depth"],
        fold_index=compact["fold_index"],
        fold=fold,
    )
    assert schedule["schedule_id"] == compact["schedule_id"]
    assert schedule["cnot_count"] == 256
    code = code_for_schedule(schedule)
    assert code.num_logicals == 4
    assert len(code.fold_permutation) == 32


def load_inputs() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    presentation = json.loads(PRESENTATION.read_text(encoding="utf-8"))
    folds = [
        json.loads(line)
        for line in (FOLLOWUPS / "distance-six-folds.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    reference = json.loads(
        (FOLLOWUPS / "optimal-clean-joint-12-layer.json").read_text(encoding="utf-8")
    )
    return presentation, folds, reference


def test_enumerator_finds_a_distinct_valid_depth_twelve_schedule() -> None:
    presentation, folds, reference = load_inputs()
    checks_x = matrix_from_supports(presentation["checks_x"])
    enumerator = ScheduleEnumerator(
        checks_x,
        folds[0]["permutation"],
        presentation["translation_action_on_x_checks"],
        timeout_seconds=10,
        minimum_color_changes=4,
    )
    enumerator.exclude(reference["edge_orbit_colors_zero_based"])
    result = enumerator.next()
    assert result.status == "sat"
    assert result.colors is not None
    reference_colors = tuple(reference["edge_orbit_colors_zero_based"])
    assert sum(left != right for left, right in zip(result.colors, reference_colors, strict=True)) >= 4
    assert sum(
        left != right
        for left, right in zip(result.colors, reverse_colors(reference_colors, 12), strict=True)
    ) >= 4
    schedule = build_schedule_record(
        checks_x,
        folds[0]["permutation"],
        enumerator.edge_orbit,
        result.colors,
        depth=12,
        fold_index=0,
        fold=folds[0],
    )
    assert schedule["cnot_depth"] == 12
    assert schedule["cnot_count"] == 256
    assert all(len(layer["gates"]) <= 32 for layer in schedule["layers"])


def test_candidate_fold_logicals_build_deterministic_memory_circuits() -> None:
    presentation, folds, reference = load_inputs()
    code = code_for_schedule(reference)
    for basis in ("X", "Z"):
        circuit = build_memory_circuit(basis, 2, code=code)
        detectors, observables = circuit.compile_detector_sampler().sample(
            shots=32, separate_observables=True
        )
        assert not np.any(detectors)
        assert not np.any(observables)


def test_four_fault_meet_in_the_middle_finds_exact_disjoint_witness() -> None:
    effects = tuple(
        FaultEffect(detector, observable, 1e-3, index)
        for index, (detector, observable) in enumerate(
            [(0b001, 0), (0b010, 0), (0b100, 0), (0b111, 1)]
        )
    )
    witness = find_four_fault_witness(effects, num_detectors=3)
    assert witness is not None
    assert sorted(witness) == [0, 1, 2, 3]


def test_four_fault_meet_in_the_middle_proves_small_negative_instance() -> None:
    effects = tuple(
        FaultEffect(detector, 0, 1e-3, index)
        for index, detector in enumerate((0b001, 0b010, 0b100, 0b111))
    )
    assert find_four_fault_witness(effects, num_detectors=3) is None


def test_four_fault_meet_in_the_middle_matches_brute_force() -> None:
    generator = random.Random(260828)
    for _ in range(30):
        signatures: dict[tuple[int, int], FaultEffect] = {}
        while len(signatures) < 9:
            detector = generator.randrange(1, 1 << 7)
            observable = generator.randrange(4)
            key = (detector, observable)
            signatures[key] = FaultEffect(detector, observable, 1e-3, len(signatures))
        effects = tuple(signatures.values())
        brute = next(
            (
                indices
                for indices in itertools.combinations(range(len(effects)), 4)
                if not (
                    effects[indices[0]].detector_mask
                    ^ effects[indices[1]].detector_mask
                    ^ effects[indices[2]].detector_mask
                    ^ effects[indices[3]].detector_mask
                )
                and (
                    effects[indices[0]].observable_mask
                    ^ effects[indices[1]].observable_mask
                    ^ effects[indices[2]].observable_mask
                    ^ effects[indices[3]].observable_mask
                )
            ),
            None,
        )
        exact = find_four_fault_witness(effects, num_detectors=7)
        assert (exact is not None) == (brute is not None)
