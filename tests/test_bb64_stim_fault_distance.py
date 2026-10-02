"""Regression tests for the [[64,8,8]] simultaneous-basis Stim pipeline."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from searches.bb64_simultaneous_basis_search.schedule_fault_search.enumerate_schedules import (
    build_edge_orbits,
    materialize_schedule,
)
from searches.bb64_simultaneous_basis_search.stim_fault_distance.circuit import (
    NoiseModel,
    build_bulk_circuit,
    build_memory_circuit,
)
from searches.bb64_simultaneous_basis_search.stim_fault_distance.effects import (
    raw_dem_effects,
    translation_anchor_variables,
)
from searches.bb64_simultaneous_basis_search.stim_fault_distance.model import (
    BASIS_PATH,
    DEFAULT_SCHEDULE_PATH,
    load_code_data,
)
from searches.bb64_simultaneous_basis_search.stim_fault_distance.run_exact_five_mitm import (
    find_five_fault_witness,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    FaultEffect,
    detector_error_model,
    extract_fault_effects,
)


def test_saved_inputs_have_expected_shape_and_optimal_depth() -> None:
    code = load_code_data()
    assert len(code.checks_x) == len(code.checks_z) == 32
    assert {len(row) for row in code.checks_x + code.checks_z} == {8}
    assert len(code.logicals_x) == len(code.logicals_z) == 8
    assert {len(row) for row in code.logicals_x + code.logicals_z} == {8}
    assert len(code.redundant_relations) == 4
    assert len(code.schedule) == 8
    assert all(len(layer) == 64 for layer in code.schedule)
    assert sum(map(len, code.schedule)) == 512


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_memory_has_deterministic_detectors_and_logicals(basis: str) -> None:
    circuit = build_memory_circuit(basis, 3)
    detectors, observables = circuit.compile_detector_sampler().sample(
        shots=32,
        separate_observables=True,
    )
    assert circuit.num_qubits == 128
    assert circuit.num_detectors == 216
    assert circuit.num_measurements == 256
    assert circuit.num_observables == 8
    assert not np.any(detectors)
    assert not np.any(observables)
    circuit.detector_error_model(allow_gauge_detectors=False)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noisy_bulk_circuit_has_expected_fault_signatures(basis: str) -> None:
    circuit = build_bulk_circuit(basis, noise=NoiseModel.cnot_only())
    dem = detector_error_model(circuit)
    effects = extract_fault_effects(dem)
    assert dem.num_errors == 3968
    assert len(effects) == 3968
    assert any(effect.observable_mask for effect in effects)


def test_baseline_schedule_reconstructs_from_its_orbit_colors() -> None:
    archive = np.load(BASIS_PATH)
    checks = np.asarray(archive["matrix_x"], dtype=np.uint8)
    _orbits, edge_orbit = build_edge_orbits(checks)
    saved = json.loads(DEFAULT_SCHEDULE_PATH.read_text(encoding="utf-8"))
    rebuilt = materialize_schedule(
        checks,
        edge_orbit,
        tuple(int(color) for color in saved["edge_orbit_colors_zero_based"]),
    )
    assert rebuilt is not None
    assert rebuilt["schedule_id"] == saved["schedule_id"]
    assert rebuilt["cnot_depth"] == 8
    assert rebuilt["idle_free_entangling_layers"]
    assert rebuilt["clean_cross_ancilla_backaction"]


def test_translation_anchor_includes_all_merged_cell_zero_locations() -> None:
    circuit = build_bulk_circuit("X", noise=NoiseModel.cnot_only())
    effects = raw_dem_effects(detector_error_model(circuit))
    anchors = translation_anchor_variables(circuit, effects)
    assert len(anchors) == 98
    assert all(effects[variable - 1].observable_mask for variable in anchors)


def brute_five(effects: tuple[FaultEffect, ...]) -> tuple[int, ...] | None:
    return next(
        (
            indices
            for indices in itertools.combinations(range(len(effects)), 5)
            if not _xor(effects[index].detector_mask for index in indices)
            and _xor(effects[index].observable_mask for index in indices)
        ),
        None,
    )


def _xor(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result ^= int(value)
    return result


def test_exact_five_meet_in_the_middle_matches_brute_force() -> None:
    generator = random.Random(260901)
    for _ in range(12):
        effects = tuple(
            FaultEffect(
                detector_mask=generator.randrange(1, 1 << 7),
                observable_mask=generator.randrange(4),
                probability=1e-3,
                dem_instruction_index=index,
            )
            for index in range(9)
        )
        anchors = [
            index for index, effect in enumerate(effects) if effect.observable_mask
        ]
        brute = brute_five(effects)
        exact, _statistics = find_five_fault_witness(
            effects,
            anchors,
            num_detectors=7,
            batch_groups=8,
            heartbeat_seconds=60,
            progress=lambda _event: None,
            workers=2,
        )
        assert (exact is not None) == (brute is not None)
        if exact is not None:
            assert len(set(exact)) == 5
            assert not _xor(effects[index].detector_mask for index in exact)
            assert _xor(effects[index].observable_mask for index in exact)
