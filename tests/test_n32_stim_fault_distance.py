"""Tests for the [[32,4,6]] Stim circuit-fault implementation."""

from __future__ import annotations

import numpy as np
import pytest

from codes.n32_k4_d6_reference_code.stim_fault_distance.circuit import (
    NoiseModel,
    build_bulk_circuit,
    build_memory_circuit,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    certify_fault_distance,
    detector_error_model,
    extract_fault_effects,
)
from codes.n32_k4_d6_reference_code.stim_fault_distance.model import load_code_data


def test_saved_inputs_have_expected_shape() -> None:
    code = load_code_data()
    assert len(code.checks_x) == len(code.checks_z) == 16
    assert {len(row) for row in code.checks_x + code.checks_z} == {8}
    assert len(code.logicals_x) == len(code.logicals_z) == 4
    assert {len(row) for row in code.logicals_x + code.logicals_z} == {7}
    assert len(code.schedule) == 12
    assert sum(map(len, code.schedule)) == 256


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_memory_has_deterministic_detectors_and_logicals(basis: str) -> None:
    circuit = build_memory_circuit(basis, 3)
    detectors, observables = circuit.compile_detector_sampler().sample(
        shots=64,
        separate_observables=True,
    )
    assert circuit.num_qubits == 64
    assert circuit.num_observables == 4
    assert not np.any(detectors)
    assert not np.any(observables)
    circuit.detector_error_model(allow_gauge_detectors=False)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noisy_bulk_circuit_has_hypergraph_fault_effects(basis: str) -> None:
    circuit = build_bulk_circuit(basis, noise=NoiseModel.cnot_only())
    dem = detector_error_model(circuit)
    effects = extract_fault_effects(dem)
    assert dem.num_errors > 0
    assert effects
    assert any(effect.observable_mask for effect in effects)


def test_folded_bases_have_matching_circuit_dimensions() -> None:
    circuit_x = build_memory_circuit("X", 2, noise=NoiseModel.without_idles())
    circuit_z = build_memory_circuit("Z", 2, noise=NoiseModel.without_idles())
    assert circuit_x.num_qubits == circuit_z.num_qubits
    assert circuit_x.num_detectors == circuit_z.num_detectors
    assert circuit_x.num_measurements == circuit_z.num_measurements
    assert circuit_x.num_observables == circuit_z.num_observables == 4


def test_bulk_cnot_fault_distance_is_exactly_four() -> None:
    circuit = build_bulk_circuit("X", noise=NoiseModel.cnot_only())
    result = certify_fault_distance(circuit, max_faults=4, heartbeat_seconds=60)
    assert result["distance"] == 4
    assert result["lower_bound"] == 4
    assert result["heuristic_upper_bound"] == 4
    assert result["exact_within_tested_range"]
    assert [attempt["status"] for attempt in result["attempts"]] == [
        "unsat",
        "unsat",
        "unsat",
    ]
