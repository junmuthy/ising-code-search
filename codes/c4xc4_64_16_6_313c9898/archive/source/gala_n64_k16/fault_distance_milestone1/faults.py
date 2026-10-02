"""Reuse exact fault signatures; independently replay physical Pauli witnesses."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import stim

from model import GALA, ROOT, digest, load, save
from circuit import NOISE, build

sys.path.append(str(GALA))
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import (
    detector_error_model, extract_fault_effects,
)
from n32_k4_d6_reference_code.schedule_fault_search.screening import (
    certify_through_three, certify_through_four, estimate_four_fault_table,
)


def inventory():
    files = [
        GALA / 'n32_k4_d6_reference_code/stim_fault_distance/fault_distance.py',
        GALA / 'n32_k4_d6_reference_code/schedule_fault_search/screening.py',
        *sorted(ROOT.glob('*.py')),
    ]
    return dict(source_sha256={str(path): digest(path.read_bytes()) for path in files},
                versions={name: metadata.version(name) for name in ('stim', 'numpy', 'z3-solver')})


def masks(error):
    d = o = 0
    for item in error.dem_error_terms:
        t = item.dem_target
        if t.is_relative_detector_id():
            d ^= 1 << t.val
        elif t.is_logical_observable_id():
            o ^= 1 << t.val
        else:
            raise ValueError(f'Unexpected DEM target {t}')
    return d, o


def explain_masks(circuit, d, o):
    targets = [f'D{i}' for i in range(d.bit_length()) if d >> i & 1]
    targets += [f'L{i}' for i in range(o.bit_length()) if o >> i & 1]
    filtered = stim.DetectorErrorModel('error(1) ' + ' '.join(targets))
    errors = circuit.explain_detector_error_model_errors(dem_filter=filtered, reduce_to_one_representative_error=True)
    assert len(errors) == 1 and masks(errors[0]) == (d, o)
    return errors[0]


def replay(circuit, errors, *, require_logical=True, shots=16):
    """Compare a faulted Clifford reference against the *original* reference.

    Using the faulted circuit's own detector sampler alone would hide
    deterministic inserted Paulis in its reference. The original converter
    is essential. Exact gauge rejection establishes determinism separately.
    """
    assert circuit == circuit.flattened()
    channels = {}
    expected_d = expected_o = 0
    raw_locations = []
    for error in errors:
        d, o = masks(error)
        expected_d ^= d
        expected_o ^= o
        assert error.circuit_error_locations
        location = error.circuit_error_locations[0]
        assert location.flipped_measurement is None and len(location.stack_frames) == 1
        offset = location.stack_frames[0].instruction_offset
        target = location.instruction_targets
        instruction = circuit[offset]
        assert instruction.name == target.gate and instruction.name in NOISE
        assert instruction.gate_args_copy()[0] > 0
        key = (offset, target.target_range_start, target.target_range_end)
        record = channels.setdefault(key, dict(instruction_offset=offset, gate=target.gate,
                                               target_range=[target.target_range_start, target.target_range_end], paulis={}))
        allowed_qubits = {t.gate_target.value for t in target.targets_in_range}
        for item in location.flipped_pauli_product:
            pauli = item.gate_target
            q = pauli.value
            assert q in allowed_qubits
            bits = 3 if pauli.is_y_target else 1 if pauli.is_x_target else 2 if pauli.is_z_target else 0
            assert bits
            if target.gate.endswith('_ERROR'):
                assert bits == {'X_ERROR': 1, 'Z_ERROR': 2, 'Y_ERROR': 3}[target.gate]
            record['paulis'][q] = record['paulis'].get(q, 0) ^ bits
        raw_locations.append(str(location))
    normalized = []
    insertions = {}
    for record in channels.values():
        paulis = [[q, {1: 'X', 2: 'Z', 3: 'Y'}[bits]] for q, bits in sorted(record['paulis'].items()) if bits]
        if paulis:
            record['paulis'] = paulis
            normalized.append(record)
            insertions.setdefault(record['instruction_offset'], []).extend(paulis)
    faulted = stim.Circuit()
    for i, instruction in enumerate(circuit):
        if instruction.name not in NOISE:
            faulted.append(instruction)
        for q, pauli in insertions.get(i, []):
            faulted.append(pauli, [q])
    reference = circuit.without_noise()
    # Reject nondeterministic detectors in either ideal Clifford circuit.
    reference.detector_error_model(allow_gauge_detectors=False)
    faulted.detector_error_model(allow_gauge_detectors=False)
    converter = reference.compile_m2d_converter()
    samples = faulted.reference_sample()[None, :]
    d, o = converter.convert(measurements=samples, separate_observables=True)
    dm = sum(int(v) << i for i, v in enumerate(d[0]))
    om = sum(int(v) << i for i, v in enumerate(o[0]))
    assert (dm, om) == (expected_d, expected_o)
    sd, so = converter.convert(measurements=faulted.compile_sampler(seed=20260911).sample(shots), separate_observables=True)
    assert np.all(sd == d[0]) and np.all(so == o[0])
    if require_logical:
        assert dm == 0 and om != 0 and normalized
    return dict(status='physical_pauli_replay_verified', fault_count=len(normalized),
                detector_mask=dm, observable_mask=om, circuit_sha256=digest(str(circuit).encode()),
                faults=normalized, original_explanations=raw_locations,
                deterministic_detector_check=True, corroborating_shots=shots,
                faulted_circuit=str(faulted))


def static_upper(circuit, code, basis, experiment):
    """Replay the saved weight-eight static witness at permitted locations."""
    meta = json.loads((ROOT / 'inputs/metadata.json').read_text())
    side = 'Z' if basis == 'X' else 'X'
    support = meta['distance_witnesses'][side]
    errors = []
    if experiment == 'memory':
        target_gate = 'Z_ERROR' if basis == 'X' else 'X_ERROR'
        offset = max(i for i, inst in enumerate(circuit) if inst.name == target_gate)
        # Final data measurement flips form a known upper bound in full models.
        assert [t.value for t in circuit[offset].targets_copy()] == list(range(64))
        ticks = sum(inst.name == 'TICK' for inst in list(circuit)[:offset])
        positions = {(q,): (offset, q, q + 1) for q in support}
    else:
        # In a guarded CNOT-only experiment, choose the last noisy gate on q.
        positions = {}
        for offset, inst in enumerate(circuit):
            if inst.name != 'DEPOLARIZE2':
                continue
            targets = [t.value for t in inst.targets_copy()]
            for start in range(0, len(targets), 2):
                for q in targets[start:start + 2]:
                    if q in support:
                        positions[q,] = (offset, start, start + 2)
        assert len(positions) == len(support)
    # Obtain exact signatures by replacing the chosen channel with one Pauli
    # error, then use Stim's own explanation for a permitted representative.
    for q in support:
        offset, start, stop = positions[q,]
        probe = stim.Circuit()
        for i, inst in enumerate(circuit):
            if inst.name not in NOISE:
                probe.append(inst)
            if i == offset:
                probe.append(side + '_ERROR', [q], 0.001)
        effects = extract_fault_effects(detector_error_model(probe))
        assert len(effects) == 1
        errors.append(explain_masks(circuit, effects[0].detector_mask, effects[0].observable_mask))
    return replay(circuit, errors)


def bounded_heuristic(circuit_path, output_path, seconds=12):
    command = [sys.executable, '-B', str(ROOT / 'faults.py'), '--heuristic', str(circuit_path), '--output', str(output_path)]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        return dict(status='timeout', seconds=seconds, upper_bound=None)
    if result.returncode:
        return dict(status='worker_error', stderr=result.stderr, upper_bound=None)
    record = json.loads(Path(output_path).read_text())
    record['wall_seconds'] = time.monotonic() - started
    return record


def heuristic_worker(path, output):
    circuit = stim.Circuit.from_file(path)
    try:
        errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=24,
            dont_explore_edges_increasing_symptom_degree=True,
            canonicalize_circuit_errors=True)
    except ValueError as exc:
        save(output, dict(status='no_witness_within_heuristic_restrictions', upper_bound=None, reason=str(exc)))
        return
    validated = replay(circuit, errors)
    faulted = validated.pop('faulted_circuit')
    Path(output).with_suffix('.witness.stim').write_text(faulted)
    save(output, dict(status='witness_verified', upper_bound=validated['fault_count'], witness=validated,
                      method='bounded Stim heuristic: upper bound only'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heuristic', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    heuristic_worker(args.heuristic, args.output)
