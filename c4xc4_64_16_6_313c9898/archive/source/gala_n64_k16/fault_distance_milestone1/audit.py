"""Audit completed artifacts by rebuilding circuits and replaying saved faults."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stim

from model import ROOT, digest, load, save
from circuit import NOISE, build
from schedule import verify
from faults import detector_error_model, extract_fault_effects


def replay_saved(circuit, witness, depth):
    assert digest(str(circuit).encode()) == witness['circuit_sha256']
    insertion, channels, described = {}, set(), []
    for fault in witness['faults']:
        offset = fault['instruction_offset']
        start, stop = fault['target_range']
        inst = circuit[offset]
        assert inst.name == fault['gate'] and inst.name in NOISE
        assert inst.gate_args_copy()[0] > 0
        targets = inst.targets_copy()[start:stop]
        expected_length = 2 if inst.name == 'DEPOLARIZE2' else 1
        assert len(targets) == stop - start == expected_length
        assert start % expected_length == 0
        qubits = {target.value for target in targets}
        key = offset, start, stop
        assert key not in channels
        channels.add(key)
        assert fault['paulis']
        assert len({q for q, _ in fault['paulis']}) == len(fault['paulis'])
        for q, pauli in fault['paulis']:
            assert q in qubits and pauli in ('X', 'Y', 'Z')
            if inst.name.endswith('_ERROR'):
                assert pauli == inst.name[0]
            insertion.setdefault(offset, []).append((q, pauli))
        ticks = sum(item.name == 'TICK' for item in list(circuit)[:offset])
        described.append(dict(gate=inst.name, paulis=fault['paulis'], ticks=ticks,
                              round=(ticks - 2) // (depth + 2) if inst.name == 'DEPOLARIZE2' else None,
                              layer=(ticks - 2) % (depth + 2) if inst.name == 'DEPOLARIZE2' else None))
    actual = stim.Circuit()
    for offset, instruction in enumerate(circuit):
        if instruction.name not in NOISE:
            actual.append(instruction)
        for q, pauli in insertion.get(offset, []):
            actual.append(pauli, [q])
    actual.detector_error_model(allow_gauge_detectors=False)
    converter = circuit.without_noise().compile_m2d_converter()
    reference_d, reference_o = converter.convert(measurements=actual.reference_sample()[None, :], separate_observables=True)
    dm = sum(int(v) << i for i, v in enumerate(reference_d[0]))
    om = sum(int(v) << i for i, v in enumerate(reference_o[0]))
    assert dm == witness['detector_mask'] == 0
    assert om == witness['observable_mask'] and om != 0
    assert len(channels) == witness['fault_count']
    d, o = converter.convert(measurements=actual.compile_sampler(seed=17).sample(32), separate_observables=True)
    assert np.all(d == reference_d) and np.all(o == reference_o)
    return dict(fault_count=len(channels), zero_detectors=True,
                logical_indices=[i for i in range(16) if om >> i & 1], physical_locations=described)


def run(folder):
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text())
    assert report['status'] == 'complete'
    for name, expected in report['provenance']['source_sha256'].items():
        assert digest(Path(name).read_bytes()) == expected, name
    code = load()
    selected = json.loads((folder / 'selected_schedule.json').read_text())
    verify(code, selected)
    entries = [(folder / 'screen' / item['schedule_id'],
                json.loads((ROOT / 'schedules_v1' / f"schedule_{item['schedule_id']}.json").read_text()))
               for item in report['screen']]
    entries += [(folder / name, selected) for name in report['cases']]
    verified, signatures = {}, {}
    for path, record in entries:
        result = json.loads((path / 'result.json').read_text())
        config = result['configuration']
        circuit = build(code, record, config['basis'], experiment=config['experiment'], noise=config['noise'])
        assert circuit == stim.Circuit.from_file(path / 'circuit.stim')
        assert digest(str(circuit).encode()) == config['circuit_sha256']
        exact = json.loads((path / 'exact_through_three.json').read_text())
        assert result['lower_bound'] == exact['lower_bound'] == 4
        assert [(a['fault_count'], a['status']) for a in exact['attempts']] == [(1, 'unsat'), (2, 'unsat'), (3, 'unsat')]
        witness = json.loads((path / 'physical_witness.json').read_text())
        replayed = replay_saved(circuit, witness, record['depth'])
        assert replayed['fault_count'] == result['upper_bound']
        name = str(path.relative_to(folder))
        verified[name] = replayed
        if path.parent == folder:
            signatures[path.name] = {(e.detector_mask, e.observable_mask)
                                     for e in extract_fault_effects(detector_error_model(circuit))}
    comparisons = {}
    for basis in ('X', 'Z'):
        base = signatures[f'memory_cnot_{basis}']
        assert base == signatures[f'memory_full_{basis}'] == signatures[f'memory_full_wait_{basis}']
        assert signatures[f'bulk_cnot_{basis}'] <= base
        comparisons[basis] = dict(memory_noise_models_have_identical_signature_sets=True,
                                  distinct_memory_effects=len(base), guarded_effects_are_subset=True)
    return dict(status='independent_artifact_replay_passed', verified_cases=verified,
                noise_signature_comparison=comparisons,
                source_sha256=digest(Path(__file__).read_bytes()),
                note='Witness replay, fingerprints, and signature comparisons audited here; lower enumeration source and saved UNSAT stages checked, not rerun.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = run(args.input)
    save(destination, result)
    print(json.dumps(dict(status=result['status'], cases=len(result['verified_cases']),
                          noise_comparison=result['noise_signature_comparison'])))
