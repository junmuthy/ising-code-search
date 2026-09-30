"""Independent checks of code, schedule, detectors, and physical replay."""
import copy
import json
import numpy as np
import pytest

from model import ROOT, load, relations
from schedule import materialize, verify
from circuit import build
from faults import detector_error_model, extract_fault_effects, explain_masks, replay


def schedules():
    report = json.loads((ROOT / 'schedules_v1/report.json').read_text())
    names = ['baseline_depth24.json'] + [item['file'] for item in report['schedules']]
    return [json.loads((ROOT / 'schedules_v1' / name).read_text()) for name in names]


def test_pinned_code_and_all_schedule_structures():
    code = load()
    for kind, deps in code['relations'].items():
        assert len(deps) == 8
        for dep in deps:
            word = 0
            for i in dep:
                word ^= code['checks'][kind][i]
            assert word == 0
    records = schedules()
    assert len(records) == 9
    for record in records:
        result = verify(code, record)
        assert result['cnots'] == 768
        assert result['depth_optimal'] == (record['depth'] == 12)
        if record['depth'] == 12:
            assert result['layer_sizes'] == [64] * 12


@pytest.mark.parametrize('basis', ['X', 'Z'])
@pytest.mark.parametrize('rounds', [1, 3, 6])
def test_ideal_detectors_and_logicals(basis, rounds):
    code = load()
    for record in schedules():
        circuit = build(code, record, basis, rounds=rounds, noise='none')
        circuit.detector_error_model(allow_gauge_detectors=False)
        d, o = circuit.compile_detector_sampler(seed=8).sample(64, separate_observables=True)
        assert not np.any(d) and not np.any(o)
        assert circuit.num_observables == 16 and circuit.num_detectors == 80 * rounds


def test_schedule_rejects_duplicate_edge():
    code = load()
    record = copy.deepcopy(schedules()[1])
    record['layers'][0].append(record['layers'][0][0])
    with pytest.raises(AssertionError):
        verify(code, record)


@pytest.mark.parametrize('basis', ['X', 'Z'])
@pytest.mark.parametrize('noise', ['cnot', 'full', 'full_wait'])
def test_fault_signature_replay(basis, noise):
    code = load()
    circuit = build(code, schedules()[1], basis, noise=noise)
    effects = extract_fault_effects(detector_error_model(circuit))
    selected = [effects[i] for i in (0, len(effects) // 3, len(effects) // 2, len(effects) - 1)]
    for effect in selected:
        error = explain_masks(circuit, effect.detector_mask, effect.observable_mask)
        result = replay(circuit, [error], require_logical=False, shots=4)
        assert result['fault_count'] == 1
        assert result['detector_mask'] == effect.detector_mask
        assert result['observable_mask'] == effect.observable_mask


def test_empty_and_full_rank_dependencies():
    assert relations([1, 2, 4]) == []
    assert relations([1, 2, 3]) == [[0, 1, 2]]
