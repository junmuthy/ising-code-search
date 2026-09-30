"""Dedicated regressions added before the September 14 code-extension pilot."""
import copy
import importlib.util
import json
import random

import numpy as np
import pytest
import stim

from model import ROOT, SHARED, load
from schedule import separated_schedule, verify
from circuit import build
from faults import detector_error_model, explain_masks, extract_fault_effects, replay

spec = importlib.util.spec_from_file_location('shared_replay_audit', SHARED/'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('presentation', ['independent', 'redundant'])
@pytest.mark.parametrize('seed', [17, 31, 47])
def test_schedule_and_noiseless_boundaries(presentation, seed):
    code = load(presentation)
    schedule = separated_schedule(code, seed)
    check = verify(code, schedule)
    assert check['cnots'] == (544 if presentation == 'independent' else 768)
    for basis in ('X', 'Z'):
        for rounds in (1, 3, 6):
            c = build(code, schedule, basis, rounds=rounds, noise='none')
            c.detector_error_model(allow_gauge_detectors=False)
            d, o = c.compile_detector_sampler(seed=17).sample(32, separate_observables=True)
            assert not np.any(d) and not np.any(o)
            assert c.num_qubits == (112 if presentation == 'independent' else 128)


@pytest.mark.parametrize('noise', ['cnot', 'full', 'full_wait'])
@pytest.mark.parametrize('basis', ['X', 'Z'])
def test_random_single_fault_effects(noise, basis):
    code = load()
    c = build(code, separated_schedule(code, 17), basis, noise=noise)
    effects = extract_fault_effects(detector_error_model(c))
    for effect in random.Random(62).sample(effects, 5):
        error = explain_masks(c, effect.detector_mask, effect.observable_mask)
        result = replay(c, [error], require_logical=False, shots=4)
        assert result['fault_count'] == 1
        assert result['detector_mask'] == effect.detector_mask
        assert result['observable_mask'] == effect.observable_mask


def test_every_pilot_witness_independently_replayed():
    report = json.loads((ROOT/'pilot_v1/report.json').read_text())
    assert len(report['schedules']) == 9
    for result in report['schedules']:
        folder = ROOT/'pilot_v1'/result['schedule_id']
        schedule = json.loads((folder/'schedule.json').read_text())
        code = json.loads((folder/'code.json').read_text())
        verify(code, schedule)
        for basis, case in result['cases'].items():
            c = stim.Circuit.from_file(folder/f'{basis}.stim')
            assert c == build(code, schedule, basis)
            witness = case['heuristic']['witness']
            replayed = audit.replay_saved(c, witness, schedule['depth'])
            assert replayed['fault_count'] == case['upper_bound'] < 5
            bad = copy.deepcopy(witness)
            bad['observable_mask'] ^= 1
            with pytest.raises(AssertionError):
                audit.replay_saved(c, bad, schedule['depth'])


def test_duplicate_gate_rejected():
    code = load()
    schedule = separated_schedule(code)
    schedule['layers'][0].append(schedule['layers'][0][0])
    with pytest.raises(AssertionError):
        verify(code, schedule)
