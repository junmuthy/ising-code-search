"""Small exhaustive distance controls and saved-witness negative controls."""
import copy
import itertools
import json
import random

import pytest
import stim

from model import ROOT
from audit import replay_saved
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import FaultEffect, _find_low_cardinality_witness


@pytest.mark.parametrize('seed', range(16))
def test_direct_distance_engine_against_brute_force(seed):
    rng = random.Random(seed)
    effects = tuple(FaultEffect(rng.randrange(1 << 5), rng.randrange(1 << 3), .001, i) for i in range(12))
    for weight in (1, 2, 3):
        expected = False
        for combination in itertools.combinations(effects, weight):
            d = o = 0
            for effect in combination:
                d ^= effect.detector_mask
                o ^= effect.observable_mask
            expected |= d == 0 and o != 0
        result = _find_low_cardinality_witness(effects, weight, heartbeat_seconds=20, callback=lambda _: None)
        assert (result is not None) == expected


def test_physical_witness_replay_rejects_tampered_logical_mask():
    folder = ROOT / 'results_v1/memory_cnot_X'
    circuit = stim.Circuit.from_file(folder / 'circuit.stim')
    witness = json.loads((folder / 'physical_witness.json').read_text())
    assert replay_saved(circuit, witness, 12)['fault_count'] == 5
    bad = copy.deepcopy(witness)
    bad['observable_mask'] ^= 1
    with pytest.raises(AssertionError):
        replay_saved(circuit, bad, 12)


def test_physical_witness_replay_rejects_unsupported_fault_location():
    folder = ROOT / 'results_v1/memory_cnot_X'
    circuit = stim.Circuit.from_file(folder / 'circuit.stim')
    witness = json.loads((folder / 'physical_witness.json').read_text())
    witness['faults'][0]['paulis'][0][0] = 10000
    with pytest.raises(AssertionError):
        replay_saved(circuit, witness, 12)
