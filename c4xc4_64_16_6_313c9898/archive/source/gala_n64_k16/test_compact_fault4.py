import itertools
import random

import pytest

from compact_fault4 import find_four
from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import FaultEffect
from n32_k4_d6_reference_code.schedule_fault_search.screening import find_four_fault_witness


def brute(effects):
    for indices in itertools.combinations(range(len(effects)),4):
        d=o=0
        for i in indices:
            d^=effects[i].detector_mask
            o^=effects[i].observable_mask
        if d==0 and o:
            return indices
    return None


@pytest.mark.parametrize('seed',range(20))
@pytest.mark.parametrize('projection_bits',[1,3,None])
def test_exact_against_brute_with_deliberate_hash_collisions(seed,projection_bits):
    rng=random.Random(seed)
    masks=rng.sample(range(1,128),16)
    # Spread the syndrome across four chunks, including the high chunk.
    masks=[m ^ (m<<67) ^ (m<<193) for m in masks]
    effects=tuple(FaultEffect(d,rng.randrange(16),0.001,i) for i,d in enumerate(masks))
    expected=brute(effects)
    actual,stats=find_four(effects,projection_bits=projection_bits)
    assert (actual is None)==(expected is None)
    old=find_four_fault_witness(effects,num_detectors=200)
    assert (actual is None)==(old is None)
    assert stats['pair_count']==120


def test_duplicate_detector_masks_are_rejected():
    effects=tuple(FaultEffect(d,o,0.001,i) for i,(d,o) in enumerate([(1,0),(1,1),(2,0),(3,0)]))
    with pytest.raises(ValueError,match='unique detector'):
        find_four(effects)


def test_no_false_witness_for_same_observable_pair_collisions():
    effects=tuple(FaultEffect(i,i&1,0.001,i) for i in range(1,25))
    assert brute(effects) is None
    assert find_four(effects,projection_bits=1)[0] is None
