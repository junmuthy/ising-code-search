import copy

import numpy as np
import pytest

from schedule_313c9898 import (OLD, optimal_basis, restrict_rank, select_indices,
                               source_schedule, translation_edges)
from orbit_schedule import circuits, old_schedule, solver
from sparse_core import permute, rank, same_space


@pytest.mark.parametrize('count,paired', [(24, True), (24, False), (28, True)])
def test_rank_restriction(count, paired):
    code, schedule = source_schedule(sorted(OLD.glob('redundant_*'))[0])
    new, result = restrict_rank(code, schedule, 19, count, paired)
    assert result['depth'] == 12 and result['validation']['cnots'] == 24*count
    for s in ('X', 'Z'):
        assert len(new['checks'][s]) == count
        assert rank(new['checks'][s]) == 24
        assert same_space(new['checks'][s], code['checks'][s])
        assert all(len(v) == 6 for v in new['logical_supports'][s])
        for rounds in (1, 3, 6):
            c = circuits.build(new, result, s, rounds=rounds, noise='none')
            c.detector_error_model(allow_gauge_detectors=False)
            d, o = c.compile_detector_sampler(seed=17).sample(16, separate_observables=True)
            assert not np.any(d) and not np.any(o)
    bad = copy.deepcopy(result)
    bad['layers'][0].append(bad['layers'][0][0])
    with pytest.raises(AssertionError):
        old_schedule.verify(new, bad)


def test_invalid_retention():
    with pytest.raises(ValueError):
        select_indices([1, 2, 3], 1, 0)


def test_optimal_basis_and_order_four():
    code = optimal_basis(17)
    assert sum(w.bit_count() for rows in code['checks'].values() for w in rows) == 448
    fold = code['fold']
    square = [fold[fold[q]] for q in range(64)]
    assert square != list(range(64))
    assert [square[square[q]] for q in range(64)] == list(range(64))
    assert [permute(w, fold) for w in code['checks']['X']] == code['checks']['Z']


def test_translation_edge_model():
    code, _ = source_schedule(sorted(OLD.glob('redundant_*'))[0])
    code['edge_map'] = translation_edges(code)
    assert set(code['edge_map'].values()) == set(range(12))
    schedule, info = solver(code, 12, seed=41, milliseconds=10000)()
    assert info['status'] == 'sat'
    old_schedule.verify(code, schedule)
    for basis in ('X', 'Z'):
        c = circuits.build(code, schedule, basis, noise='none')
        c.detector_error_model(allow_gauge_detectors=False)


@pytest.mark.parametrize('small_first',[True,False])
@pytest.mark.parametrize('seed',[17,31,43])
def test_staged_optimal_schedules(small_first,seed):
    from schedule_313c9898_seeded import seeded_basis
    from schedule_313c9898_staged import small_block,staged
    source,reference=source_schedule(sorted(OLD.glob('redundant_*'))[1])
    code,selected=seeded_basis(source,seed)
    small,info=small_block(code,seed)
    assert info['status']=='sat'
    schedule=staged(code,selected,reference,small,small_first)
    assert schedule['validation']['cnots']==448
    assert schedule['depth']<=20
    for basis in ('X','Z'):
        for rounds in (1,3,6):
            c=circuits.build(code,schedule,basis,noise='none',rounds=rounds)
            c.detector_error_model(allow_gauge_detectors=False)
            d,o=c.compile_detector_sampler(seed=17).sample(16,separate_observables=True)
            assert not np.any(d) and not np.any(o)
