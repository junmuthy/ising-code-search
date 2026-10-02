import copy
import json

import numpy as np
import pytest

from orbit_schedule import HERE, circuits, old_schedule, prepare, separated, solver

ROOT = HERE/'orbit_campaign/finalists_v1'


@pytest.mark.parametrize('key', ['1fad29e8c9bc9e4f3d381f850ec4665de20b46b6aa41ecc677a4cd6b2a1b196d',
                               'bc5e4e6336735d0eb4bfd1cb7059314667efaf33b3491f574c30cf5f9a5f3e5e'])
@pytest.mark.parametrize('presentation', ['independent', 'full_orbits'])
def test_generic_separated_schedule(key, presentation):
    data = json.loads((ROOT/key/'independent/candidate.json').read_text())
    code = prepare(data, presentation)
    schedule = separated(code, 17)
    old_schedule.verify(code, schedule)
    assert schedule['depth'] == 20
    expected = 14 if presentation == 'full_orbits' and key.startswith('1fad') else 10
    assert schedule['validation']['minimum_depth_lower_bound'] == expected
    for basis in ('X', 'Z'):
        for rounds in (1, 3, 6):
            c = circuits.build(code, schedule, basis, rounds=rounds, noise='none')
            c.detector_error_model(allow_gauge_detectors=False)
            d, o = c.compile_detector_sampler(seed=17).sample(16, separate_observables=True)
            assert not np.any(d) and not np.any(o)
    bad = copy.deepcopy(schedule)
    bad['layers'][0].append(bad['layers'][0][0])
    with pytest.raises(AssertionError):
        old_schedule.verify(code, bad)


def test_restriction_preserves_checks_and_clean_schedule():
    from orbit_restrict import restrict
    root = HERE/'orbit_campaign/schedules_bc5e4e63_v1/full_orbits_9674ddc3c1b01d2a'
    code = json.loads((root/'code.json').read_text())
    schedule = json.loads((root/'schedule.json').read_text())
    for seed in (17, 23, 31):
        new, reduced = restrict(code, schedule, seed)
        assert reduced['depth'] == 12
        assert reduced['validation']['cnots'] == 416
        assert all(len(rows) == 24 for rows in new['checks'].values())
        for basis in ('X', 'Z'):
            c = circuits.build(new, reduced, basis, noise='none')
            c.detector_error_model(allow_gauge_detectors=False)
            d, o = c.compile_detector_sampler(seed=17).sample(32, separate_observables=True)
            assert not np.any(d) and not np.any(o)
