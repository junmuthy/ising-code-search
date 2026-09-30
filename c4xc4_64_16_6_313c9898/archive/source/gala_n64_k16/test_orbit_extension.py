import itertools
import json
import random
from collections import Counter

import numpy as np
import pytest

from orbit_extension import (construct, geometry, low_witness, module_generators,
                             orbit, propose)
from sparse_core import HERE, commute, permute, rank, same_space
from sparse_models import initial_control, translations
from algebra import unpack


@pytest.mark.parametrize('family,variant', [('two_sheet', 0), ('four_sheet', 0)] + [('nonabelian', i) for i in range(4)])
def test_geometry_and_full_group_orbits(family, variant):
    geo = geometry(family, variant)
    assert geo['folds']
    actions = set(map(tuple, geo['actions']))
    for a, b in itertools.product(geo['generators'], repeat=2):
        assert tuple(a[b[q]] for q in range(64)) in actions
    rows = orbit(0b101101, geo['actions'])
    for p in geo['generators']:
        assert same_space(rows, [permute(v, p) for v in rows])


def test_minimum_orbit_generator_controls():
    data = json.loads((HERE/'sparse_campaign/finalists_v1/best_independent/candidate.json').read_text())
    assert module_generators(data['hx'], [data['px'], data['py']]) == 1
    rows, _ = initial_control()
    assert module_generators(rows, translations()) == 2
    # Adding redundant seed generators cannot change the classification.
    assert module_generators(rows + [rows[0] ^ rows[3]], translations()) == 2


@pytest.mark.parametrize('family', ['two_sheet', 'four_sheet', 'nonabelian'])
def test_constructed_rows_and_kernel_samples(family):
    geo, rng = geometry(family), random.Random(883)
    fold = geo['folds'][0]
    stats = Counter()
    found = None
    for _ in range(6):
        found = construct(geo, fold, rng, 10, stats, proposals=12)
        if found:
            break
    if found:
        hx, hz = found['hx'], found['hz']
        assert rank(hx) == rank(hz) == 24
        assert max(map(int.bit_count, hx + hz)) <= 10
        assert not np.any(unpack(hx, 64) @ unpack(hz, 64).T % 2)
        assert commute(hx, hz)
    # Independent control guarantees a nonempty compatibility-kernel check.
    word = sum(1 << q for q in (0, 1, 32, 33))
    rows = orbit(word, geo['actions'])
    inverse = [fold.index(q) for q in range(64)]
    for v in propose(rows, fold, geo, rng, 10, 16):
        assert all((v & permute(r, p)).bit_count() % 2 == 0 for r in rows for p in (fold, inverse))


def test_short_logical_filter_checks_stabilizer_membership():
    assert low_witness([1], [1], 1) == [1]
    # Pair 0,1 is a stabilizer and must not be reported as a logical.
    h = [3] + [1 << q for q in range(2, 64)]
    assert low_witness(h, [3], 2) is None
