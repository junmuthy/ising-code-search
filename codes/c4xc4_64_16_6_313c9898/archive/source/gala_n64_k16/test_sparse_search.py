"""Exact controls for the bounded k=16 sparse campaign."""
import itertools
import json
from pathlib import Path
import random

import numpy as np
import pytest

from sparse_core import (HERE, canonical_x, common_h, commute, grid_module,
                         independent_audit, pairing, permute, rank, rref, space_key)
from sparse_models import (SmallGroup, RING, commutator, decode, initial_control,
                           lift_gl, mat_mul, transvection, translations, two_block)
from sparse_archive import choose_grid, component_data, JOURNAL
from algebra import pack, unpack
from models import Extension


@pytest.mark.parametrize('family,alpha,beta', [('extension', a, b) for a, b in itertools.product(range(2), repeat=2)] + [('c8c4', 0, 0), ('c4c4c2', 0, 0)])
def test_groups_and_lifts(family, alpha, beta):
    e = SmallGroup(family, alpha, beta)
    t = e.table
    for a in range(e.order):
        assert np.array_equal(t[t[a, :], :], t[a, t])
    rng = random.Random(31)
    for _ in range(5):
        a, b = rng.getrandbits(e.order), rng.getrandbits(e.order)
        aa, bb = unpack(e.lift(a), e.order), unpack(e.lift(b), e.order)
        assert pack(aa@bb%2) == e.lift(e.polynomial_mul(a, b))
        hx, hz = e.checks(a, b)
        assert commute(hx, hz) == (e.polynomial_mul(a, b) == e.polynomial_mul(b, a))
        for g in e.generators:
            for h in e.generators:
                assert all(e.mul(g, e.mul(q, h)) == e.mul(e.mul(g, q), h) for q in range(e.order))
    for fold in e.folds():
        assert sorted(fold) == list(range(64))


def test_faithful_matrix_lift():
    rng = random.Random(32)
    for _ in range(8):
        a, b = decode(rng.getrandbits(64)), decode(rng.getrandbits(64))
        aa, bb = unpack(lift_gl(a), 32), unpack(lift_gl(b), 32)
        assert pack(aa@bb%2) == lift_gl(mat_mul(a, b))
        hx, hz = two_block(lift_gl(a), lift_gl(b))
        assert commute(hx, hz) == (commutator(a, b) == 0)


def test_known_positive_and_equivariant_basis_recovery():
    data = json.loads((HERE/'certified/64_16_6_c4xc4/candidate.json').read_text())
    info, grid = grid_module(data['hx'], data['hz'], data['px'], data['py'], data['central'])
    assert info['status'] == 'regular_grid'
    h, logical = common_h(data['hx'], data['hz'], grid, data['certificate']['fold'])
    assert h['status'] == 'logical_hadamard'
    certified = dict(data, certificate={**logical, 'fold': data['certificate']['fold']})
    result, _ = independent_audit(certified)
    assert result['distance']['X']['lower_bound'] == result['distance']['X']['upper_bound'] == 6
    assert result['metrics']['row_space_component_sizes'] == [64]


def test_negative_module_and_wrong_ranks():
    h, z = initial_control()
    identity = list(range(64))
    assert grid_module(h, h, identity, identity)[0]['status'] == 'norm_zero'
    assert grid_module([], [], identity, identity)[0]['status'] == 'wrong_ranks'


def test_orthogonal_control_and_reconstruction():
    h, z = initial_control()
    rng = random.Random(33)
    for _ in range(5):
        h, z = transvection(h, z, rng.getrandbits(16), rng.getrandbits(16), rng.sample(range(4), 4))
        assert rank(h) == 24 and commute(h, h) and commute(h, z)
        assert pairing(z, z) == [1 << i for i in range(16)]
        info, grid = grid_module(h, h, *translations())
        assert info['status'] == 'regular_grid'
        grid['z'], grid['x'] = z, z
        assert common_h(h, h, grid, list(range(64)))[0]['status'] == 'logical_hadamard'


def test_archive_contains_known_positive():
    records = [json.loads(s) for s in JOURNAL.read_text().splitlines()]
    original = json.loads((HERE/'certified/64_16_6_c4xc4/candidate.json').read_text())
    record = next(r for r in records if r['id'] == original['parent_id'])
    data, rights, folds = component_data(record, {tuple(record['extension']): Extension(*record['extension'])})
    assert (data['hx'], data['hz']) == (original['hx'], original['hz'])
    info, grid = choose_grid(data, rights)
    assert info['status'] == 'regular_grid'
    assert any(common_h(data['hx'], data['hz'], grid, p)[1] is not None for _, _, p in folds)


def test_row_space_key_generator_invariant():
    a = [7, 25, 42]
    b = [a[2], a[0]^a[1], a[1], 0, a[0]^a[2]]
    assert rref(a) == rref(b)
    assert space_key(a, a) == space_key(b, b)


def test_twisted_family_reconstructs_known_code():
    from sparse_twisted import GROUP, checks, known_seed
    a, b, fold, coordinates = known_seed()
    assert sorted(coordinates) == list(range(64))
    hx, hz = checks(a, b, fold)
    info, grid = grid_module(hx, hz, *GROUP.physical())
    assert info['status'] == 'regular_grid'
    assert common_h(hx, hz, grid, fold)[0]['status'] == 'logical_hadamard'
    # The conventional transpose-exchange parent is genuinely different.
    standard_x, standard_z = GROUP.checks(a, b)
    assert space_key(hx, hz) != space_key(standard_x, standard_z)


def test_twisted_linear_equations_and_folds():
    from sparse_twisted import GROUP, checks, equation, fold_catalog
    from algebra import combine, power
    from search import linear_kernel
    rng = random.Random(34)
    folds = list(fold_catalog())
    for fold in folds:
        assert power(fold, 2) in (list(range(64)), GROUP.right(16))
    for fold in rng.sample(folds, 8):
        a, b, c = [rng.getrandbits(32) for _ in range(3)]
        assert equation(a, 0, fold) == 0
        assert equation(a, b^c, fold) == equation(a, b, fold)^equation(a, c, fold)
        kernel = linear_kernel([equation(a, 1 << i, fold) for i in range(32)], 32)
        for _ in range(3):
            partner = combine(kernel, rng.getrandbits(len(kernel)))
            hx, hz = checks(a, partner, fold)
            assert equation(a, partner, fold) == 0 and commute(hx, hz)


def test_selected_sparse_hit_and_generator_certificate():
    from collections import Counter
    from algebra import combine, same_space
    from sparse_twisted import checks
    from check_weight_reduction import enumerate_space, orbit_search
    root = HERE/'sparse_campaign/finalists_v1'
    report = json.loads((root/'report.json').read_text())
    original = json.loads((root/'best_symmetric/candidate.json').read_text())
    reduced = json.loads((root/'best_independent/candidate.json').read_text())
    hx, hz = checks(original['a'], original['b'], original['certificate']['fold'])
    assert hx == original['hx'] and hz == original['hz']
    assert report['selected_id'] == 'd9173bb8cd878461c62a0dd5b7f9cb5f494b401aaf662cf40e45eb883551ebcf'
    saved_weights = json.loads((root/f"weights_{report['selected_id']}.json").read_text())
    for side in ('hx', 'hz'):
        assert same_space(original[side], reduced[side])
        assert len(reduced[side]) == rank(reduced[side]) == 24
        assert Counter(v.bit_count() for v in reduced[side]) == {8: 4, 12: 20}
        enum = enumerate_space(original[side], ceiling=12)
        assert json.loads(json.dumps(enum)) == saved_weights[side]
        assert enum['rank_filtration'][11] == 4
        assert enum['minimum_possible_maximum_check_weight'] == 12
        assert enum['minimum_basis_total_weight'] == 272
        change = report['basis_changes'][side]
        assert [combine(original[side], c) for c in change['new_from_original']] == reduced[side]
        assert [combine(reduced[side], c) for c in change['original_from_new']] == original[side]
        orbit = orbit_search(enum['low_weight_rows'], original['px'], original['py'], original['certificate']['fold'])
        assert orbit['minimum_total_weight_per_sector'] == 384
        assert orbit['number_checks_per_sector'] == 32
        for name, p in (('Tx', original['px']), ('Ty', original['py'])):
            assert [combine(reduced[side], c) for c in report['stabilizer_actions'][side][name]] == [permute(v, p) for v in reduced[side]]
    assert original['certificate'] == reduced['certificate']
    audit, _ = independent_audit(reduced)
    assert audit['metrics']['row_space_component_sizes'] == [64]
    assert audit['actions']['H']['logical_permutation'] == [4*((-i)%4)+(2-j)%4 for i in range(4) for j in range(4)]
    for side in ('X', 'Z'):
        assert audit['distance'][side]['lower_bound'] == audit['distance'][side]['upper_bound'] == 6


def test_all_eight_new_hits_are_reconstructible_and_certified():
    from sparse_twisted import GROUP, checks
    paths = sorted((HERE/'sparse_campaign/twisted_w12_v1/certified').glob('*/candidate.json'))
    assert len(paths) == 8
    for path in paths:
        data = json.loads(path.read_text())
        hx, hz = checks(data['a'], data['b'], data['certificate']['fold'])
        assert (hx, hz) == (data['hx'], data['hz'])
        assert max(v.bit_count() for v in hx+hz) == 12
        info, _ = grid_module(hx, hz, data['px'], data['py'], data['central'])
        assert info['status'] == 'regular_grid'
        audit, _ = independent_audit(data)
        assert audit['metrics']['row_space_component_sizes'] == [64]
        for side in ('X', 'Z'):
            assert audit['distance'][side]['lower_bound'] >= 6
            upper = audit['distance'][side]['upper_bound']
            assert upper is None or upper >= audit['distance'][side]['lower_bound']


def test_weight_seven_matching_against_small_bruteforce():
    from sparse_distance_upgrade import weight_seven_search
    from algebra import basis, reduce, quotient_basis, nullspace
    rng = random.Random(35)
    for _ in range(10):
        h = [rng.getrandbits(9) for _ in range(3)]
        stabilizers = nullspace(h, 9)[:2]
        b = basis(stabilizers)
        brute = [support for support in itertools.combinations(range(9), 7)
                 if commute(h, [sum(1 << q for q in support)])
                 and reduce(sum(1 << q for q in support), b)]
        result = weight_seven_search(h, stabilizers, n=9)
        assert (result['status'] == 'logical_witness') == bool(brute)
        if brute:
            assert tuple(result['support']) in brute


def test_distance_eight_handoff():
    from sparse_distance_upgrade import weight_seven_search
    from algebra import basis, reduce, power, combine, same_space
    root = HERE/'sparse_campaign/distance8_handoff_v1'
    code = json.loads((root/'independent/candidate.json').read_text())
    original = json.loads((root/'symmetric/candidate.json').read_text())
    full = json.loads((root/'full_audit.json').read_text())
    report = json.loads((root/'report.json').read_text())
    fold = code['certificate']['fold']
    assert power(fold, 2) == list(range(64))
    assert [permute(v, fold) for v in code['hx']] == code['hz']
    assert [permute(v, fold) for v in code['hz']] == code['hx']
    for side in ('hx', 'hz'):
        assert same_space(code[side], original[side])
        assert len(code[side]) == rank(code[side]) == 24
        assert all(v.bit_count() == 12 for v in code[side])
        assert report['enumerations'][side]['minimum_possible_maximum_check_weight'] == 12
        changes = report['basis_changes'][side]
        assert [combine(original[side], c) for c in changes['new_from_original']] == code[side]
        assert [combine(code[side], c) for c in changes['original_from_new']] == original[side]
    audit, _ = independent_audit(code)
    assert audit['metrics']['row_space_component_sizes'] == [64]
    assert audit['actions']['H']['logical_permutation'] == [4*i+(2*i+1-j)%4 for i in range(4) for j in range(4)]
    for side, h, s in (('Z', code['hx'], code['hz']), ('X', code['hz'], code['hx'])):
        assert audit['distance'][side]['lower_bound'] == 7
        seven = weight_seven_search(h, s)
        assert seven['status'] == 'excluded'
        witness = full['distance'][side]['weight_eight_witness']
        word = sum(1 << q for q in witness)
        assert len(witness) == 8 and commute(h, [word]) and reduce(word, basis(s))
        assert full['distance'][side]['lower_bound'] == full['distance'][side]['upper_bound'] == 8
