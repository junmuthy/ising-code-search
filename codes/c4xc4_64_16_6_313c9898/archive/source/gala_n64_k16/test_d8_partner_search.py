"""Independent controls for exact d=8 sparse-partner searches."""
import itertools
import json
import random

import numpy as np
import pytest

from d8_partner_search import (HERE, GROUP, SEEDS, assess_eight, checks, enumerate_kernel,
                               export_hit, partner_space, seed_data, short_error, six_error)
from algebra import basis, combine, commute, nullspace, rank, reduce, unpack


@pytest.mark.parametrize('key', SEEDS)
@pytest.mark.parametrize('side', ['a', 'b'])
def test_exact_linear_rank_space_contains_original(key, side):
    data = seed_data(key)
    fixed, variable = data[side], data['b' if side == 'a' else 'a']
    info, kernel = partner_space(fixed, data['certificate']['fold'], side)
    assert info['fixed_rank'] == 24 and len(kernel) == 20
    assert not reduce(variable, basis(kernel))
    assert combine(info['equation_images'], variable) == 0
    rng = random.Random(71)
    for _ in range(10):
        v = combine(kernel, rng.getrandbits(len(kernel)))
        a, b = (fixed, v) if side == 'a' else (v, fixed)
        hx, hz = checks(a, b, data['certificate']['fold'])
        assert rank(hx) == rank(hz) == 24 and commute(hx, hz)
    for _ in range(10):
        v = rng.getrandbits(32)
        a, b = (fixed, v) if side == 'a' else (v, fixed)
        hx, hz = checks(a, b, data['certificate']['fold'])
        assert (combine(info['equation_images'], v) == 0) == (rank(hx) == 24 and commute(hx, hz))


def test_rank_below_target_does_not_impose_false_containment():
    data = seed_data(SEEDS[0])
    fixed = (1 << 0) | (1 << 16)
    info, kernel = partner_space(fixed, data['certificate']['fold'])
    assert info['fixed_rank'] == 16
    assert info['rank_constraint'] == 'combined_rank_checked_after_enumeration'
    assert all(image.bit_length() <= 32 for image in info['equation_images'])


def test_uint32_enumeration_against_scalar():
    kernel = [3, 5, 24]
    words = sorted([combine(kernel, c) for c in range(1 << len(kernel))], key=lambda v: (v.bit_count(), v))
    selected, info = enumerate_kernel(kernel, ceiling=3)
    assert selected == [v for v in words if v.bit_count() <= 3]
    assert info['enumerated'] == 8 and sum(info['weight_histogram'].values()) == 8


def test_short_and_six_errors_against_bruteforce():
    rng = random.Random(72)
    for _ in range(20):
        n = 9
        h = [rng.getrandbits(n) for _ in range(3)]
        stabilizers = nullspace(h, n)[:2]
        pivots = basis(stabilizers)
        def nontrivial(support):
            v = sum(1 << q for q in support)
            return commute(h, [v]) and reduce(v, pivots)
        small = [s for w in range(1, 5) for s in itertools.combinations(range(n), w) if nontrivial(s)]
        six = [s for s in itertools.combinations(range(n), 6) if nontrivial(s)]
        result = short_error(h, stabilizers, n=n)
        assert (result is not None) == bool(small)
        if result is not None:
            assert tuple(result) in small
        result = six_error(h, stabilizers, n=n)
        assert (result is not None) == bool(six)
        if result is not None:
            assert tuple(result) in six


def test_positive_eight_pipeline_and_export(tmp_path):
    data = seed_data(SEEDS[0])
    result, certificate = assess_eight(data['hx'], data['hz'], data['certificate']['fold'])
    assert result['status'] == 'accepted' and result['distance_lower_bound'] == 8
    data['certificate'] = certificate
    export_hit(data, tmp_path/'positive')
    audit = json.loads((tmp_path/'positive/full_audit.json').read_text())
    for side in ('X', 'Z'):
        assert audit['distance'][side]['lower_bound'] == 8
        assert audit['distance'][side]['weight_seven_exclusion']['status'] == 'excluded'


def test_fixed_campaign_full_coverage_and_all_witnesses():
    from certify import gf_rank
    root = HERE/'distance8_weight_search/fixed_w10_v1'
    slices = [json.loads(s) for s in (root/'slices.jsonl').read_text().splitlines()]
    candidates = [json.loads(s) for s in (root/'candidates.jsonl').read_text().splitlines()]
    assert len(slices) == 4 and len(candidates) == 1172
    assert sorted(s['eligible_partners'] for s in slices) == [9, 57, 57, 1049]
    for s in slices:
        assert s['status'] == 'complete_slice' and s['enumerated'] == 2**20
        eligible, _ = enumerate_kernel(s['kernel'], s['partner_weight_ceiling'])
        rows = [r for r in candidates if r['slice_id'] == s['slice_id']]
        assert sorted(r['b' if s['fixed_side'] == 'a' else 'a'] for r in rows) == sorted(eligible)
        for r in rows:
            hx, hz = checks(r['a'], r['b'], s['fold'])
            support = r['witness']
            assert 1 <= len(support) <= 4
            h = unpack(hx, 64)
            stabilizers = unpack(hz, 64)
            v = np.zeros((1, 64), dtype=np.uint8)
            v[0, support] = 1
            assert not np.any(h@v.T%2)
            assert gf_rank(np.vstack((stabilizers, v))) > gf_rank(stabilizers)


def test_support_meet_in_middle_matches_full_kernel():
    from d8_local_complete import sparse_supports
    data = seed_data(SEEDS[0])
    info, kernel = partner_space(data['a'], data['certificate']['fold'])
    expected, _ = enumerate_kernel(kernel, 6)
    found, _ = sparse_supports(info['equation_images'], 6)
    assert found == expected
    all_small, _ = sparse_supports([0]*32, 2)
    assert set(all_small) == {sum(1 << q for q in support)
                              for w in range(3) for support in itertools.combinations(range(32), w)}


def test_fixed_distance_obstruction_certificate():
    from d8_local_complete import fixed_distance_obstruction
    fixed = (1 << 0) | (1 << 16)
    proof = fixed_distance_obstruction(fixed)
    assert proof['status'] == 'fixed_block_short_logical_obstruction'
    assert proof['fixed_rank'] == 16
    assert proof['light_kernel_rank'] > proof['stabilizer_intersection_dimension'] == 8
    assert commute(GROUP.lift(fixed), proof['light_kernel_rows'])
    for key in SEEDS:
        for side in ('a', 'b'):
            assert fixed_distance_obstruction(seed_data(key)[side]) is None


def test_complete_local_coverage_and_dimension_certificates():
    from d8_partner_search import configurations
    from certify import gf_rank
    root = HERE/'distance8_weight_search'
    for phase, name, expected_count in (('neighborhood', 'neighborhood_complete_w10_v1', 680),
                                         ('folds', 'folds_complete_w10_v1', 384)):
        folder = root/name
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['status'] == 'complete_explicit_neighborhood'
        assert summary['configurations_processed'] == expected_count
        records = [json.loads(line) for line in (folder/'slices.jsonl').read_text().splitlines()]
        assert len(records) == len(configurations(phase)) == expected_count
        for r in records:
            if r['status'] == 'covered_by_prior_certificate':
                # Exact prior paths are retained in the certificate.
                from pathlib import Path
                prior = [json.loads(s) for s in Path(r['prior']).read_text().splitlines()]
                witness = next(s for s in prior if s['slice_id'] == r['slice_id'])
                assert witness['status'] in ('complete_slice', 'fixed_rank_obstruction')
            elif r['status'] == 'fixed_rank_obstruction':
                assert rank(GROUP.lift(r['fixed'])) > 24
            elif r['status'] == 'fixed_block_short_logical_obstruction':
                matrix = unpack(GROUP.lift(r['fixed']), 32)
                light = unpack(r['light_kernel_rows'], 32)
                assert not np.any(matrix@light.T%2)
                assert np.all(light.sum(axis=1) <= 7)
                assert gf_rank(light) > 24-gf_rank(matrix)
            else:
                assert r['status'] == 'complete_slice'
                assert r['processed_eligible_partners'] == r['eligible_partners']
                assert not r['candidate_counts'].get('accepted', 0)


def test_scalar_sparse_support_crosscheck_for_large_kernel():
    from pathlib import Path
    root = HERE/'distance8_weight_search/neighborhood_complete_w10_v1'
    records = [json.loads(s) for s in (root/'slices.jsonl').read_text().splitlines()]
    r = next(s for s in records if s.get('method') == 'complete_16_plus_16_support_meet_in_middle'
             and s['partner_weight_ceiling'] <= 4)
    expected = {sum(1 << q for q in support) for w in range(r['partner_weight_ceiling']+1)
                for support in itertools.combinations(range(32), w)
                if combine(r['equation_images'], sum(1 << q for q in support)) == 0}
    rows = [json.loads(s) for s in (root/'candidates.jsonl').read_text().splitlines()]
    found = {s['b' if r['fixed_side'] == 'a' else 'a'] for s in rows if s['slice_id'] == r['slice_id']}
    assert found == expected


def test_shape_pilot_coverage_and_obstruction_certificates():
    from certify import gf_rank
    from d8_shape_pilot import canonical_translation
    root = HERE/'distance8_weight_search/shape_pilot_w10_v1'
    manifest = json.loads((root/'shape_manifest.json').read_text())
    assert manifest['translation_classes'] == 1138
    assert manifest['surviving_translation_classes'] == 368
    assert manifest['surviving_shape_classes'] == 15
    automorphisms = {tuple(v%32 for v in p[:32]) for p in GROUP.folds()}
    assert len(automorphisms) == manifest['group_automorphisms'] == 128
    survivors = set()
    for orbit in manifest['orbits']:
        support = [q for q in range(32) if orbit['representative'] >> q & 1]
        expected = {canonical_translation(sum(1 << p[q] for q in support))
                    for p in automorphisms}
        assert expected == set(orbit['translated_classes'])
        assert not expected & survivors
        survivors |= expected
    assert len(survivors) == 368
    assert len(manifest['obstructions']) == 770
    assert not survivors & {int(v) for v in manifest['obstructions']}
    for word, proof in manifest['obstructions'].items():
        matrix = unpack(GROUP.lift(int(word)), 32)
        r = gf_rank(matrix)
        assert r == proof['fixed_rank']
        if proof['status'] == 'fixed_rank_obstruction':
            assert r > 24
        else:
            assert proof['status'] == 'fixed_block_short_logical_obstruction'
            light = unpack(proof['light_kernel_rows'], 32)
            assert not np.any(matrix@light.T%2)
            assert np.all(light.sum(axis=1) <= 7)
            assert gf_rank(light) > 24-r
    slices = [json.loads(s) for s in (root/'slices.jsonl').read_text().splitlines()]
    assert len(slices) == 60
    expected = {(o['representative'], key, side)
                for o in manifest['orbits'] for key in SEEDS for side in ('a', 'b')}
    assert {(s['fixed'], s['seed_id'], s['fixed_side']) for s in slices} == expected
    for s in slices:
        assert s['fold'] == seed_data(s['seed_id'])['certificate']['fold']
        assert s['status'] == 'complete_slice'
        assert s['processed_eligible_partners'] == s['eligible_partners']
        assert not s['candidate_counts'].get('accepted', 0)
