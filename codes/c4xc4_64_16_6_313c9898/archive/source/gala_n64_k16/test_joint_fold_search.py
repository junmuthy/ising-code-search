"""Independent controls for the joint coefficient/fold coverage argument."""
import itertools
import json
import random

import numpy as np
import pytest

from joint_fold_search import (GROUP, SEEDS, affine_folds, assess_eight, assess_six,
                               automorphisms, canonical_translation, checks,
                               configurations, export_six, gauge_fold, load_prior,
                               make_manifest, normalize_pair, seed_data,
                               transform_pair, translate)
from algebra import bits, combine, commute, compose, permute, rank, same_space, unpack
from sparse_twisted import fold_catalog


@pytest.fixture(scope='module')
def manifest():
    return make_manifest(6)


def test_weight_reductions_and_full_shape_coverage(manifest):
    from certify import gf_rank
    assert manifest['shape_classes'] == 17
    assert manifest['surviving_translation_classes'] == 388
    assert manifest['translation_classes'] == 1138
    assert manifest['raw_slices'] == len(list(configurations(manifest))) == 3264
    classes = {canonical_translation(sum(1 << q for q in support))
               for support in itertools.combinations(range(32), 4) if 0 in support}
    covered = {int(v) for v in manifest['obstructions']}
    for orbit in manifest['orbits']:
        assert not covered & set(orbit['translated_classes'])
        covered.update(orbit['translated_classes'])
    assert covered == classes
    # Every unanchored support translates to an anchored one. Also recover
    # the full finite support count explicitly, independent of that argument.
    all_supports = {translate(w, g) for w in covered for g in range(32)}
    assert len(all_supports) == 35960
    assert all(w.bit_count() == 4 for w in all_supports)
    proofs = {**manifest['obstructions'], **manifest['zero_and_weight_two_obstructions']}
    for word, proof in proofs.items():
        matrix = unpack(GROUP.lift(int(word)), 32)
        r = gf_rank(matrix)
        assert proof['fixed_rank'] == r
        if proof['status'] == 'fixed_rank_obstruction':
            assert r > 24
        else:
            light = unpack(proof['light_kernel_rows'], 32)
            assert not np.any(matrix @ light.T % 2)
            assert np.all(light.sum(axis=1) <= 5)
            assert gf_rank(light) > 24-r
    for word, norm in manifest['normalizers'].items():
        assert translate(permute(int(word), norm['automorphism']),
                         GROUP.inverse[norm['fixed_sheet_shift']]) == norm['representative']


def test_odd_coefficients_have_explicit_inverse():
    rng = random.Random(2307)
    for _ in range(50):
        word = rng.getrandbits(32)
        if word.bit_count() % 2 == 0:
            word ^= 1
        squared = GROUP.polynomial_mul(word, word)
        fourth = GROUP.polynomial_mul(squared, squared)
        assert GROUP.polynomial_mul(fourth, fourth) == 1
        inverse = GROUP.polynomial_mul(word, GROUP.polynomial_mul(squared, fourth))
        assert GROUP.polynomial_mul(word, inverse) == 1
        assert rank(GROUP.lift(word)) == 32


def test_affine_fold_closure_and_all_gauge_reductions():
    catalog = {tuple(p) for p in fold_catalog()}
    folds = list(affine_folds())
    assert len({tuple(p) for p in folds}) == len(folds) == 3072
    assert catalog <= {tuple(p) for p in folds}
    rng = random.Random(2308)
    for index, p in enumerate(folds):
        assert sorted(p) == list(range(64))
        square = compose(p, p)
        assert square == list(range(64)) or square == GROUP.physical()[2]
        for side in ('a', 'b'):
            a, b = rng.getrandbits(32), rng.getrandbits(32)
            aa, bb, pp, q = gauge_fold(a, b, p, side)
            assert tuple(pp) in catalog
            assert (aa if side == 'a' else bb) == (a if side == 'a' else b)
            if index % 31 == 0:
                hx, hz = checks(a, b, p)
                xx, zz = checks(aa, bb, pp)
                assert same_space([permute(v, q) for v in hx], xx)
                assert same_space([permute(v, q) for v in hz], zz)
    # All simultaneous group automorphisms preserve the normalized catalog,
    # including the designated central element and induced quotient group.
    for alpha in automorphisms():
        assert alpha[16] == 16
        for p in catalog:
            _, _, pp, _ = transform_pair(1, 1, p, alpha)
            assert tuple(pp) in catalog


def test_normalization_of_every_surviving_shape(manifest):
    rng = random.Random(2309)
    folds = list(affine_folds())
    for word in manifest['normalizers']:
        word = translate(int(word), rng.randrange(32))
        side = rng.choice(('a', 'b'))
        other = rng.getrandbits(32)
        a, b = (word, other) if side == 'a' else (other, word)
        p = rng.choice(folds)
        aa, bb, pp, q = normalize_pair(a, b, p, side, manifest)
        hx, hz = checks(a, b, p)
        xx, zz = checks(aa, bb, pp)
        assert same_space([permute(v, q) for v in hx], xx)
        assert same_space([permute(v, q) for v in hz], zz)
        assert aa.bit_count() == a.bit_count() and bb.bit_count() == b.bit_count()


@pytest.mark.parametrize('key', SEEDS)
def test_transformed_distance_eight_positive_controls(key, manifest):
    data = seed_data(key)
    alpha = automorphisms()[53]
    a, b, fold, q = transform_pair(data['a'], data['b'], data['certificate']['fold'], alpha, (7, 13))
    if a.bit_count() == 4:
        a, b, fold, _ = normalize_pair(a, b, fold, 'a', manifest)
    else:
        a, b, fold, _ = gauge_fold(a, b, fold, 'b')
    hx, hz = checks(a, b, fold)
    result, certificate = assess_eight(hx, hz, fold)
    assert result['status'] == 'accepted'
    assert result['distance_lower_bound'] == 8
    assert certificate is not None
    result, _ = assess_six(hx, hz, fold)
    assert result['status'] == 'accepted'


def test_distance_six_is_now_accepted_and_exported(tmp_path):
    from joint_fold_search import HERE
    path = HERE/'sparse_campaign/twisted_w12_v1/certified/d9173bb8cd878461c62a0dd5b7f9cb5f494b401aaf662cf40e45eb883551ebcf/candidate.json'
    data = json.loads(path.read_text())
    result, certificate = assess_six(data['hx'], data['hz'], data['certificate']['fold'])
    assert result['status'] == 'accepted'
    strict, _ = assess_eight(data['hx'], data['hz'], data['certificate']['fold'])
    assert strict['status'] == 'logical_weight_six'
    data['certificate'] = certificate
    audit = export_six(data, tmp_path/'distance_six_positive')
    assert all(audit['distance'][s]['lower_bound'] == audit['distance'][s]['upper_bound'] == 6
               for s in ('X', 'Z'))
    assert (tmp_path/'distance_six_positive/weights.json').exists()


def test_prior_distance_floor_is_not_silently_relaxed(tmp_path):
    from joint_fold_search import HERE
    old = HERE/'distance8_weight_search/shape_pilot_w10_v1'
    assert len(load_prior([old], 6, 10)) == 60
    unsafe = tmp_path/'unsafe'
    unsafe.mkdir()
    (unsafe/'summary.json').write_text(json.dumps(dict(parameters=dict(maxweight=10, min_distance=8),
                                                     candidate_counts=dict(logical_weight_six=1))))
    assert load_prior([unsafe], 6, 10) == {}


def test_even_multiplier_submodule_is_proper():
    # A low-weight independent generating set must contain an odd-multiplier
    # row, whose entire translation orbit also spans the code's check space.
    rng = random.Random(2310)
    for key in SEEDS:
        data = seed_data(key)
        hx = data['hx']
        px, py, _ = GROUP.physical()
        radical = [v ^ permute(v, p) for p in (px, py) for v in hx]
        assert rank(hx) == 24 and rank(radical) == 23
        for _ in range(20):
            coefficients = rng.getrandbits(32)
            word = combine(hx, coefficients)
            orbit = [permute(word, GROUP.right(g)) for g in range(32)]
            if coefficients.bit_count() % 2:
                assert same_space(orbit, hx)
            else:
                assert rank(orbit) < 24


@pytest.mark.parametrize('sealed', [False, True])
def test_audit_prefix_reuse_requires_matching_journal_hash(tmp_path, monkeypatch, manifest, sealed):
    import audit_joint_fold as verifier
    from joint_fold_search import HERE, digest
    source = HERE/'joint_fold_campaign/pilot_d6_w10_v1'
    first = json.loads((source/'slices.jsonl').read_text().splitlines()[0])
    records = []
    with (source/'candidates.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['slice_id'] != first['slice_id']:
                break
            records.append(row)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    late = dict(first, slice_id=first['slice_id']+'_late')
    (tmp_path/'slices.jsonl').write_text(json.dumps(first)+'\n'+(json.dumps(late)+'\n' if sealed else ''))
    candidate_text = ''.join(json.dumps(r)+'\n' for r in records)
    if sealed:
        candidate_text += json.dumps(dict(records[0], slice_id=late['slice_id']))+'\n'
    (tmp_path/'candidates.jsonl').write_text(candidate_text)
    (tmp_path/'summary.json').write_text(json.dumps(dict(
        status='interrupted_checkpoint' if sealed else 'complete_joint_scope',
        parameters=dict(min_distance=6, maxweight=10), configurations_processed=1,
        candidate_journal_completed_prefix_rows=len(records) if sealed else None,
        manifest_sha256=digest(manifest))))
    # Coverage normalization has its own exhaustive tests. Isolate a single
    # genuine slice here to exercise the audit's reuse and integrity boundary.
    monkeypatch.setattr(verifier, 'configurations', lambda m: [first])
    original = verifier.audit(tmp_path, allow_partial=sealed)
    assert original['complete_partner_sets'] == 1
    assert sum(original['candidate_counts'].values()) == len(records)
    proof = tmp_path/'previous.json'
    proof.write_text(json.dumps(original))
    repeated = verifier.audit(tmp_path, allow_partial=sealed, previous_checks=proof)
    assert repeated['reused_verified_journal_rows'] == len(records)
    assert not repeated['rejection_certificates_rechecked']
    records[0]['witness'] = [63]
    (tmp_path/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(AssertionError):
        verifier.audit(tmp_path, allow_partial=sealed, previous_checks=proof)


def test_shard_worker_restores_configuration_builder(tmp_path):
    import joint_fold_parallel as runner
    original = runner.search.configurations
    parameters = dict(min_distance=6, maxweight=10, seconds=30, max_slices=10000)
    for index in (0, 1):
        result = runner.worker(parameters, [index], str(tmp_path/f'shard_{index}'))
        assert result['status'] == 'complete_configured_shard'
        assert result['configurations'] == result['processed'] == 1
        assert runner.search.configurations is original


def test_equivalence_continuation_respects_sealed_slice_prefix(tmp_path, manifest):
    from joint_equivalence_complete import direct_certificates
    configs = list(configurations(manifest))
    rows = [dict(c, status='complete_slice', slice_id=str(i), eligible_partners=0,
                 processed_eligible_partners=0, partner_weight_ceiling=6, candidate_counts={})
            for i, c in enumerate(configs[:2])]
    (tmp_path/'slices.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='interrupted_checkpoint', configurations_processed=1)))
    direct = direct_certificates([tmp_path], configs)
    assert set(direct) == {0}


def test_all_joint_orbit_maps_transport_checks_and_weights(manifest):
    from joint_fold_search import HERE, config_key
    from joint_fold_orbits import inverse, transport
    artifact = json.loads((HERE/'joint_fold_campaign/joint_orbits_d6_v1.json').read_text())
    configs = list(configurations(manifest))
    assert artifact['orbits'] == 1092
    assert artifact['configuration_keys'] == [config_key(c) for c in configs]
    members = [n for o in artifact['orbit_records'] for n in o['members']]
    assert sorted(members) == list(range(3264))
    auts = set(automorphisms())
    rng = random.Random(2311)
    for orbit in artifact['orbit_records']:
        root = configs[orbit['representative']]
        other = rng.getrandbits(32)
        a, b = (root['fixed'], other) if root['fixed_side'] == 'a' else (other, root['fixed'])
        hx, hz = checks(a, b, root['fold'])
        for index in orbit['members']:
            q = orbit['root_to_member'][str(index)]
            c = configs[index]
            assert sorted(q) == list(range(64))
            alpha = tuple(GROUP.mul(q[g] % 32, GROUP.inverse[q[0] % 32]) for g in range(32))
            assert alpha in auts
            aa, bb, pp = transport(a, b, root['fold'], q)
            assert pp == c['fold']
            assert (aa if c['fixed_side'] == 'a' else bb) == c['fixed']
            assert (bb if c['fixed_side'] == 'a' else aa).bit_count() == other.bit_count()
            xx, zz = checks(aa, bb, pp)
            assert same_space([permute(v, q) for v in hx], xx)
            assert same_space([permute(v, q) for v in hz], zz)
    positive = seed_data(SEEDS[0])
    a, b, p, _ = normalize_pair(positive['a'], positive['b'], positive['certificate']['fold'], 'a', manifest)
    node = next(i for i, c in enumerate(configs) if c['fixed']==a and c['fixed_side']=='a' and c['fold']==p)
    orbit = next(o for o in artifact['orbit_records'] if node in o['members'])
    target = orbit['members'][-1]
    q = compose(orbit['root_to_member'][str(target)], inverse(orbit['root_to_member'][str(node)]))
    a, b, p = transport(a, b, p, q)
    result, _ = assess_eight(*checks(a, b, p), p)
    assert result['status'] == 'accepted'
