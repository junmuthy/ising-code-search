"""Controls for translation-orbit and exact coset logical optimization."""
import random

import numpy as np
import pytest

from distance8_weight10_search import (CosetEnumerator, audit_basis, export_basis,
    h_action, logical_shifts, make_catalog, seed_basis, unit_orbits)
from distance8_logical_handoff import simplify
from distance8_logical_search import coset_weights
from sparse_core import combine, permute, unpack


def test_complete_unit_orbits_and_free_translation_action():
    orbits, canonical = unit_orbits()
    assert len(orbits) == 2048
    assert set(canonical) == {i for i in range(65536) if i.bit_count()%2}
    for orbit in orbits:
        assert orbit['seed'] == min(orbit['members'])
        assert {permute(orbit['seed'], p) for p in logical_shifts()} == set(orbit['members'])
        assert all(canonical[v] == orbit['seed'] for v in orbit['members'])


def test_coset_scan_against_scalar_and_existing_enumerator():
    rows = [3, 12, 48, 192, 768]
    engine = CosetEnumerator(rows)
    for seed in (0, 341, 1023):
        expected = [combine(rows, c) ^ seed for c in range(32)]
        minimum = min(map(int.bit_count, expected))
        result, words = engine.scan(seed, collect_weight=minimum, chunk=3)
        assert result['minimum_weight'] == minimum
        assert result['minimum_word'] == min(w for w in expected if w.bit_count() == minimum)
        assert words == sorted(w for w in expected if w.bit_count() == minimum)
        old = coset_weights(rows, seed, width=10, chunk=5)
        assert result['histogram'] == old['histogram']


def test_saved_positive_control_and_unchanged_checks(tmp_path):
    data, _, _ = simplify()
    assert audit_basis(data, data, 12)['total_logical_support'] == 384
    x, z = seed_basis(data, 1)
    assert h_action(x, z, data['certificate']['fold']) == data['hadamard']['logical_permutation']
    report = export_basis(data, 1, data['certificate']['z'][0], data['certificate']['fold'], tmp_path/'positive')
    assert report['total_logical_support'] == 384


def test_logical_label_translations_match_physical_action():
    from distance8_weight10_search import physical_shifts
    from certify import in_space
    data, _, _ = simplify()
    rng = random.Random(2411)
    for _ in range(10):
        label = rng.randrange(65536)
        word = combine(data['certificate']['z'], label)
        for logical, physical in zip(logical_shifts(), physical_shifts(data)):
            a = permute(word, physical)
            b = combine(data['certificate']['z'], permute(label, logical))
            assert in_space(unpack([a^b], 64), unpack(data['hz'], 64))


def test_complete_catalog_with_independent_array_algebra():
    from distance8_weight10_audit import audit_catalog
    data, _, _ = simplify()
    catalog, _ = make_catalog(data)
    report = audit_catalog(data, catalog)
    assert report['compatible_orbits'] == 256
    assert report['compatible_unit_classes'] == 4096
    catalog[0]['saved_hadamard_permutation'] = None
    with pytest.raises(AssertionError):
        audit_catalog(data, catalog)


def test_weight_ten_finalist_and_balancing_controls():
    import json
    from distance8_weight10_finalize import balance_options
    from sparse_core import HERE
    data, _, _ = simplify()
    root = HERE/'distance8_code_study/weight_ten_handoff_v1'
    candidate = json.loads((root/'best/candidate.json').read_text())
    report = json.loads((root/'report.json').read_text())
    checked = audit_basis(candidate, data, 10)
    assert checked['combined_qubit_loads'] == [5]*64
    assert checked['maximum_pairwise_support_overlap'] == 4
    assert len(report['ranking']) == 14 and report['selected_seed'] == 7
    words = candidate['certificate']['z']
    selected, score = balance_options([[w] for w in words], data['certificate']['fold'], trials=1)
    assert selected == words and score['score'] == [5, 4, 1600]
    candidate['certificate']['x'][0] ^= 1
    with pytest.raises(AssertionError):
        audit_basis(candidate, data, 10)
