"""Small independent exhaustive controls for fixed-code logical searches."""
import itertools
import random

import pytest

from distance8_logical_search import all_weight_eight, check_equivariant_bases, coset_weights, run
from sparse_core import combine


@pytest.mark.parametrize('seed', [15, 16, 17])
def test_weight_eight_enumeration_against_bruteforce(seed):
    rng = random.Random(seed)
    checks = [rng.getrandbits(12) for _ in range(3)]
    expected = []
    for support in itertools.combinations(range(12), 8):
        word = sum(1 << q for q in support)
        if all((row & word).bit_count()%2 == 0 for row in checks):
            expected.append(word)
    assert all_weight_eight(checks, 12) == sorted(expected)


def test_coset_enumeration_against_scalar():
    rows = [3, 12, 48, 192, 768]
    representative = 341
    words = [combine(rows, c) ^ representative for c in range(32)]
    result = coset_weights(rows, representative, width=10, chunk=3)
    assert result['minimum_weight'] == min(map(int.bit_count, words))
    assert result['minimum_word'] == min(words, key=lambda v: (v.bit_count(), v))
    assert result['histogram'] == {w: sum(v.bit_count()==w for v in words) for w in set(map(int.bit_count, words))}


def test_complete_fixed_code_catalog_and_equivariant_basis_obstruction(tmp_path):
    catalog = run(tmp_path/'catalog')
    assert catalog['fixed_basis_single_logical_coset']['enumerated'] == 2**24
    assert catalog['fixed_basis_all_32_single_logical_minimum'] == 12
    assert catalog['weight_eight_pure_Z_operators'] == catalog['weight_eight_pure_X_operators'] == 1016
    assert catalog['weight_eight_Z_logical_classes'] == 414
    assert catalog['weight_eight_Z_logical_span_rank'] == 16
    grid = check_equivariant_bases(tmp_path/'catalog', tmp_path/'grid')
    assert grid['seed_classes_tested'] == 192
    assert grid['all_weight_eight_canonical_grids'] == 0
    assert grid['saved_H_compatible_seeds'] == 0
