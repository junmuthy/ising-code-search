import numpy as np

from algebra import (
    COEFFICIENT_WORDS,
    QUOTIENT_ORDER,
    build_stabilizer_basis,
    fibre_logicals,
    gf8_rref,
    iter_monomial_ideal_thresholds,
    iter_sparse_polynomials,
    monomial_ideal_basis,
    quotient_annihilator_basis,
    quotient_ideal_basis,
    validate_stabilizer,
)


def test_counts():
    assert len(list(iter_monomial_ideal_thresholds())) == 495
    sparse = list(iter_sparse_polynomials())
    assert sparse[0].family == "monomial"
    assert any(item.family == "binomial" for item in sparse)
    assert any(item.family == "trinomial" for item in sparse)


def test_simplex_words():
    assert np.count_nonzero(COEFFICIENT_WORDS[0]) == 0
    assert all(np.count_nonzero(COEFFICIENT_WORDS[value]) == 4 for value in range(1, 8))
    assert not np.any((COEFFICIENT_WORDS @ COEFFICIENT_WORDS.T) % 2)


def test_uncoupled_baseline():
    unit = np.zeros(QUOTIENT_ORDER, dtype=np.uint8)
    unit[0] = 1
    ideal = quotient_ideal_basis([unit])
    annihilator = quotient_annihilator_basis([unit])
    assert len(ideal) == 32
    assert len(annihilator) == 0
    stabilizer = build_stabilizer_basis(ideal, annihilator)
    assert validate_stabilizer(stabilizer) == {
        "rank": 96,
        "self_orthogonal": True,
        "fibres_commute": True,
        "fibres_orthonormal": True,
        "fibres_complete": True,
    }
    assert np.all(np.count_nonzero(fibre_logicals(), axis=1) == 7)


def test_monomial_ideal_dimensions():
    for thresholds in list(iter_monomial_ideal_thresholds())[::31]:
        ideal = monomial_ideal_basis(thresholds)
        generators = []
        # The RREF basis itself is a valid generating set for annihilator tests.
        annihilator = quotient_annihilator_basis(list(ideal))
        assert len(ideal) + len(annihilator) == 32


if __name__ == "__main__":
    test_counts()
    test_simplex_words()
    test_uncoupled_baseline()
    test_monomial_ideal_dimensions()
    print("ok")
