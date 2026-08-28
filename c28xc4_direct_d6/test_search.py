"""Regression checks for the direct ``C28 x C4`` construction."""

from __future__ import annotations

import numpy as np

from algebra import (
    CHECK_RANK,
    GROUP_ORDER,
    fibre_logicals,
    gf8_rref,
    iter_monomial_ideal_thresholds,
    monomial_ideal_basis,
    quotient_annihilator_basis,
    build_stabilizer_basis,
    validate_stabilizer,
)


def test_catalog_has_seventy_monomial_ideals() -> None:
    assert len(list(iter_monomial_ideal_thresholds())) == 70


def test_fibres_form_disjoint_c4xc4_grid() -> None:
    fibres = fibre_logicals()
    assert fibres.shape == (16, GROUP_ORDER)
    assert np.all(fibres.sum(axis=1) == 7)
    assert np.max(fibres.sum(axis=0)) == 1
    assert np.array_equal(fibres @ fibres.T % 2, np.eye(16, dtype=np.uint8))


def test_every_monomial_construction_has_fixed_properties() -> None:
    for thresholds in iter_monomial_ideal_thresholds():
        ideal = monomial_ideal_basis(thresholds)
        annihilator = quotient_annihilator_basis(list(ideal))
        check = build_stabilizer_basis(ideal, annihilator)
        properties = validate_stabilizer(check)
        assert len(gf8_rref(ideal)[0]) + len(gf8_rref(annihilator)[0]) == 16
        assert properties["rank"] == CHECK_RANK
        assert properties["self_orthogonal"]
        assert properties["fibres_commute"]
        assert properties["fibres_orthonormal"]
        assert properties["fibres_complete"]
