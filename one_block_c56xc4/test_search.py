import numpy as np

from algebra import QUOTIENT_ORDER, quotient_ideal_basis
from search import (
    IdealCandidate,
    analyze_candidate,
    build_sparse_catalog,
    enumerate_ideal_candidates,
)


def test_catalog_and_candidate_counts():
    catalog = build_sparse_catalog()
    counts = {}
    for entry in catalog:
        counts[entry.polynomial.family] = counts.get(entry.polynomial.family, 0) + 1
    assert counts == {"monomial": 1, "binomial": 17, "trinomial": 930}
    candidates, raw = enumerate_ideal_candidates(catalog)
    assert raw == {
        "monomial_ideal": 495,
        "monomial": 1,
        "binomial": 17,
        "trinomial": 930,
    }
    assert len(candidates) <= sum(raw.values())


def test_uncoupled_distance_three():
    catalog = build_sparse_catalog()
    unit = np.zeros(QUOTIENT_ORDER, dtype=np.uint8)
    unit[0] = 1
    ideal = quotient_ideal_basis([unit])
    candidate = IdealCandidate(
        "uncoupled",
        ideal,
        [{"family": "baseline", "description": "full plus sector"}],
    )
    record = analyze_candidate(candidate, catalog, exact_distance=False)
    assert record["n"] == 224
    assert record["k"] == 32
    assert record["distance"] == 3
    assert record["check_weight"] == 4
    assert not record["structural_checks"]["tanner_connected"]


if __name__ == "__main__":
    test_catalog_and_candidate_counts()
    test_uncoupled_distance_three()
    print("ok")
