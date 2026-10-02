import numpy as np

from component_code import (
    check_seed_polynomials,
    check_space_basis,
    logical_fibres,
    qldpc_code,
    translated_check_catalog,
    validate,
)


def test_code_properties():
    assert [int(seed.sum()) for seed in check_seed_polynomials()] == [16, 16, 16]
    rows, _metadata, orbit_sizes = translated_check_catalog()
    assert orbit_sizes == (28, 28, 112)
    assert len(rows) == 168
    basis = check_space_basis()
    properties = validate(basis)
    assert properties == {
        "n": 112,
        "check_rank": 48,
        "k": 16,
        "self_orthogonal": True,
        "exact_zx_self_dual": True,
        "doubly_even_check_space": True,
        "logical_count": 16,
        "logical_weights": [7],
        "logical_gram_identity": True,
        "logicals_commute_with_checks": True,
        "logicals_independent_mod_checks": 16,
    }
    assert np.all(logical_fibres().sum(axis=1) == 7)
    code = qldpc_code(basis)
    assert len(code) == 112
    assert code.dimension == 16


if __name__ == "__main__":
    test_code_properties()
    print("ok")
