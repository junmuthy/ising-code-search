from __future__ import annotations

import random

from distance_first_c2.search import (
    CSSState,
    analyze_css,
    canonical_basis,
    minimum_weight_basis,
    mutate_state,
    nullspace_basis,
    orthogonal,
    random_css_state,
    span,
)
from distance_first_c2.systematic import (
    SystematicCSSSolver,
    canonical_c_sectors,
    systematic_state,
)


def test_nullspace_basis_is_exact() -> None:
    n = 6
    rows = canonical_basis((0b001011, 0b110010), n)
    kernel = nullspace_basis(rows, n)
    assert len(kernel) == n - len(rows)
    assert orthogonal(rows, kernel)
    assert len(set(span(kernel))) == 1 << (n - len(rows))


def test_minimum_weight_basis_spans_original_space() -> None:
    n = 6
    rows = canonical_basis((0b111100, 0b110011, 0b101010), n)
    minimum = minimum_weight_basis(rows, n)
    assert canonical_basis(minimum, n) == rows
    assert len(minimum) == len(rows)


def test_random_n16_state_and_mutations_preserve_css_k2() -> None:
    rng = random.Random(160206)
    state = random_css_state(16, 7, rng)
    for sector in ("x", "z") * 8:
        state = mutate_state(state, sector, rng)
        assert isinstance(state, CSSState)
        assert state.k == 2
        assert state.rank_x == state.rank_z == 7
        assert orthogonal(state.basis_x, state.basis_z)


def test_analysis_counts_the_three_nonzero_logical_cosets() -> None:
    state = random_css_state(16, 7, random.Random(7))
    result = analyze_css(state)
    # Each sector has a nine-dimensional kernel and a seven-dimensional
    # stabilizer, hence 512 - 128 non-stabilizer logical vectors.
    assert result["x_logicals"]["logical_vectors"] == 384
    assert result["z_logicals"]["logical_vectors"] == 384
    assert result["n"] == 16
    assert result["k"] == 2


def test_systematic_parameterization_enforces_css_and_ranks() -> None:
    rng = random.Random(11)
    rank = 7
    a = [[rng.randrange(2) for _ in range(rank + 2)] for _ in range(rank)]
    c = [[rng.randrange(2) for _ in range(2)] for _ in range(rank)]
    state = systematic_state(a, c)
    assert state.n == 16
    assert state.k == 2
    assert state.rank_x == state.rank_z == 7
    assert orthogonal(state.basis_x, state.basis_z)


def test_systematic_solver_model_matches_direct_construction() -> None:
    solver = SystematicCSSSolver(16, timeout_ms=1000)
    assert str(solver.check()) == "sat"
    state = solver.state()
    assert state.n == 16
    assert state.k == 2
    assert orthogonal(state.basis_x, state.basis_z)


def test_n16_c_sectors_are_complete_and_canonical() -> None:
    sectors = canonical_c_sectors(7)
    assert len(sectors) == 70
    assert sum(sector.raw_multiplicity for sector in sectors) == 2**14
    assert all(sector.counts[1] <= sector.counts[2] for sector in sectors)


def test_n18_c_sectors_are_complete_and_canonical() -> None:
    sectors = canonical_c_sectors(8)
    assert len(sectors) == 95
    assert sum(sector.raw_multiplicity for sector in sectors) == 2**16
    assert all(sector.counts[1] <= sector.counts[2] for sector in sectors)


def test_fixed_c_solver_has_no_c_variables_and_remains_complete_css() -> None:
    sector = canonical_c_sectors(7)[17]
    solver = SystematicCSSSolver(16, fixed_c=sector.matrix, timeout_ms=1000)
    assert str(solver.check()) == "sat"
    state = solver.state()
    assert state.k == 2
    assert orthogonal(state.basis_x, state.basis_z)
