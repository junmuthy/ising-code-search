"""Tests for the batched ``C4 x C2`` GALA search."""

from __future__ import annotations

import numpy as np
from qldpc import codes

from searches.batched_c4xc2_search.distance import (
    find_logical_milp,
    find_logical_through_weight_four,
    low_weight_logical_spectrum,
)
from searches.batched_c4xc2_search.halfswap_model import (
    HalfSwapCandidate,
    build_half_swap_code,
    inverse_orbits,
    random_half_swap_candidates,
)
from searches.batched_c4xc2_search.faithful_twisted_halfswap_model import (
    FaithfulTwistedHalfSwapCandidate,
    faithful_commutant_g_terms,
    faithful_twisted_check_matrices,
    faithful_twisted_transpose_orbits,
    faithful_x_reflection_permutation,
    random_faithful_commutant_candidates,
    random_faithful_twisted_half_swap_candidates,
)
from searches.batched_c4xc2_search.faithful_twisted_protograph_model import (
    FaithfulTwistedProtographCandidate,
    faithful_twisted_protograph_check_matrices,
    random_faithful_twisted_protograph_candidates,
)
from searches.batched_c4xc2_search.analyze_published_bb64_logicals import (
    orbit as published_bb64_orbit,
    paper_logical,
)
from searches.batched_c4xc2_search.analyze_published_selfdual_bb64 import (
    affine_permutation as published_bb64_affine_permutation,
    full_permutation as published_bb64_full_permutation,
    published_code as published_bb64_code,
    translation_permutation as published_bb64_translation_permutation,
)
from searches.batched_c4xc2_search.logicals import (
    batch_disjointness,
    batch_schemes,
    exhaustive_logical_seeds,
    logical_action_profile,
    translation_algebra_profile,
    translation_orbit,
)
from searches.batched_c4xc2_search.model import (
    Candidate,
    Monomial,
    _actual_transpose_checks,
    analyze_structure,
    binary_transpose_terms,
    build_code,
    candidate_generators,
    gf2_rank,
    quotient_s3_seed,
    random_candidates,
    zx_fold_permutation,
)
from searches.batched_c4xc2_search.twisted_halfswap_model import (
    TwistedHalfSwapCandidate,
    build_twisted_half_swap_code,
    random_twisted_half_swap_candidates,
    sample_twisted_double_neighbors,
    twisted_support_neighbors,
    twisted_support_orbits,
    twisted_transpose_orbits,
    x_reflection_permutation,
)


def test_all_batch_schemes_partition_the_grid() -> None:
    schemes = batch_schemes()
    assert len(schemes) == 6
    assert sorted(len(batches) for batches in schemes.values()) == [2, 2, 2, 4, 4, 4]
    for batches in schemes.values():
        flattened = [site for batch in batches for site in batch]
        assert sorted(flattened) == list(range(8))
        assert len(flattened) == len(set(flattened))


def test_single_site_orbit_is_a_clean_c4xc2_grid() -> None:
    mm = Monomial
    candidate = Candidate(
        top_representation="trivial",
        entries=(
            tuple(sorted((mm(0, 0, 0), mm(0, 1, 0)))),
            tuple(sorted((mm(0, 0, 1), mm(0, 1, 1)))),
        ),
    )
    seed = np.zeros(candidate.num_qubits, dtype=np.uint8)
    seed[0] = 1
    orbit = translation_orbit(candidate, seed)
    assert orbit.shape == (8, 32)
    assert np.all(np.count_nonzero(orbit, axis=1) == 1)
    assert np.all(np.sum(orbit, axis=0) <= 1)
    assert all(
        result["disjoint"] for result in batch_disjointness(orbit).values()
    )


def test_quotient_seed_is_connected_exact_zx_weight16() -> None:
    candidate = quotient_s3_seed()
    analysis = analyze_structure(candidate)
    assert analysis["accepted"]
    assert (analysis["n"], analysis["k"]) == (96, 52)
    assert analysis["check_weights_x"] == analysis["check_weights_z"] == [16]
    assert analysis["column_degrees_x"] == analysis["column_degrees_z"] == [4]
    assert analysis["checks"]["strict_ZX_fold"]
    assert analysis["zx_fold_is_identity"]


def test_quotient_seed_has_a_weight_two_logical() -> None:
    result = find_logical_milp(
        build_code(quotient_s3_seed()), pauli="Z", maximum_weight=5
    )
    assert result["weight"] == 2
    assert result["support"] is not None


def test_random_candidate_stream_is_reproducible_and_weighted() -> None:
    kwargs = {
        "top_representation": "s3-natural",
        "entry_weights": (3, 3),
        "count": 10,
        "seed": 17,
    }
    first = list(random_candidates(**kwargs))
    second = list(random_candidates(**kwargs))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(candidate.nominal_check_weight == 12 for candidate in first)


def test_l2_candidates_have_the_expected_lengths_and_identity_fold() -> None:
    for representation, expected_n in (("s3-linear", 32), ("s3-natural", 48)):
        candidate = next(
            random_candidates(
                top_representation=representation,
                entry_weights=(4,),
                count=1,
                seed=23,
            )
        )
        assert candidate.num_blocks == 2
        assert candidate.num_qubits == expected_n
        assert np.array_equal(
            zx_fold_permutation(candidate), np.arange(expected_n)
        )
        ff, gg = candidate_generators(candidate)
        if representation == "s3-natural":
            code = build_code(candidate, skip_validation=True)
            hx = np.asarray(code.matrix_x, dtype=np.uint8)
            hz = np.asarray(code.matrix_z, dtype=np.uint8)
        else:
            hx, hz = _actual_transpose_checks(ff, gg)
        assert np.array_equal(hx, hz)


def test_faithful_s3_uses_actual_binary_transpose_and_n64() -> None:
    mm = Monomial
    candidate = Candidate(
        top_representation="s3-linear",
        entries=(
            tuple(sorted((mm(0, 0, 0), mm(1, 1, 0)))),
            tuple(sorted((mm(2, 0, 1), mm(3, 1, 1)))),
        ),
    )
    hx, hz = _actual_transpose_checks(*candidate_generators(candidate))
    assert candidate.num_qubits == hx.shape[1] == 64
    assert np.array_equal(hx, hz)


def test_nonidentity_fold_swaps_protograph_blocks_for_both_s3_lifts() -> None:
    mm = Monomial
    entries = (
        tuple(sorted((mm(0, 0, 0), mm(1, 1, 0)))),
        tuple(sorted((mm(2, 0, 1), mm(3, 1, 1)))),
    )
    for representation in ("s3-natural", "s3-linear"):
        candidate = Candidate(
            top_representation=representation,
            entries=entries,
            fold_axis=1,
        )
        ff, gg = candidate_generators(candidate)
        if representation == "s3-natural":
            code = build_code(candidate, skip_validation=True)
            hx = np.asarray(code.matrix_x, dtype=np.uint8)
            hz = np.asarray(code.matrix_z, dtype=np.uint8)
        else:
            hx, hz = _actual_transpose_checks(ff, gg)
        fold = zx_fold_permutation(candidate)
        folded_hx = np.zeros_like(hx)
        folded_hx[:, fold] = hx
        assert np.array_equal(folded_hx, hz)
        assert not np.array_equal(fold, np.arange(candidate.num_qubits))


def test_l6_fold_uses_opposite_reflection_axes_on_the_two_halves() -> None:
    mm = Monomial
    entries = (
        tuple(sorted((mm(0, 0, 0), mm(1, 1, 0)))),
        tuple(sorted((mm(2, 0, 1), mm(3, 1, 1)))),
        tuple(sorted((mm(4, 2, 0), mm(5, 3, 1)))),
    )
    for representation in ("s3-natural", "s3-linear"):
        for axis in range(3):
            candidate = Candidate(
                top_representation=representation,
                entries=entries,
                fold_axis=axis,
            )
            ff, gg = candidate_generators(candidate)
            if representation == "s3-natural":
                code = build_code(candidate, skip_validation=True)
                hx = np.asarray(code.matrix_x, dtype=np.uint8)
                hz = np.asarray(code.matrix_z, dtype=np.uint8)
            else:
                hx, hz = _actual_transpose_checks(ff, gg)
            fold = zx_fold_permutation(candidate)
            folded_hx = np.zeros_like(hx)
            folded_hx[:, fold] = hx
            assert np.array_equal(folded_hx, hz)


def test_distance_milp_distinguishes_a_logical_from_a_stabilizer() -> None:
    code = codes.CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]])
    result = find_logical_milp(code, pauli="Z", maximum_weight=1)
    assert result["weight"] == 1
    assert result["support"][0] in {2, 3}


def test_syndrome_hash_matches_the_weight_one_logical() -> None:
    code = codes.CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]])
    result = find_logical_through_weight_four(code, pauli="Z")
    assert result["weight"] == 1
    assert result["support"][0] in {2, 3}


def test_low_weight_spectrum_counts_all_minimum_logicals() -> None:
    code = codes.CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]])
    result = low_weight_logical_spectrum(code, maximum_weight=4)
    assert result["minimum_logical_weight_found"] == 1
    assert result["counts"]["1"] == 2
    assert result["witnesses"]["1"][0] in {2, 3}


def test_small_logical_space_is_enumerated_exactly() -> None:
    code = codes.CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]])
    seeds = list(exhaustive_logical_seeds(code))
    assert len(seeds) == 3
    classes = {np.packbits(seed).tobytes() for seed in seeds}
    assert len(classes) == 3
    candidate = Candidate(
        top_representation="trivial",
        entries=((Monomial(0, 0, 0),),),
    )
    checks = np.zeros((7, 16), dtype=np.uint8)
    checks[np.arange(7), np.arange(7)] = 1
    checks[np.arange(7), 7 + np.arange(7)] = 1
    profile = logical_action_profile(codes.CSSCode(checks, checks), candidate)
    assert profile["exact"]
    assert profile["tested_classes"] == 3


def test_halfswap_inverse_orbits_partition_all_monomials() -> None:
    fixed, pairs = inverse_orbits()
    flattened = [*fixed, *(term for pair in pairs for term in pair)]
    assert len(flattened) == len(set(flattened)) == 48


def test_halfswap_stream_is_reproducible_inverse_closed_and_weighted() -> None:
    kwargs = {"f_weight": 6, "g_weight": 6, "count": 10, "seed": 41}
    first = list(random_half_swap_candidates(**kwargs))
    second = list(random_half_swap_candidates(**kwargs))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(candidate.nominal_check_weight == 12 for candidate in first)
    assert all(Monomial(0, 0, 0) in candidate.f_terms for candidate in first)


def test_independent_symmetric_fg_has_exact_nonidentity_halfswap_fold() -> None:
    mm = Monomial
    support = tuple(sorted((mm(0, 0, 0), mm(0, 2, 0))))
    candidate = HalfSwapCandidate(support, support)
    code = build_half_swap_code(candidate)
    fold = zx_fold_permutation(candidate)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    folded_hx = np.zeros_like(hx)
    folded_hx[:, fold] = hx
    assert candidate.num_qubits == 48
    assert not np.array_equal(fold, np.arange(48))
    assert np.array_equal(folded_hx, hz)


def test_twisted_transpose_orbits_partition_all_monomials() -> None:
    fixed, pairs = twisted_transpose_orbits()
    flattened = [*fixed, *(term for pair in pairs for term in pair)]
    assert len(flattened) == len(set(flattened)) == 48
    assert len(fixed) > len(pairs)


def test_twisted_stream_is_reproducible_and_weighted() -> None:
    kwargs = {"f_weight": 6, "g_weight": 6, "count": 10, "seed": 53}
    first = list(random_twisted_half_swap_candidates(**kwargs))
    second = list(random_twisted_half_swap_candidates(**kwargs))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(candidate.nominal_check_weight == 12 for candidate in first)
    assert all(Monomial(0, 0, 0) in candidate.f_terms for candidate in first)


def test_x_reflected_halfswap_is_an_exact_broader_zx_fold() -> None:
    mm = Monomial
    support = tuple(sorted((mm(0, 0, 0), mm(0, 1, 0))))
    candidate = TwistedHalfSwapCandidate(support, support)
    code = build_twisted_half_swap_code(candidate)
    fold = zx_fold_permutation(candidate)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    folded_hx = np.zeros_like(hx)
    folded_hx[:, fold] = hx
    assert candidate.num_qubits == 48
    assert np.array_equal(folded_hx[x_reflection_permutation()], hz)
    assert support != binary_transpose_terms("s3-natural", support)
    algebra = translation_algebra_profile(code, candidate)
    assert algebra["group_relations_verified"]
    assert 1 <= algebra["dimension"] <= 8


def test_twisted_neighbors_preserve_weights_fold_and_f_anchor() -> None:
    candidate = next(
        random_twisted_half_swap_candidates(
            f_weight=6, g_weight=10, count=1, seed=67
        )
    )
    neighbors = twisted_support_neighbors(candidate)
    for _ in range(20):
        neighbor, mutation = next(neighbors)
        assert len(neighbor.f_terms) == len(candidate.f_terms)
        assert len(neighbor.g_terms) == len(candidate.g_terms)
        assert Monomial(0, 0, 0) in neighbor.f_terms
        assert mutation["side"] in {"F", "G"}


def test_twisted_double_neighbors_are_reproducible_exact_radius_two() -> None:
    candidate = next(
        random_twisted_half_swap_candidates(
            f_weight=6, g_weight=10, count=1, seed=71
        )
    )
    first = list(sample_twisted_double_neighbors(candidate, count=25, seed=73))
    second = list(sample_twisted_double_neighbors(candidate, count=25, seed=73))
    assert [item[0] for item in first] == [item[0] for item in second]
    assert len(first) == 25
    assert len({item[0].candidate_id for item in first}) == 25

    orbit_lookup = {
        term: orbit_id
        for orbit_id, orbit in enumerate(twisted_support_orbits())
        for term in orbit
    }
    original = {
        "F": {orbit_lookup[term] for term in candidate.f_terms},
        "G": {orbit_lookup[term] for term in candidate.g_terms},
    }
    for neighbor, mutation in first:
        updated = {
            "F": {orbit_lookup[term] for term in neighbor.f_terms},
            "G": {orbit_lookup[term] for term in neighbor.g_terms},
        }
        assert sum(len(original[side] - updated[side]) for side in ("F", "G")) == 2
        assert sum(len(updated[side] - original[side]) for side in ("F", "G")) == 2
        assert len(neighbor.f_terms) == len(candidate.f_terms)
        assert len(neighbor.g_terms) == len(candidate.g_terms)
        assert Monomial(0, 0, 0) in neighbor.f_terms
        assert mutation["kind"] == "coordinated_two_orbit"

    code = build_twisted_half_swap_code(first[0][0], skip_validation=True)
    fold = zx_fold_permutation(first[0][0])
    folded_hx = np.zeros_like(code.matrix_x)
    folded_hx[:, fold] = code.matrix_x
    assert np.array_equal(folded_hx[x_reflection_permutation()], code.matrix_z)


def test_faithful_twisted_orbits_partition_all_monomials() -> None:
    fixed, pairs = faithful_twisted_transpose_orbits()
    flattened = [*fixed, *(term for pair in pairs for term in pair)]
    assert len(flattened) == len(set(flattened)) == 48


def test_faithful_twisted_stream_is_reproducible_n32_and_exact_zx() -> None:
    kwargs = {"f_weight": 6, "g_weight": 10, "count": 10, "seed": 79}
    first = list(random_faithful_twisted_half_swap_candidates(**kwargs))
    second = list(random_faithful_twisted_half_swap_candidates(**kwargs))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(candidate.num_qubits == 32 for candidate in first)
    assert all(Monomial(0, 0, 0) in candidate.f_terms for candidate in first)

    candidate: FaithfulTwistedHalfSwapCandidate = first[0]
    hx, hz = faithful_twisted_check_matrices(candidate)
    fold = zx_fold_permutation(candidate)
    folded_hx = np.zeros_like(hx)
    folded_hx[:, fold] = hx
    assert np.array_equal(
        folded_hx[faithful_x_reflection_permutation()], hz
    )


def test_faithful_commutant_formula_cancels_over_gf2() -> None:
    mm = Monomial
    ff = tuple(sorted((mm(0, 0, 0), mm(2, 1, 0))))
    gg = faithful_commutant_g_terms(
        ff,
        p_support=((0, 0),),
        q_support=((0, 0),),
    )
    assert gg == (mm(2, 1, 0),)


def test_faithful_commutant_stream_is_reproducible_exact_css_and_zx() -> None:
    kwargs = {
        "f_weight": 4,
        "p_weight": 2,
        "q_weight": 1,
        "check_weight_ceiling": 16,
        "count": 10,
        "seed": 83,
    }
    first = list(random_faithful_commutant_candidates(**kwargs))
    second = list(random_faithful_commutant_candidates(**kwargs))
    assert first == second
    assert len(first) == len(set(first)) == 10
    for candidate in first:
        assert candidate.num_qubits == 32
        assert candidate.nominal_check_weight <= 16
        hx, hz = faithful_twisted_check_matrices(candidate)
        assert not np.any((hx @ hz.T) % 2)
        fold = zx_fold_permutation(candidate)
        folded_hx = np.zeros_like(hx)
        folded_hx[:, fold] = hx
        assert np.array_equal(
            folded_hx[faithful_x_reflection_permutation()], hz
        )


def test_faithful_twisted_protograph_stream_is_reproducible_css_and_zx() -> None:
    profiles = (
        ((2, 2), (2, 2), 64),
        ((2, 1, 1), (2, 2, 2), 96),
    )
    for f_profile, p_profile, expected_n in profiles:
        kwargs = {
            "f_weights": f_profile,
            "p_weights": p_profile,
            "coupling_shift": 1,
            "check_weight_ceiling": 16,
            "count": 10,
            "seed": 89,
        }
        first = list(random_faithful_twisted_protograph_candidates(**kwargs))
        second = list(random_faithful_twisted_protograph_candidates(**kwargs))
        assert first == second
        assert len(first) == len(set(first)) == 10
        for candidate in first:
            candidate: FaithfulTwistedProtographCandidate
            assert candidate.num_qubits == expected_n
            assert candidate.nominal_check_weight <= 16
            hx, hz = faithful_twisted_protograph_check_matrices(candidate)
            assert not np.any((hx @ hz.T) % 2)
            fold = zx_fold_permutation(candidate)
            folded_hx = np.zeros_like(hx)
            folded_hx[:, fold] = hx
            assert np.array_equal(
                folded_hx[faithful_x_reflection_permutation()], hz
            )


def test_published_bb64_has_the_required_raw_two_logical_batches() -> None:
    code = published_bb64_code()
    check = np.asarray(code.matrix_x, dtype=np.uint8)
    assert (code.num_qubits, code.dimension) == (64, 8)
    assert np.array_equal(code.matrix_x, code.matrix_z)
    assert set(np.count_nonzero(check, axis=1)) == {8}

    grid_x = published_bb64_full_permutation(
        published_bb64_translation_permutation((0, 1)), swap_halves=False
    )
    grid_y = published_bb64_full_permutation(
        published_bb64_affine_permutation((3, 4), (0, 1), (1, 0)),
        swap_halves=True,
    )
    assert np.array_equal(grid_x[grid_y], grid_y[grid_x])
    alpha = paper_logical(
        [(0, -2), (2, -2), (4, 0)],
        [(3, -1), (3, 0), (3, 1), (0, -2), (2, -2)],
    )
    logicals = published_bb64_orbit(alpha, grid_x, grid_y)
    assert not np.any((check @ logicals.T) % 2)
    assert gf2_rank(np.vstack([check, logicals]), code.field) - code.code_x.rank == 8
    assert gf2_rank((logicals @ logicals.T) % 2, code.field) == 8
    assert np.all(np.count_nonzero(logicals, axis=1) == 8)
    batching = batch_disjointness(logicals)
    assert batching["four_C2_y"]["disjoint"]
