"""Tests for the S3 packed-Ising polynomial GALA search."""

from __future__ import annotations

import numpy as np

from qldpc import codes

from gala_search.s3_ising import (
    BOTTOM_ORDER,
    ProductMonomial,
    S3L4Candidate,
    S3L4SelfDualW16Candidate,
    S3L4W16Candidate,
    S3L8Candidate,
    S3L8W16Candidate,
    _group_data,
    analyze_s3_l4_w16_candidate,
    analyze_self_dual_translation_logical_seed,
    analyze_translation_logical_seed,
    build_s3_l4_w16_code,
    candidate_generators,
    find_zero_syndrome_by_tanner_search,
    l8_candidate_generators,
    random_s3_l4_candidates,
    random_s3_l4_self_dual_w16_candidates,
    random_s3_l4_w16_candidates,
    random_s3_l8_candidates,
    random_s3_l8_w16_candidates,
    find_logical_up_to_weight_four,
    logical_translation_orbit,
    s3_internal_translation_orbits,
    self_dual_candidate_generators,
)
from gala_search.s3_fold_analysis import (
    analyze_logical_action,
    bottom_automorphisms,
    build_certified_compact_code,
    certified_logical_grid,
    gf2_rank,
    grid_pairing,
    internal_affine_permutation,
    is_zx_fold,
    structured_physical_permutation,
)
from gala_search.s3_single_grid import (
    build_certified_single_grid_code,
    certified_single_grid_candidate,
    certified_single_grid_support,
)


def test_s3_product_uses_natural_degree_three_lift() -> None:
    ring, _top, _xx, _yy = _group_data()
    assert ring.group.order == 6 * BOTTOM_ORDER
    assert ring.group.lift_dim == 3 * BOTTOM_ORDER


def test_l4_zx_partner_reverses_the_two_generators() -> None:
    candidate = S3L4Candidate(
        f0=tuple(sorted((ProductMonomial(0, 0, 0), ProductMonomial(3, 1, 0)))),
        f1=tuple(
            sorted(
                (
                    ProductMonomial(0, 0, 1),
                    ProductMonomial(1, 1, 1),
                    ProductMonomial(2, 0, 2),
                    ProductMonomial(5, 3, 1),
                )
            )
        ),
    )
    ff, gg = candidate_generators(candidate)
    assert gg[0] == ff[1].T
    assert gg[1] == ff[0].T
    assert candidate.num_qubits == 384


def test_random_candidates_are_reproducible_and_distinct() -> None:
    first = list(random_s3_l4_candidates(10, seed=17))
    second = list(random_s3_l4_candidates(10, seed=17))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(len(set(candidate.f0)) == 2 for candidate in first)
    assert all(len(set(candidate.f1)) == 4 for candidate in first)


def test_random_l4_w16_candidates_are_reproducible() -> None:
    first = list(random_s3_l4_w16_candidates(10, seed=19))
    second = list(random_s3_l4_w16_candidates(10, seed=19))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(isinstance(candidate, S3L4W16Candidate) for candidate in first)
    assert all(tuple(map(len, candidate.entries)) == (4, 4) for candidate in first)


def test_random_self_dual_l4_w16_candidates_are_reproducible() -> None:
    first = list(random_s3_l4_self_dual_w16_candidates(10, seed=21))
    second = list(random_s3_l4_self_dual_w16_candidates(10, seed=21))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(isinstance(candidate, S3L4SelfDualW16Candidate) for candidate in first)
    for candidate in first:
        ff, gg = self_dual_candidate_generators(candidate)
        assert gg == (ff[0].T, ff[1].T)


def test_l8_zx_partner_is_shifted_by_two_entries() -> None:
    candidate = S3L8Candidate(
        f0=(ProductMonomial(0, 0, 0),),
        f1=(ProductMonomial(1, 1, 0),),
        f2=(ProductMonomial(2, 0, 1),),
        f3=tuple(
            sorted(
                (
                    ProductMonomial(3, 1, 1),
                    ProductMonomial(4, 2, 1),
                    ProductMonomial(5, 3, 1),
                )
            )
        ),
    )
    ff, gg = l8_candidate_generators(candidate)
    for index in range(4):
        assert gg[(index + 2) % 4] == ff[index].T
    assert candidate.num_qubits == 768


def test_random_l8_candidates_are_reproducible() -> None:
    first = list(random_s3_l8_candidates(10, seed=23))
    second = list(random_s3_l8_candidates(10, seed=23))
    assert first == second
    assert len(first) == len(set(first)) == 10


def test_random_l8_w16_candidates_are_reproducible_and_binomial() -> None:
    first = list(random_s3_l8_w16_candidates(10, seed=29))
    second = list(random_s3_l8_w16_candidates(10, seed=29))
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(isinstance(candidate, S3L8W16Candidate) for candidate in first)
    assert all(tuple(map(len, candidate.entries)) == (2, 2, 2, 2) for candidate in first)


def test_low_weight_logical_mitm_distinguishes_stabilizers() -> None:
    # The repeated weight-two row is a stabilizer, while qubits 2 and 3 are
    # unconstrained weight-one logicals.  The search must return a logical.
    code = codes.CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]])
    result = find_logical_up_to_weight_four(code, pauli="Z")
    assert result is not None
    assert result["minimum_weight_upper_bound"] == 1
    assert result["support"][0] in {2, 3}


def test_bottom_translation_partition_and_clean_single_site_orbit() -> None:
    internal_orbits = s3_internal_translation_orbits(8)
    assert len(internal_orbits) == 24
    assert {len(orbit) for orbit in internal_orbits} == {32}
    assert set().union(*map(set, internal_orbits)) == set(range(768))

    seed = np.zeros(768, dtype=np.uint8)
    seed[internal_orbits[0][0]] = 1
    translated = logical_translation_orbit(seed)
    assert translated.shape == (32, 768)
    assert np.all(np.count_nonzero(translated, axis=1) == 1)
    assert np.all(np.sum(translated, axis=0) <= 1)


def test_certified_compact_s3_ising_code() -> None:
    pp = ProductMonomial
    candidate = S3L4W16Candidate(
        f0=tuple(
            sorted((pp(0, 0, 0), pp(0, 0, 2), pp(1, 5, 3), pp(5, 7, 0)))
        ),
        f1=tuple(
            sorted((pp(0, 4, 3), pp(0, 7, 3), pp(2, 7, 2), pp(5, 3, 0)))
        ),
    )
    analysis = analyze_s3_l4_w16_candidate(candidate, include_graph_metrics=False)
    assert analysis["accepted"]
    assert (analysis["n"], analysis["k"]) == (384, 194)
    assert analysis["column_degrees_x"] == analysis["column_degrees_z"] == [4]

    code = build_s3_l4_w16_code(candidate)
    for weight in range(1, 6):
        result = find_zero_syndrome_by_tanner_search(code, weight=weight)
        assert result is not None and result["search_exhaustive"]
        assert result["support"] is None
    distance_witness = find_zero_syndrome_by_tanner_search(code, weight=6)
    assert distance_witness is not None
    assert distance_witness["is_nontrivial_logical"]
    assert distance_witness["graph_supported"]

    orbit = analyze_translation_logical_seed(code, distance_witness["support"])
    assert orbit["pairwise_disjoint"]
    assert orbit["translation_orbit_size"] == 32
    assert orbit["orbit_rank_mod_stabilizers"] == 32
    assert orbit["zx_pairing_rank"] == 0


def test_certified_self_dual_single_grid_code() -> None:
    candidate = certified_single_grid_candidate()
    assert candidate.candidate_id == "s3-l4-j1-self-dual-w16-36d018790fa7a28a"
    code = build_certified_single_grid_code()
    assert (code.num_qubits, code.dimension) == (384, 200)
    assert np.array_equal(code.matrix_x, code.matrix_z)
    assert code.is_swel

    grid = analyze_self_dual_translation_logical_seed(
        code, certified_single_grid_support()
    )
    assert grid["weight"] == 7
    assert grid["graph_supported"]
    assert grid["pairwise_disjoint"]
    assert grid["z_orbit_in_kernel"] and grid["x_partner_orbit_in_kernel"]
    assert grid["orbit_rank_mod_stabilizers"] == 32
    assert grid["zx_pairing_rank"] == 32
    assert grid["pairing_is_identity"]
    assert grid["identity_zx_fold"]
    assert grid["x_translation_is_C8_grid_shift"]
    assert grid["y_translation_is_C4_grid_shift"]
    assert grid["logical_support_union_weight"] == 224
    assert grid["unused_physical_qubits"] == 160

    for weight in range(1, 7):
        result = find_zero_syndrome_by_tanner_search(
            code, weight=weight, max_nodes=5_000_000
        )
        assert result is not None and result["search_exhaustive"]
        assert result["support"] is None


def test_compact_s3_structured_folds_and_full_logical_action() -> None:
    code = build_certified_compact_code()
    grid = certified_logical_grid()
    assert len(bottom_automorphisms()) == 128

    original = structured_physical_permutation(
        (1, 0, 3, 2), internal_affine_permutation((0, 1, 2), (1, 0, 0, 1))
    )
    alternative = structured_physical_permutation(
        (2, 3, 0, 1), internal_affine_permutation((0, 1, 2), (7, 0, 0, 3))
    )
    matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
    matrix_z = np.asarray(code.matrix_z, dtype=np.uint8)
    for fold in (original, alternative):
        assert is_zx_fold(matrix_x, matrix_z, fold)
        assert gf2_rank(grid_pairing(grid, fold), code.field) == 0
        summary, arrays = analyze_logical_action(code, fold, grid)
        assert summary["is_symplectic"]
        assert summary["action_order_two"]
        assert summary["action_quadratic_form_zero"]
        assert summary["grid_orbit_closure_dimension"] == 64
        assert summary["grid_orbit_closure_symplectic_rank"] == 0
        assert summary["global_z_fold_form_rank"] == 194
        assert summary["global_z_fold_form_alternating"]
        assert summary["hadamard_swap_pairs"] == 97
        assert summary["canonical_hadamard_swap_verified"]
        assert arrays["hyperbolic_basis_transform"].shape == (194, 194)
