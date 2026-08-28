"""Regression tests for the paper-faithful abelian ``n=32`` searches."""

from __future__ import annotations

import numpy as np

from gala_search.abelian_bb32 import (
    INVOLUTIONS,
    build_checks as build_bb_checks,
    c4_fibres as bb_fibres,
    diagonal_fold,
    index as bb_index,
    paired_graph_seeds as bb_paired_seeds,
    verify_fold as verify_bb_fold,
)
from gala_search.abelian_single_row import (
    DUALITY_SECTORS,
    GROUP_SPECS,
    LOGICAL_WEIGHT,
    NUM_QUBITS,
    build_checks,
    c4_fibres,
    derive_g_supports,
    physical_fold,
    verify_paper_fold,
)


def test_degree_eight_groups_have_eight_c4_fibres() -> None:
    for spec in GROUP_SPECS.values():
        assert len(spec.elements) == 8
        fibres = c4_fibres(spec)
        assert len(fibres) == 8
        assert all(len(fibre) == 4 for fibre in fibres)
        assert sorted(qubit for fibre in fibres for qubit in fibre) == list(
            range(NUM_QUBITS)
        )


def test_all_four_gala_sector_folds_exchange_checks() -> None:
    f_supports = ((0, 1, 3), (2, 5, 7))
    for group_key, spec in GROUP_SPECS.items():
        translation = spec.involutions[-1]
        for sector in DUALITY_SECTORS:
            g_supports = derive_g_supports(
                spec,
                f_supports,
                relation=sector.relation,
                translation=translation,
            )
            matrix_x, matrix_z = build_checks(group_key, f_supports, g_supports)
            assert not np.any((matrix_x @ matrix_z.T) % 2)
            analysis = verify_paper_fold(
                matrix_x,
                matrix_z,
                physical_fold(
                    spec,
                    outer_shift=sector.outer_shift,
                    translation=translation,
                ),
            )
            assert analysis["involution"]
            assert analysis["exact_with_paper_row_reindexing"]


def test_c4xc4_identity_fold_and_odd_disjoint_seeds() -> None:
    check, check_z = build_bb_checks((0, 1, 5, 7, 10, 15))
    assert np.array_equal(check, check_z)
    assert not np.any((check @ check.T) % 2)
    identity = INVOLUTIONS[0]
    analysis = verify_bb_fold(check, diagonal_fold(identity))
    assert analysis["involution"]
    assert analysis["exact_without_row_reindexing"]
    assert len(bb_fibres()) == 8
    seeds = bb_paired_seeds(bb_index(identity))
    assert len(seeds) == 8 * 4 ** (LOGICAL_WEIGHT - 1)
    assert all(len(seed) == LOGICAL_WEIGHT for seed in seeds[:10])
