"""Tests for the ``n=128`` two-batch geometry screen."""

from __future__ import annotations

import random

import numpy as np

from gala_search.batched_half_grid import (
    NUM_LOGICALS,
    analyze_seed,
    exact_geometry_certificate,
    iter_n128_folds,
    random_batched_seed,
    sheet_permutation,
)


def test_strict_n128_folds_are_fixed_point_free() -> None:
    folds = tuple(iter_n128_folds())
    assert len(folds) == 4
    assert all(
        all(sheet != image for sheet, image in enumerate(sheet_permutation(fold)))
        for fold in folds
    )


def test_random_allowed_seeds_are_batch_disjoint_but_zx_singular() -> None:
    rng = random.Random(128320)
    for weight in (7, 8):
        for fold in iter_n128_folds():
            for _ in range(10):
                analysis = analyze_seed(
                    random_batched_seed(rng, weight=weight), fold
                )
                assert analysis["batching"]["disjoint_within_each_batch"]
                assert analysis["all_ones_in_pairing_kernel"]
                assert analysis["zx_pairing_rank"] < NUM_LOGICALS
                assert not analysis["pairing_is_permutation"]


def test_exact_occupancy_certificate_prunes_generator_search() -> None:
    result = exact_geometry_certificate()
    assert result["maximum_seed_weight_under_batching"] == 8
    assert result["all_folds_fixed_point_free"]
    assert result["all_allowed_augmentations_zero"]
    assert result["pairing_singular_for_every_allowed_seed"]
    assert result["generator_search_pruned"]
    assert all(case["augmentation_counts"] == {"0": case["occupancy_patterns"]} for case in result["cases"])
