"""Certified single-grid S3 Ising-code representative."""

from __future__ import annotations

from typing import Any

from qldpc import codes

from gala_search.s3_ising import (
    ProductMonomial,
    S3L4SelfDualW16Candidate,
    analyze_s3_l4_self_dual_w16_candidate,
    analyze_self_dual_translation_logical_seed,
    build_s3_l4_self_dual_w16_code,
)


def certified_single_grid_candidate() -> S3L4SelfDualW16Candidate:
    """Return the saved ``[[384, 200, 7]]`` candidate."""
    pp = ProductMonomial
    return S3L4SelfDualW16Candidate(
        f0=tuple(
            sorted(
                (
                    pp(0, 0, 0),
                    pp(0, 6, 2),
                    pp(2, 4, 1),
                    pp(5, 0, 1),
                )
            )
        ),
        f1=tuple(
            sorted(
                (
                    pp(0, 2, 0),
                    pp(0, 4, 3),
                    pp(3, 4, 0),
                    pp(3, 5, 2),
                )
            )
        ),
    )


def certified_single_grid_support() -> tuple[int, ...]:
    """Return one weight-seven representative for logical grid site ``(0,0)``."""
    return 32, 122, 164, 209, 266, 298, 329


def build_certified_single_grid_code() -> codes.GALACode:
    return build_s3_l4_self_dual_w16_code(certified_single_grid_candidate())


def analyze_certified_single_grid_code() -> dict[str, Any]:
    """Recompute the structural and logical-grid certificates."""
    candidate = certified_single_grid_candidate()
    code = build_s3_l4_self_dual_w16_code(candidate)
    return {
        "structure": analyze_s3_l4_self_dual_w16_candidate(candidate),
        "logical_grid": analyze_self_dual_translation_logical_seed(
            code, certified_single_grid_support()
        ),
    }
