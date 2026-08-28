#!/usr/bin/env python3
"""Audit alternative ZX folds and full logical action of the compact S3 code."""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from gala_search.s3_fold_analysis import (
    analyze_logical_action,
    build_certified_compact_code,
    certified_compact_candidate,
    certified_logical_grid,
    enumerate_structured_zx_folds,
    internal_affine_permutation,
    structured_physical_permutation,
)
from gala_search.s3_ising import _zx_fold_permutation

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    code = build_certified_compact_code()
    grid = certified_logical_grid()
    started = time.perf_counter()

    def progress(tested: int, folds: int) -> None:
        print(json.dumps({"tested": tested, "zx_folds": folds}), flush=True)

    folds, search_summary = enumerate_structured_zx_folds(
        code, grid, progress=progress
    )
    folds.sort(
        key=lambda item: (
            -item["grid_pairing_rank"],
            item["permutation_order"],
            item["block_permutation"],
            item["top_permutation"],
            item["bottom_automorphism"],
        )
    )
    original_permutation = _zx_fold_permutation(4)
    logical_summary, original_arrays = analyze_logical_action(
        code, original_permutation, grid
    )
    saved_arrays = {f"original_{key}": value for key, value in original_arrays.items()}
    for index, fold in enumerate(folds):
        permutation = structured_physical_permutation(
            fold["block_permutation"],
            internal_affine_permutation(
                fold["top_permutation"], fold["bottom_automorphism"]
            ),
        )
        fold_summary, fold_arrays = analyze_logical_action(code, permutation, grid)
        fold["logical_action"] = fold_summary
        for key in (
            "physical_permutation",
            "symplectic",
            "action_quadratic_matrix",
            "grid_image_coordinates",
            "grid_orbit_closure",
            "grid_orbit_closure_pairing",
            "global_z_fold_form",
            "grid_z_fold_form",
            "hyperbolic_basis_transform",
            "hyperbolic_form",
            "grid_partner_z_coordinates",
            "grid_partner_physical_z",
            "canonical_z_to_x",
            "canonical_x_to_z",
        ):
            saved_arrays[f"fold_{index}_{key}"] = fold_arrays[key]
    report = {
        "candidate_id": certified_compact_candidate().candidate_id,
        "parameters": {"n": code.num_qubits, "k": code.dimension, "d": 6},
        "structured_fold_search": search_summary,
        "best_grid_pairing_rank": max(
            (fold["grid_pairing_rank"] for fold in folds), default=None
        ),
        "grid_pairing_rank_histogram": {
            str(rank): sum(fold["grid_pairing_rank"] == rank for fold in folds)
            for rank in sorted({fold["grid_pairing_rank"] for fold in folds})
        },
        "folds": folds,
        "original_fold_logical_action": logical_summary,
        "seconds": round(time.perf_counter() - started, 6),
    }
    json_path = RESULTS_DIR / "s3-compact-fold-analysis.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    npz_path = RESULTS_DIR / "s3-compact-logical-action.npz"
    np.savez_compressed(npz_path, **saved_arrays)
    print(json.dumps({"json": str(json_path), "npz": str(npz_path), **report}), flush=True)


if __name__ == "__main__":
    main()
