#!/usr/bin/env python3
"""Search affine C2 presentations of the published BB64 logical grid."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from .analyze_published_bb64_logicals import orbit, paper_logical
from .analyze_published_selfdual_bb64 import (
    ELEMENTS,
    IDENTITY,
    INDEX,
    add,
    full_permutation,
    inverse,
    linear_automorphisms,
    permutation_order,
    permute_columns,
    published_code,
    translation_equivalent,
)
from .logicals import batch_disjointness
from .model import gf2_rank
from .run_faithful_twisted_search import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-npz", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    saved = np.load(args.grid_npz)
    tx = np.asarray(saved["grid_x_permutation"], dtype=int)
    code = published_code()
    check = np.asarray(code.matrix_x, dtype=np.uint8)
    alpha = paper_logical(
        [(0, -2), (2, -2), (4, 0)],
        [(3, -1), (3, 0), (3, 1), (0, -2), (2, -2)],
    )

    base_support = {IDENTITY, (1, 0), (0, 1), inverse((0, 1))}
    inverse_support = {inverse(element) for element in base_support}
    tested = 0
    code_symmetries = 0
    commuting_involutions = 0
    regular_presentations: list[dict[str, Any]] = []
    raw_successes: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    last_progress = 0
    for image_x, image_y, linear in linear_automorphisms():
        image_support = {IDENTITY, image_x, image_y, inverse(image_y)}
        allowed_swaps = []
        if translation_equivalent(image_support, base_support):
            allowed_swaps.append(False)
        if translation_equivalent(image_support, inverse_support):
            allowed_swaps.append(True)
        for translation in ELEMENTS:
            bottom = np.asarray(
                [
                    INDEX[add(translation, ELEMENTS[int(target)])]
                    for target in linear
                ],
                dtype=int,
            )
            for swap in allowed_swaps:
                permutation = full_permutation(bottom, swap_halves=swap)
                key = permutation.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                tested += 1
                if gf2_rank(
                    np.vstack([check, permute_columns(check, permutation)]),
                    code.field,
                ) != code.code_x.rank:
                    continue
                code_symmetries += 1
                if permutation_order(permutation) != 2:
                    continue
                if not np.array_equal(permutation[tx], tx[permutation]):
                    continue
                commuting_involutions += 1
                logicals = orbit(alpha, tx, permutation)
                rank_mod_stabilizers = (
                    gf2_rank(np.vstack([check, logicals]), code.field)
                    - code.code_x.rank
                )
                pairing_rank = gf2_rank((logicals @ logicals.T) % 2, code.field)
                if rank_mod_stabilizers != 8 or pairing_rank != 8:
                    continue
                pairing_matrix = (logicals @ logicals.T) % 2
                batches = batch_disjointness(logicals)
                record = {
                    "image_x": list(image_x),
                    "image_y": list(image_y),
                    "translation": list(translation),
                    "swap_halves": swap,
                    "orbit_weights": np.count_nonzero(logicals, axis=1).astype(int).tolist(),
                    "zx_pairing": pairing_matrix.astype(int).tolist(),
                    "zx_pairing_row_weights": pairing_matrix.sum(axis=1).astype(int).tolist(),
                    "zx_pairing_is_permutation": bool(
                        np.all(pairing_matrix.sum(axis=0) == 1)
                        and np.all(pairing_matrix.sum(axis=1) == 1)
                    ),
                    "raw_batching": batches,
                    "best_maximum_multiplicity": min(
                        max(result["maximum_multiplicity"])
                        for result in batches.values()
                    ),
                }
                regular_presentations.append(record)
                successful_schemes = [
                    name for name, result in batches.items() if result["disjoint"]
                ]
                if successful_schemes:
                    record["successful_schemes"] = successful_schemes
                    raw_successes.append(record)
                    np.savez_compressed(
                        args.output_dir / "raw-disjoint-grid.npz",
                        matrix_x=check,
                        matrix_z=check,
                        seed=alpha,
                        orbit=logicals,
                        grid_x_permutation=tx,
                        grid_y_permutation=permutation,
                    )
                    break
            if raw_successes:
                break
        if raw_successes:
            break
        if tested - last_progress >= 64:
            print(
                json.dumps(
                    {
                        "tested": tested,
                        "code_symmetries": code_symmetries,
                        "commuting_involutions": commuting_involutions,
                        "regular_presentations": len(regular_presentations),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_progress = tested

    best = sorted(
        regular_presentations,
        key=lambda record: (
            record["best_maximum_multiplicity"],
            sum(record["orbit_weights"]),
        ),
    )[:10]
    summary = {
        "tested_affine_presentations": tested,
        "code_symmetries": code_symmetries,
        "commuting_physical_involutions": commuting_involutions,
        "regular_zx_presentations": len(regular_presentations),
        "raw_disjoint_presentation_found": bool(raw_successes),
        "raw_success": raw_successes[0] if raw_successes else None,
        "best_presentations": best,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
