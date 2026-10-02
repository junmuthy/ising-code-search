#!/usr/bin/env python3
"""Test the paper's explicit weight-eight logicals for BB64 STAR batching."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from .analyze_published_selfdual_bb64 import (
    ELEMENTS,
    INDEX,
    add,
    inverse,
    published_code,
    scale,
)
from .logicals import batch_disjointness, dress_logicals_for_batches
from .model import gf2_rank
from .run_faithful_twisted_search import atomic_json


def monomial_index(xx: int, yy: int) -> int:
    element = add(scale(xx, (1, 0)), scale(yy, (0, 1)))
    return INDEX[element]


def polynomial_support(terms: list[tuple[int, int]]) -> list[int]:
    return sorted({monomial_index(xx, yy) for xx, yy in terms})


def paper_logical(
    first: list[tuple[int, int]], second: list[tuple[int, int]]
) -> np.ndarray:
    vector = np.zeros(64, dtype=np.uint8)
    vector[polynomial_support(first)] = 1
    vector[[32 + index for index in polynomial_support(second)]] = 1
    return vector


def apply_permutation(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vector)
    output[permutation] = vector
    return output


def inversion_permutation() -> np.ndarray:
    bottom = np.asarray([INDEX[inverse(element)] for element in ELEMENTS], dtype=int)
    return np.hstack([bottom, 32 + bottom]).astype(int)


def half_swap_permutation() -> np.ndarray:
    return np.hstack([32 + np.arange(32), np.arange(32)]).astype(int)


def orbit(
    seed: np.ndarray, tx: np.ndarray, ty: np.ndarray
) -> np.ndarray:
    rows = []
    current_x = seed.copy()
    for _xx in range(4):
        current_y = current_x.copy()
        for _yy in range(2):
            rows.append(current_y)
            current_y = apply_permutation(current_y, ty)
        current_x = apply_permutation(current_x, tx)
    return np.asarray(rows, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-npz", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dressing-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    saved = np.load(args.grid_npz)
    tx = np.asarray(saved["grid_x_permutation"], dtype=int)
    ty = np.asarray(saved["grid_y_permutation"], dtype=int)
    code = published_code()
    check = np.asarray(code.matrix_x, dtype=np.uint8)

    alpha = paper_logical(
        [(0, -2), (2, -2), (4, 0)],
        [(3, -1), (3, 0), (3, 1), (0, -2), (2, -2)],
    )
    beta = paper_logical(
        [(0, 0), (0, 4)],
        [(-1, -1), (-1, 0), (-1, 1), (-1, 3), (-1, 4), (-1, 5)],
    )
    inversion = inversion_permutation()
    half_swap = half_swap_permutation()
    variants: list[tuple[str, np.ndarray]] = []
    for name, seed in (("alpha", alpha), ("beta", beta)):
        variants.extend(
            [
                (name, seed),
                (f"{name}-inverted", apply_permutation(seed, inversion)),
                (f"{name}-halfswap", apply_permutation(seed, half_swap)),
                (
                    f"{name}-inverted-halfswap",
                    apply_permutation(
                        apply_permutation(seed, inversion), half_swap
                    ),
                ),
            ]
        )

    records: list[dict[str, Any]] = []
    saved_success = False
    for name, seed in variants:
        translated = orbit(seed, tx, ty)
        kernel = not bool(np.any((check @ translated.T) % 2))
        rank_mod_stabilizers = (
            gf2_rank(np.vstack([check, translated]), code.field)
            - code.code_x.rank
        )
        pairing_rank = gf2_rank((translated @ translated.T) % 2, code.field)
        batches = batch_disjointness(translated)
        record: dict[str, Any] = {
            "variant": name,
            "seed_support": np.flatnonzero(seed).astype(int).tolist(),
            "seed_weight": int(np.count_nonzero(seed)),
            "orbit_weights": np.count_nonzero(translated, axis=1).astype(int).tolist(),
            "orbit_in_kernel": kernel,
            "orbit_rank_mod_stabilizers": rank_mod_stabilizers,
            "zx_pairing_rank": pairing_rank,
            "raw_batching": batches,
        }
        if kernel and rank_mod_stabilizers == 8 and pairing_rank == 8:
            dressed = []
            for scheme, raw in batches.items():
                if raw["disjoint"]:
                    result = {
                        "scheme": scheme,
                        "feasible": True,
                        "already_disjoint_without_dressing": True,
                    }
                else:
                    result = dress_logicals_for_batches(
                        code,
                        translated,
                        scheme,
                        time_limit=args.dressing_seconds,
                    )
                dressed.append(result)
                print(
                    json.dumps(
                        {
                            "variant": name,
                            "scheme": scheme,
                            "feasible": result.get("feasible"),
                            "solver_status": result.get("solver_status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if result.get("feasible") is True:
                    saved_success = True
                    np.savez_compressed(
                        args.output_dir / "published-logical-grid.npz",
                        matrix_x=check,
                        matrix_z=check,
                        seed=seed,
                        orbit=translated,
                        grid_x_permutation=tx,
                        grid_y_permutation=ty,
                    )
                    break
            record["dressed_batching"] = dressed
        records.append(record)

    summary = {
        "source": "explicit alpha/beta logicals from arXiv:2510.05211v2",
        "regular_variants": sum(
            record["orbit_in_kernel"]
            and record["orbit_rank_mod_stabilizers"] == 8
            and record["zx_pairing_rank"] == 8
            for record in records
        ),
        "star_batch_found": saved_success,
        "variants": records,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
