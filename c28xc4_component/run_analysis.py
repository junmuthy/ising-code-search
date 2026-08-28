#!/usr/bin/env python3
"""Analyze and save the C28 x C4 exact-self-dual Ising component code."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter
from typing import Any

import numpy as np

from component_code import (
    CHECK_RANK,
    N,
    check_seed_polynomials,
    check_space_basis,
    independent_selection,
    logical_fibres,
    qldpc_code,
    translated_check_catalog,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--basis-trials", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    rows, metadata, orbit_sizes = translated_check_catalog()
    rref_basis = check_space_basis()
    properties = validate(rref_basis)
    code = qldpc_code(rref_basis)
    properties["qldpc_length"] = len(code)
    properties["qldpc_dimension"] = int(code.dimension)

    generator = random.Random(args.seed)
    order = list(range(len(rows)))
    best = None
    for trial in range(args.basis_trials):
        generator.shuffle(order)
        selected = independent_selection(rows, order)
        if len(selected) != CHECK_RANK:
            raise RuntimeError("translated seed catalog did not span rank 48")
        degrees = rows[selected].sum(axis=0)
        score = (
            int(degrees.max()),
            int(np.sum(degrees.astype(int) ** 2)),
            int(degrees.max() - degrees.min()),
        )
        if best is None or score < best[0]:
            best = (score, list(selected), degrees.copy(), trial)
    assert best is not None
    score, selected, degrees, best_trial = best
    check = rows[selected]
    if len(check_space_basis()) != len(check):
        raise RuntimeError("selected basis has the wrong rank")

    sibling = pathlib.Path(__file__).resolve().parents[1] / "one_block_c56xc4"
    if not sibling.exists():
        sibling = pathlib.Path(
            "/home/judah_unmuth/gala-code-search/one_block_c56xc4"
        )
    sys.path.insert(0, str(sibling))
    from search import find_logical_through_weight_five, find_weight_six_logical

    logical = logical_fibres()[0]
    low_weight, low_support, low_seconds = find_logical_through_weight_five(
        check, logical
    )
    if low_weight is not None:
        distance = low_weight
        distance_support = low_support
        distance_method = "exact_subset_search_through_five"
    else:
        six_support, six_seconds = find_weight_six_logical(check, logical)
        low_seconds += six_seconds
        if six_support is not None:
            distance = 6
            distance_support = six_support
            distance_method = "exact_meet_in_the_middle_3_plus_3"
        else:
            distance = 7
            distance_support = np.flatnonzero(logical).astype(int).tolist()
            distance_method = (
                "exact_exclusion_through_six_with_weight_seven_fibre"
            )

    basis_summary = {
        "random_trials": args.basis_trials,
        "random_seed": args.seed,
        "best_trial": best_trial,
        "score": list(score),
        "independent_checks": len(selected),
        "check_weight_histogram": dict(
            sorted(Counter(map(int, check.sum(axis=1))).items())
        ),
        "qubit_degree_histogram": dict(sorted(Counter(map(int, degrees)).items())),
        "minimum_qubit_degree": int(degrees.min()),
        "maximum_qubit_degree": int(degrees.max()),
        "average_qubit_degree": float(degrees.mean()),
        "selected_checks": [
            {
                "seed": metadata[index][0],
                "shift_u": metadata[index][1],
                "shift_v": metadata[index][2],
            }
            for index in selected
        ],
    }
    polynomial_summary = {
        "ring": "F2[u^+-1,v^+-1]/(u^28-1,v^4-1)",
        "simplex": "s(u)=1+u^4+u^8+u^16",
        "simplex_dagger": "s_dagger(u)=1+u^12+u^20+u^24",
        "check_seeds": [
            "h1=s(u)(1+v+v^2+v^3)",
            "h2=s(u)(1+u^7+u^14+u^21)",
            "h3=s_dagger(u)(1+u^21)(1+v)",
        ],
        "seed_weights": [int(seed.sum()) for seed in check_seed_polynomials()],
        "unique_translation_orbit_sizes": list(orbit_sizes),
        "individual_orbit_ranks": [12, 12, 27],
        "combined_orbit_rank": len(rref_basis),
        "logical_fibres": "ell_(a,b)=u^a v^b sum_(r=0)^6 u^(4r), a,b in C4",
    }
    summary = {
        **properties,
        "distance": distance,
        "distance_support": distance_support,
        "distance_method": distance_method,
        "distance_seconds": low_seconds,
        "minimum_check_weight": 16,
        "minimum_check_weight_certificate": (
            "weight<=12 MILP infeasible; check space doubly even; weight-16 seeds exist"
        ),
        "check_seed_types": 3,
        "available_weight_16_translated_checks": len(rows),
        "selected_check_basis_maximum_qubit_degree": int(degrees.max()),
        "single_css_type_cnot_depth": 16,
        "single_css_type_cnot_depth_optimal": True,
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "polynomials.json", polynomial_summary)
    write_json(args.output / "balanced-check-basis.json", basis_summary)
    np.savez_compressed(
        args.output / "code-matrices.npz",
        check_x=check,
        check_z=check,
        logical_x=logical_fibres(),
        logical_z=logical_fibres(),
        rref_check_space=rref_basis,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
