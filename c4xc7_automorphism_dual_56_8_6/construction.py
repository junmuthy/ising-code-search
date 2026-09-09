#!/usr/bin/env python3
"""Exact C4 x C_m automorphism-dual BB search for odd m.

The family has n=8m and b=sigma(a)^dagger.  For sigma reflecting the C4
coordinate, the product of the universal inversion/half-swap ZX fold with
the half-preserving sigma fold supplies a commuting C2 code automorphism.
No representative-support or logical-disjointness condition is imposed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

from common import (
    append_jsonl,
    atomic_json,
    compose,
    find_css_z_logical_through_weight_five,
    find_regular_permutation_h_grid_css,
    rref,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odd-order", required=True, choices=(3, 5, 7), type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-polynomial-weight", default=6, type=int)
    parser.add_argument("--minimum-polynomial-weight", default=1, type=int)
    parser.add_argument("--checkpoint-every", default=2000, type=int)
    return parser.parse_args()


class Family:
    def __init__(self, odd_order: int) -> None:
        self.odd_order = odd_order
        self.elements = tuple(itertools.product(range(4), range(odd_order)))
        self.index = {element: member for member, element in enumerate(self.elements)}
        self.group_order = len(self.elements)
        self.num_qubits = 2 * self.group_order
        self.forward_single = [
            [
                1 << self.index[((aa + xx) % 4, (bb + yy) % odd_order)]
                for xx, yy in self.elements
            ]
            for aa, bb in self.elements
        ]
        self.inverse_single = [
            [
                1 << self.index[((aa - xx) % 4, (bb - yy) % odd_order)]
                for xx, yy in self.elements
            ]
            for aa, bb in self.elements
        ]

    def transformed_partner(
        self, support: tuple[int, ...], sigma_y: int
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.index[((xx) % 4, (-sigma_y * yy) % self.odd_order)]
                for xx, yy in (self.elements[item] for item in support)
            )
        )

    def build_checks(
        self, support_a: tuple[int, ...], support_b: tuple[int, ...]
    ) -> tuple[list[int], list[int]]:
        hx = []
        hz = []
        for anchor in range(self.group_order):
            forward_a = 0
            forward_b = 0
            inverse_a = 0
            inverse_b = 0
            for term in support_a:
                forward_a ^= self.forward_single[anchor][term]
                inverse_a ^= self.inverse_single[anchor][term]
            for term in support_b:
                forward_b ^= self.forward_single[anchor][term]
                inverse_b ^= self.inverse_single[anchor][term]
            hx.append(forward_a | (forward_b << self.group_order))
            hz.append(inverse_b | (inverse_a << self.group_order))
        return hx, hz

    def physical_maps(
        self, sigma_y: int
    ) -> tuple[list[int], list[int], list[int]]:
        t4_bottom = [
            self.index[((xx + 1) % 4, yy)] for xx, yy in self.elements
        ]
        t4 = t4_bottom + [self.group_order + item for item in t4_bottom]
        universal = [0] * self.num_qubits
        sigma_fold = [0] * self.num_qubits
        for half in range(2):
            for member, (xx, yy) in enumerate(self.elements):
                inv = self.index[((-xx) % 4, (-yy) % self.odd_order)]
                transformed = self.index[
                    ((-xx) % 4, (sigma_y * yy) % self.odd_order)
                ]
                universal[half * self.group_order + member] = (
                    (1 - half) * self.group_order + inv
                )
                sigma_fold[half * self.group_order + member] = (
                    half * self.group_order + transformed
                )
        t2 = compose(universal, sigma_fold)
        return t4, t2, universal


def main() -> None:
    args = parse_args()
    if not 1 <= args.minimum_polynomial_weight <= args.maximum_polynomial_weight <= 6:
        raise SystemExit("polynomial weights must lie in 1..6 (check-weight ceiling 12)")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    family = Family(args.odd_order)
    records = args.output_dir / "grid-candidates.jsonl"
    counts: Counter[str] = Counter()
    started = time.perf_counter()

    for sigma_y in (1, args.odd_order - 1):
        t4, t2, fold = family.physical_maps(sigma_y)
        if compose(t4, t2) != compose(t2, t4):
            raise AssertionError("constructed C4 and C2 do not commute")
        for weight in range(
            args.minimum_polynomial_weight, args.maximum_polynomial_weight + 1
        ):
            for support_a in itertools.combinations(range(family.group_order), weight):
                counts["polynomials"] += 1
                counts[f"sigma_y_{sigma_y}_polynomials"] += 1
                support_b = family.transformed_partner(support_a, sigma_y)
                hx, hz = family.build_checks(support_a, support_b)
                sx, _ = rref(hx, family.num_qubits)
                dimension = family.num_qubits - 2 * len(sx)
                counts[f"k_{dimension}"] += 1
                if dimension != 8:
                    continue
                diagnostics: dict[str, object] = {}
                grid = find_regular_permutation_h_grid_css(
                    hx, hz, family.num_qubits, t4, t2, fold, diagnostics
                )
                counts[f"grid_failure_{diagnostics.get('failure')}"] += int(grid is None)
                if "action_algebra_rank" in diagnostics:
                    counts[f"action_algebra_rank_{diagnostics['action_algebra_rank']}"] += 1
                if grid is None:
                    continue
                counts["regular_permutation_h_grid"] += 1
                witness = find_css_z_logical_through_weight_five(
                    hx, hz, family.num_qubits
                )
                counts[
                    "distance_at_least_six"
                    if witness is None
                    else f"distance_at_most_{witness['weight']}"
                ] += 1
                append_jsonl(
                    records,
                    {
                        "family": f"C4xC{args.odd_order}-automorphism-dual-BB",
                        "parameters": [family.num_qubits, 8],
                        "sigma": [3, sigma_y],
                        "a_indices": list(support_a),
                        "a_support": [list(family.elements[item]) for item in support_a],
                        "b_indices": list(support_b),
                        "b_support": [list(family.elements[item]) for item in support_b],
                        "check_weight": 2 * weight,
                        "grid": grid,
                        "low_weight_z_logical": witness,
                        "distance_at_least_six": witness is None,
                        "distance_x_equals_distance_z_by_fold": True,
                    },
                )
                if witness is None:
                    # Keep searching to characterize the family, but make a hit
                    # visible immediately to monitoring clients.
                    print(
                        json.dumps(
                            {
                                "status": "hit",
                                "odd_order": args.odd_order,
                                "sigma_y": sigma_y,
                                "support_a": list(support_a),
                                "check_weight": 2 * weight,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if counts["polynomials"] % args.checkpoint_every == 0:
                    atomic_json(
                        args.output_dir / "summary.json",
                        {
                            "complete": False,
                            "counts": dict(sorted(counts.items())),
                            "seconds": round(time.perf_counter() - started, 6),
                        },
                    )

    summary = {
        "complete": True,
        "target": (
            f"[[{family.num_qubits},8,d>=6]] regular logical C4 x C2 and permutation-H"
        ),
        "scope": (
            f"complete C4xC{args.odd_order} automorphism-dual family for both "
            "C4-reflecting diagonal involutions in the stated weight range"
        ),
        "odd_order": args.odd_order,
        "minimum_polynomial_weight": args.minimum_polynomial_weight,
        "maximum_polynomial_weight": args.maximum_polynomial_weight,
        "maximum_check_weight": 2 * args.maximum_polynomial_weight,
        "hard_check_weight_ceiling": 12,
        "counts": dict(sorted(counts.items())),
        "records": str(records),
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
