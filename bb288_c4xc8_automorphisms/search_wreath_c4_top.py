#!/usr/bin/env python3
"""Exact normalized search for ``C_4 x C_8`` in the Tanner wreath product.

For four isomorphic connected components, the full Tanner automorphism group is
``G wr S_4``.  Any element whose component action is a four-cycle is conjugate
to a normalized twisted cycle ``Y=(0123; 1,1,1,p)``.  This script enumerates
every such twist ``p`` with nontrivial logical order two and the complete
centralizer of each normalized ``Y``.  It therefore exactly covers the case in
which the top action of the desired abelian group is cyclic of order four.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynauty
from qldpc import codes

import analyze as base
import analyze_components as components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--cyclic-seed-trials", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=4288)
    parser.add_argument(
        "--allow-non-order-two-physical-twists",
        action="store_true",
        help="also use twists with logical order two but larger physical order",
    )
    return parser.parse_args()


def inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    output = np.empty_like(permutation)
    output[permutation] = np.arange(len(permutation), dtype=np.int64)
    return output


def wreath_action(top_power: int, locals_: Sequence[np.ndarray]) -> np.ndarray:
    """Logical row-action matrix for source-local maps and top ``(0123)^r``."""
    local_dimension = len(locals_[0])
    output = np.zeros((4 * local_dimension, 4 * local_dimension), dtype=np.uint8)
    for source, action in enumerate(locals_):
        target = (source + top_power) % 4
        output[
            source * local_dimension : (source + 1) * local_dimension,
            target * local_dimension : (target + 1) * local_dimension,
        ] = action
    return output


def wreath_physical(
    top_power: int,
    locals_: Sequence[np.ndarray],
    isomorphisms: Sequence[np.ndarray],
) -> np.ndarray:
    output = np.empty(base.NUM_QUBITS, dtype=np.int64)
    for source, local in enumerate(locals_):
        target = (source + top_power) % 4
        output[isomorphisms[source]] = isomorphisms[target][local]
    return output


def exact_order_four(matrix: np.ndarray) -> bool:
    square = (matrix @ matrix) % 2
    return not np.array_equal(square, np.eye(len(matrix), dtype=np.uint8)) and np.array_equal(
        (square @ square) % 2, np.eye(len(matrix), dtype=np.uint8)
    )


def independent_actions(action_x: np.ndarray, action_y: np.ndarray) -> list[np.ndarray] | None:
    actions = base.action_group(action_x, action_y)
    return actions if len({base.matrix_key(action) for action in actions}) == 32 else None


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    hx, hz = base.build_checks()
    full_graph = base.build_tanner_graph(hx, hz)
    graph_components = base.graph_connected_components(full_graph)
    parts = [components.component_parts(component) for component in graph_components]
    parts.sort(key=lambda item: item[0][0])
    data_parts = [part[0] for part in parts]
    isomorphisms, shifts = components.component_isomorphisms(data_parts)

    data, checks_x, checks_z = parts[0]
    local_hx = hx[np.ix_(checks_x, data)]
    local_hz = hz[np.ix_(checks_z, data)]
    local_graph = components.local_tanner_graph(local_hx, local_hz)
    raw_generators, size_base, size_exponent, _orbits, _num_orbits = pynauty.autgrp(local_graph)
    group_size = base.nauty_group_size(size_base, size_exponent)
    generators = [
        np.asarray(generator[: len(data)], dtype=np.int64)
        for generator in raw_generators
    ]
    elements = components.generate_local_group(generators, group_size)

    local_code = codes.CSSCode(local_hx, local_hz)
    logical_z, logical_x, pairing_inverse = base.logical_bases(local_code)
    identity_physical = np.arange(len(data), dtype=np.int64)
    identity_logical = np.eye(local_code.dimension, dtype=np.uint8)
    local_actions = {
        element.tobytes(): base.induced_z_action(
            element, logical_z, logical_x, pairing_inverse
        )
        for element in elements
    }
    local_orders = {
        element.tobytes(): base.matrix_order(
            local_actions[element.tobytes()],
            maximum=base.permutation_order(element),
        )
        for element in elements
    }

    twists = [
        element
        for element in elements
        if local_orders[element.tobytes()] == 2
        and (
            args.allow_non_order_two_physical_twists
            or base.permutation_order(element) == 2
        )
    ]
    base.checkpoint(
        "wreath_setup",
        component_group_size=group_size,
        twists=len(twists),
        physical_order_two_only=not args.allow_non_order_two_physical_twists,
        component_shifts=shifts,
    )

    counters: Counter[str] = Counter()
    rng = np.random.default_rng(args.random_seed)
    witness = None
    for twist_index, twist in enumerate(twists, start=1):
        twist_action = local_actions[twist.tobytes()]
        y_locals_physical = [identity_physical, identity_physical, identity_physical, twist]
        y_locals_logical = [identity_logical, identity_logical, identity_logical, twist_action]
        action_y = wreath_action(1, y_locals_logical)
        if base.matrix_order(action_y, maximum=8) != 8:
            raise RuntimeError("normalized four-cycle twist does not have logical order eight")

        gy = y_locals_physical
        for top_power in range(4):
            for initial in elements:
                counters["centralizer_candidates"] += 1
                hs: list[np.ndarray | None] = [None, None, None, None]
                hs[0] = initial
                valid = True
                for source in range(4):
                    current = hs[source]
                    if current is None:
                        valid = False
                        break
                    target = (source + 1) % 4
                    next_value = base.compose(
                        gy[(source + top_power) % 4],
                        base.compose(current, inverse_permutation(gy[source])),
                    )
                    if hs[target] is None:
                        hs[target] = next_value
                    elif not np.array_equal(hs[target], next_value):
                        valid = False
                        break
                if not valid:
                    continue
                counters["centralizer_elements"] += 1
                hs_physical = [np.asarray(item, dtype=np.int64) for item in hs]
                hs_logical = [local_actions[item.tobytes()] for item in hs_physical]
                action_x = wreath_action(top_power, hs_logical)
                if not exact_order_four(action_x):
                    continue
                counters["logical_order_four"] += 1
                if not np.array_equal(
                    (action_x @ action_y) % 2,
                    (action_y @ action_x) % 2,
                ):
                    raise RuntimeError("centralizer recurrence produced noncommuting actions")
                actions = independent_actions(action_x, action_y)
                if actions is None:
                    continue
                counters["faithful_c4xc8"] += 1
                cyclic = base.cyclic_vector(
                    actions,
                    trials=args.cyclic_seed_trials,
                    rng=rng,
                )
                if cyclic is None:
                    continue
                counters["regular_c4xc8"] += 1
                seed, orbit = cyclic
                full_x = wreath_physical(top_power, hs_physical, isomorphisms)
                full_y = wreath_physical(1, y_locals_physical, isomorphisms)
                if not np.array_equal(base.compose(full_x, full_y), base.compose(full_y, full_x)):
                    raise RuntimeError("logical wreath witness does not commute physically")
                rank_x = base.gf2_rank(hx)
                rank_z = base.gf2_rank(hz)
                preserves = {
                    "x_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_x)])) == rank_x,
                    "x_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_x)])) == rank_z,
                    "y_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_y)])) == rank_x,
                    "y_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_y)])) == rank_z,
                }
                if not all(preserves.values()):
                    raise RuntimeError("wreath witness does not preserve full check spaces")
                witness = {
                    "scope": "exact normalized four-cycle-top wreath-product search",
                    "component_shifts": shifts,
                    "twist_local_permutation": twist.astype(int).tolist(),
                    "x_top_power": top_power,
                    "x_local_permutations": [item.astype(int).tolist() for item in hs_physical],
                    "x_physical_permutation": full_x.astype(int).tolist(),
                    "y_physical_permutation": full_y.astype(int).tolist(),
                    "x_physical_order": base.permutation_order(full_x),
                    "y_physical_order": base.permutation_order(full_y),
                    "x_logical_action_component_basis": action_x.astype(int).tolist(),
                    "y_logical_action_component_basis": action_y.astype(int).tolist(),
                    "seed_logical_coordinates": seed.astype(int).tolist(),
                    "orbit_logical_coordinates": orbit.astype(int).tolist(),
                    "orbit_rank": base.gf2_rank(orbit),
                    "preserves_check_spaces": preserves,
                }
                break
            if witness is not None:
                break
        if args.checkpoint_every and counters["centralizer_candidates"] % args.checkpoint_every < 4 * len(elements):
            base.checkpoint(
                "wreath_search",
                twists_tested=twist_index,
                total_twists=len(twists),
                seconds=round(time.perf_counter() - started, 3),
                **dict(counters),
            )
        if witness is not None:
            break

    summary: dict[str, Any] = {
        "search_scope": "all normalized C4-top cases in G wr S4",
        "component_group_size": group_size,
        "twists_tested": twist_index if twists else 0,
        "total_twists": len(twists),
        "allow_non_order_two_physical_twists": args.allow_non_order_two_physical_twists,
        "counters": dict(counters),
        "regular_c4xc8_found": witness is not None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    base.atomic_json(args.output_dir / "summary.json", summary)
    if witness is not None:
        base.atomic_json(args.output_dir / "witness.json", witness)
    base.checkpoint("complete", **summary)


if __name__ == "__main__":
    main()
