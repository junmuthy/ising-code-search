#!/usr/bin/env python3
"""Exact transitive Klein-four-top search in the Tanner wreath product.

Together with ``search_wreath_c4_top.py``, this completes the transitive top
actions possible for an abelian subgroup on four identical Tanner components.
The normalized top permutations are

    Y_top = (01)(23),   X_top = (02)(13).

After base conjugation, every order-eight ``Y`` relevant to this case has
local maps ``[1,p,1,p]`` with logical order-four twist ``p``.  Its commuting
``X`` elements have local maps ``[h0,h0,h2,h2]`` with ``h0,h2`` in the
centralizer of ``p``.  The script enumerates all such choices exactly.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynauty
from qldpc import codes

import analyze as base
import analyze_components as components


Y_TOP = np.asarray([1, 0, 3, 2], dtype=np.int64)
X_TOP = np.asarray([2, 3, 0, 1], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--cyclic-seed-trials", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=44288)
    parser.add_argument(
        "--allow-non-order-four-physical-twists",
        action="store_true",
        help="also use twists with logical order four but larger physical order",
    )
    return parser.parse_args()


def wreath_action(top: np.ndarray, locals_: Sequence[np.ndarray]) -> np.ndarray:
    local_dimension = len(locals_[0])
    output = np.zeros((4 * local_dimension, 4 * local_dimension), dtype=np.uint8)
    for source, action in enumerate(locals_):
        target = int(top[source])
        output[
            source * local_dimension : (source + 1) * local_dimension,
            target * local_dimension : (target + 1) * local_dimension,
        ] = action
    return output


def wreath_physical(
    top: np.ndarray,
    locals_: Sequence[np.ndarray],
    isomorphisms: Sequence[np.ndarray],
) -> np.ndarray:
    output = np.empty(base.NUM_QUBITS, dtype=np.int64)
    for source, local in enumerate(locals_):
        target = int(top[source])
        output[isomorphisms[source]] = isomorphisms[target][local]
    return output


def exact_order_four(matrix: np.ndarray) -> bool:
    identity = np.eye(len(matrix), dtype=np.uint8)
    square = (matrix @ matrix) % 2
    return not np.array_equal(square, identity) and np.array_equal((square @ square) % 2, identity)


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
        if local_orders[element.tobytes()] == 4
        and (
            args.allow_non_order_four_physical_twists
            or base.permutation_order(element) == 4
        )
    ]
    base.checkpoint(
        "v4_setup",
        component_group_size=group_size,
        twists=len(twists),
        physical_order_four_only=not args.allow_non_order_four_physical_twists,
        component_shifts=shifts,
    )

    counters: Counter[str] = Counter()
    rng = np.random.default_rng(args.random_seed)
    witness = None
    for twist_index, twist in enumerate(twists, start=1):
        twist_action = local_actions[twist.tobytes()]
        y_locals_physical = [identity_physical, twist, identity_physical, twist]
        y_locals_logical = [identity_logical, twist_action, identity_logical, twist_action]
        action_y = wreath_action(Y_TOP, y_locals_logical)
        if base.matrix_order(action_y, maximum=8) != 8:
            raise RuntimeError("normalized V4 twist does not have logical order eight")
        centralizer = [
            element
            for element in elements
            if np.array_equal(
                base.compose(element, twist),
                base.compose(twist, element),
            )
        ]
        counters["twist_centralizer_sizes"] += len(centralizer)
        for h0 in centralizer:
            for h2 in centralizer:
                counters["centralizer_pairs"] += 1
                x_locals_physical = [h0, h0, h2, h2]
                x_locals_logical = [
                    local_actions[h0.tobytes()],
                    local_actions[h0.tobytes()],
                    local_actions[h2.tobytes()],
                    local_actions[h2.tobytes()],
                ]
                action_x = wreath_action(X_TOP, x_locals_logical)
                if not exact_order_four(action_x):
                    continue
                counters["logical_order_four"] += 1
                if not np.array_equal(
                    (action_x @ action_y) % 2,
                    (action_y @ action_x) % 2,
                ):
                    raise RuntimeError("normalized V4 centralizer produced noncommuting actions")
                actions = base.action_group(action_x, action_y)
                if len({base.matrix_key(action) for action in actions}) != 32:
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
                full_x = wreath_physical(X_TOP, x_locals_physical, isomorphisms)
                full_y = wreath_physical(Y_TOP, y_locals_physical, isomorphisms)
                if not np.array_equal(base.compose(full_x, full_y), base.compose(full_y, full_x)):
                    raise RuntimeError("V4 witness does not commute physically")
                rank_x = base.gf2_rank(hx)
                rank_z = base.gf2_rank(hz)
                preserves = {
                    "x_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_x)])) == rank_x,
                    "x_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_x)])) == rank_z,
                    "y_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_y)])) == rank_x,
                    "y_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_y)])) == rank_z,
                }
                if not all(preserves.values()):
                    raise RuntimeError("V4 witness does not preserve full check spaces")
                witness = {
                    "scope": "exact normalized transitive-V4-top wreath-product search",
                    "component_shifts": shifts,
                    "twist_local_permutation": twist.astype(int).tolist(),
                    "x_local_h0": h0.astype(int).tolist(),
                    "x_local_h2": h2.astype(int).tolist(),
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
            if args.checkpoint_every and counters["centralizer_pairs"] % args.checkpoint_every < len(centralizer):
                base.checkpoint(
                    "v4_search",
                    twists_tested=twist_index,
                    total_twists=len(twists),
                    current_centralizer=len(centralizer),
                    seconds=round(time.perf_counter() - started, 3),
                    **dict(counters),
                )
        if witness is not None:
            break

    summary: dict[str, Any] = {
        "search_scope": "all normalized transitive V4-top cases in G wr S4",
        "component_group_size": group_size,
        "twists_tested": twist_index if twists else 0,
        "total_twists": len(twists),
        "allow_non_order_four_physical_twists": args.allow_non_order_four_physical_twists,
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
