#!/usr/bin/env python3
"""Factor the disconnected ``[[288,32,8]]`` Tanner graph componentwise.

The full canonical Tanner graph has four connected components.  This script
computes the exact automorphism group of one component, verifies the component
code parameters, searches for a regular logical ``C_4 x C_2``, and—if found—
weaves four components into a regular logical ``C_4 x C_8`` action.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynauty
from qldpc import codes

import analyze as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--cyclic-seed-trials", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=7288)
    return parser.parse_args()


def component_parts(component: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    x_offset = base.NUM_QUBITS
    z_offset = base.NUM_QUBITS + base.CELL_COUNT
    data = sorted(vertex for vertex in component if vertex < x_offset)
    checks_x = sorted(vertex - x_offset for vertex in component if x_offset <= vertex < z_offset)
    checks_z = sorted(vertex - z_offset for vertex in component if vertex >= z_offset)
    return data, checks_x, checks_z


def local_tanner_graph(hx: np.ndarray, hz: np.ndarray) -> pynauty.Graph:
    num_data = hx.shape[1]
    x_offset = num_data
    z_offset = num_data + len(hx)
    total = num_data + len(hx) + len(hz)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(total)}
    for row, support in enumerate(hx):
        check = x_offset + row
        for data in np.flatnonzero(support):
            data = int(data)
            adjacency[check].append(data)
            adjacency[data].append(check)
    for row, support in enumerate(hz):
        check = z_offset + row
        for data in np.flatnonzero(support):
            data = int(data)
            adjacency[check].append(data)
            adjacency[data].append(check)
    return pynauty.Graph(
        number_of_vertices=total,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[
            set(range(num_data)),
            set(range(x_offset, z_offset)),
            set(range(z_offset, total)),
        ],
    )


def generate_local_group(generators: Sequence[np.ndarray], size: int) -> list[np.ndarray]:
    num_data = len(generators[0]) if generators else 0
    identity = np.arange(num_data, dtype=np.int64)
    elements = [identity]
    seen = {identity.tobytes()}
    queue: deque[np.ndarray] = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            product = base.compose(generator, current)
            key = product.tobytes()
            if key in seen:
                continue
            seen.add(key)
            elements.append(product)
            queue.append(product)
    if len(elements) != size:
        raise RuntimeError(f"generated local group has {len(elements)} elements, expected {size}")
    return elements


def local_action_group(action_x: np.ndarray, action_y: np.ndarray) -> list[np.ndarray]:
    return [
        (base.matrix_power(action_x, aa) @ base.matrix_power(action_y, bb)) % 2
        for aa in range(4)
        for bb in range(2)
    ]


def find_local_c4xc2(
    elements: Sequence[np.ndarray],
    logical_z: np.ndarray,
    logical_x: np.ndarray,
    pairing_inverse: np.ndarray,
    *,
    random_seed: int,
    seed_trials: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    actions: dict[bytes, np.ndarray] = {}
    logical_order_four = []
    logical_order_two = []
    physical_histogram: Counter[int] = Counter()
    for element in elements:
        physical_order = base.permutation_order(element)
        physical_histogram[physical_order] += 1
        action = base.induced_z_action(element, logical_z, logical_x, pairing_inverse)
        actions[element.tobytes()] = action
        logical_order = base.matrix_order(action, maximum=physical_order)
        if logical_order == 4:
            logical_order_four.append(element)
        elif logical_order == 2:
            logical_order_two.append(element)

    profile: dict[str, Any] = {
        "physical_order_histogram": {
            str(key): value for key, value in sorted(physical_histogram.items())
        },
        "logical_order_four_elements": len(logical_order_four),
        "logical_order_two_elements": len(logical_order_two),
        "commuting_pairs": 0,
        "faithful_c4xc2_pairs": 0,
        "cyclic_pairs": 0,
    }
    rng = np.random.default_rng(random_seed)
    for xx in logical_order_four:
        action_x = actions[xx.tobytes()]
        for yy in logical_order_two:
            if not np.array_equal(base.compose(xx, yy), base.compose(yy, xx)):
                continue
            profile["commuting_pairs"] += 1
            action_y = actions[yy.tobytes()]
            group_actions = local_action_group(action_x, action_y)
            if len({base.matrix_key(action) for action in group_actions}) != 8:
                continue
            profile["faithful_c4xc2_pairs"] += 1
            witness = base.cyclic_vector(group_actions, trials=seed_trials, rng=rng)
            if witness is None:
                continue
            profile["cyclic_pairs"] += 1
            seed, orbit = witness
            return {
                "x_local_permutation": xx.astype(int).tolist(),
                "y_local_permutation": yy.astype(int).tolist(),
                "x_physical_profile": base.summarize_permutation(xx),
                "y_physical_profile": base.summarize_permutation(yy),
                "x_logical_action": action_x.astype(int).tolist(),
                "y_logical_action": action_y.astype(int).tolist(),
                "seed_logical_coordinates": seed.astype(int).tolist(),
                "orbit_logical_coordinates": orbit.astype(int).tolist(),
                "orbit_rank": base.gf2_rank(orbit),
            }, profile
    return None, profile


def component_isomorphisms(data_parts: Sequence[Sequence[int]]) -> tuple[list[np.ndarray], list[list[int]]]:
    """Find native translations taking component zero to every component."""
    source = set(data_parts[0])
    isomorphisms: list[np.ndarray] = []
    shifts: list[list[int]] = []
    for target_part in data_parts:
        target = set(target_part)
        found = None
        for dx in range(base.ELL):
            for dy in range(base.EMM):
                permutation = base.two_half_permutation(base.group_permutation(dx, dy))
                if {int(permutation[index]) for index in source} == target:
                    found = (permutation, [dx, dy])
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError("native translations do not identify Tanner components")
        permutation, shift = found
        isomorphisms.append(np.asarray([permutation[index] for index in data_parts[0]], dtype=np.int64))
        shifts.append(shift)
    return isomorphisms, shifts


def lift_internal(
    local: np.ndarray,
    isomorphisms: Sequence[np.ndarray],
) -> np.ndarray:
    output = np.empty(base.NUM_QUBITS, dtype=np.int64)
    for phi in isomorphisms:
        output[phi] = phi[local]
    return output


def twisted_component_cycle(
    twist: np.ndarray,
    isomorphisms: Sequence[np.ndarray],
) -> np.ndarray:
    output = np.empty(base.NUM_QUBITS, dtype=np.int64)
    for component in range(len(isomorphisms) - 1):
        output[isomorphisms[component]] = isomorphisms[component + 1]
    output[isomorphisms[-1]] = isomorphisms[0][twist]
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    hx, hz = base.build_checks()
    graph = base.build_tanner_graph(hx, hz)
    components = base.graph_connected_components(graph)
    parts = [component_parts(component) for component in components]
    parts.sort(key=lambda item: item[0][0])
    data_parts = [item[0] for item in parts]
    if len(parts) != 4:
        raise RuntimeError(f"expected four Tanner components, found {len(parts)}")

    component_parameters = []
    component_matrices = []
    for data, checks_x, checks_z in parts:
        local_hx = hx[np.ix_(checks_x, data)]
        local_hz = hz[np.ix_(checks_z, data)]
        rank_x = base.gf2_rank(local_hx)
        rank_z = base.gf2_rank(local_hz)
        component_parameters.append(
            {
                "n": len(data),
                "k": len(data) - rank_x - rank_z,
                "rank_x": rank_x,
                "rank_z": rank_z,
                "num_x_checks": len(checks_x),
                "num_z_checks": len(checks_z),
                "check_weights_x": sorted(set(map(int, np.count_nonzero(local_hx, axis=1)))),
                "check_weights_z": sorted(set(map(int, np.count_nonzero(local_hz, axis=1)))),
            }
        )
        component_matrices.append((local_hx, local_hz))
    base.checkpoint("components_reconstructed", parameters=component_parameters)

    local_hx, local_hz = component_matrices[0]
    local_graph = local_tanner_graph(local_hx, local_hz)
    generators_raw, size_base, size_exponent, _orbits, number_of_orbits = pynauty.autgrp(local_graph)
    local_group_size = base.nauty_group_size(size_base, size_exponent)
    num_local_data = len(data_parts[0])
    local_generators = [
        np.asarray(generator[:num_local_data], dtype=np.int64)
        for generator in generators_raw
    ]
    elements = generate_local_group(local_generators, local_group_size)
    base.checkpoint(
        "component_automorphisms",
        generators=len(local_generators),
        group_size=local_group_size,
        elements=len(elements),
        vertex_orbits=number_of_orbits,
    )

    local_code = codes.CSSCode(local_hx, local_hz)
    local_logical_z, local_logical_x, local_pairing_inverse = base.logical_bases(local_code)
    local_witness, local_profile = find_local_c4xc2(
        elements,
        local_logical_z,
        local_logical_x,
        local_pairing_inverse,
        random_seed=args.random_seed,
        seed_trials=args.cyclic_seed_trials,
    )
    base.checkpoint(
        "component_c4xc2_screen",
        found=local_witness is not None,
        **local_profile,
    )

    isomorphisms, shifts = component_isomorphisms(data_parts)
    full_witness = None
    full_profile: dict[str, Any] = {}
    if local_witness is not None:
        local_x = np.asarray(local_witness["x_local_permutation"], dtype=np.int64)
        local_y = np.asarray(local_witness["y_local_permutation"], dtype=np.int64)
        full_x = lift_internal(local_x, isomorphisms)
        full_y = twisted_component_cycle(local_y, isomorphisms)
        rank_x = base.gf2_rank(hx)
        rank_z = base.gf2_rank(hz)
        preserves_checks = {
            "x_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_x)])) == rank_x,
            "x_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_x)])) == rank_z,
            "y_maps_x": base.gf2_rank(np.vstack([hx, base.permute_rows(hx, full_y)])) == rank_x,
            "y_maps_z": base.gf2_rank(np.vstack([hz, base.permute_rows(hz, full_y)])) == rank_z,
        }
        full_code = codes.CSSCode(hx, hz)
        logical_z, logical_x, pairing_inverse = base.logical_bases(full_code)
        action_x = base.induced_z_action(full_x, logical_z, logical_x, pairing_inverse)
        action_y = base.induced_z_action(full_y, logical_z, logical_x, pairing_inverse)
        actions = base.action_group(action_x, action_y)
        rng = np.random.default_rng(args.random_seed + 1)
        cyclic = base.cyclic_vector(actions, trials=args.cyclic_seed_trials, rng=rng)
        full_profile = {
            "x_physical_order": base.permutation_order(full_x),
            "y_physical_order": base.permutation_order(full_y),
            "x_logical_order": base.matrix_order(action_x),
            "y_logical_order": base.matrix_order(action_y),
            "physical_actions_commute": bool(
                np.array_equal(base.compose(full_x, full_y), base.compose(full_y, full_x))
            ),
            "logical_actions_commute": bool(
                np.array_equal((action_x @ action_y) % 2, (action_y @ action_x) % 2)
            ),
            "distinct_logical_actions": len({base.matrix_key(action) for action in actions}),
            "cyclic_vector_found": cyclic is not None,
            "preserves_check_spaces": preserves_checks,
        }
        if all(preserves_checks.values()) and cyclic is not None:
            seed, orbit = cyclic
            full_witness = {
                "construction": "four-component twisted cycle from a local regular C4 x C2",
                "component_translation_isomorphism_shifts": shifts,
                "x_physical_permutation": full_x.astype(int).tolist(),
                "y_physical_permutation": full_y.astype(int).tolist(),
                "x_logical_action": action_x.astype(int).tolist(),
                "y_logical_action": action_y.astype(int).tolist(),
                "seed_logical_coordinates": seed.astype(int).tolist(),
                "orbit_logical_coordinates": orbit.astype(int).tolist(),
                "orbit_rank": base.gf2_rank(orbit),
            }
    base.checkpoint(
        "full_c4xc8_screen",
        found=full_witness is not None,
        **full_profile,
    )

    expected_wreath_size = local_group_size**4 * math.factorial(4)
    summary = {
        "full_code": {"n": 288, "k": 32, "published_distance": 8, "check_weight": 6},
        "tanner_components": len(parts),
        "component_parameters": component_parameters,
        "components_identified_by_native_shifts": shifts,
        "component_automorphism_group_size": local_group_size,
        "component_automorphism_generators": len(local_generators),
        "full_wreath_product_group_size": expected_wreath_size,
        "local_regular_c4xc2_found": local_witness is not None,
        "local_grid_search": local_profile,
        "full_regular_c4xc8_found": full_witness is not None,
        "full_grid_search": full_profile,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    base.atomic_json(args.output_dir / "summary.json", summary)
    if local_witness is not None:
        base.atomic_json(args.output_dir / "local-c4xc2-witness.json", local_witness)
    if full_witness is not None:
        base.atomic_json(args.output_dir / "full-c4xc8-witness.json", full_witness)
    np.savez_compressed(
        args.output_dir / "component-code.npz",
        matrix_x=local_hx,
        matrix_z=local_hz,
        data_indices=np.asarray(data_parts[0], dtype=np.int64),
    )
    base.checkpoint("complete", **summary)


if __name__ == "__main__":
    main()
