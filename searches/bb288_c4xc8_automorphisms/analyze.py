#!/usr/bin/env python3
"""Reconstruct and analyze the published ``[[288,32,8]]`` BB code.

The code is class ``j`` in Table 2 of Cruz-Benito et al., arXiv:2606.02418:

    A = 1 + x y^3 + x^5 y^3
    B = 1 + x^3 y + x^3 y^5

over ``F_2[C_12 x C_12]``.  This script verifies the static code properties,
computes the full color-preserving automorphism group of the canonical
translated-check Tanner graph with nauty, and tests the induced logical group
for a regular ``C_4 x C_8`` module.

The graph colors distinguish data, X-check, and Z-check vertices.  Thus the
automorphism group computed here preserves CSS type.  The ZX duality, which
exchanges X and Z checks, is verified separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pynauty
from qldpc import codes
from qldpc.objects import Pauli


ELL = 12
EMM = 12
CELL_COUNT = ELL * EMM
NUM_QUBITS = 2 * CELL_COUNT
TARGET_X_ORDER = 4
TARGET_Y_ORDER = 8

A_SUPPORT = ((0, 0), (1, 3), (5, 3))
B_SUPPORT = ((0, 0), (3, 1), (3, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enumeration-limit", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--cyclic-seed-trials", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=288328)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


def nauty_group_size(size_base: float, size_exponent: int) -> int:
    """Recover nauty's integer group size from its scientific notation."""
    return int(Decimal(str(size_base)) * (Decimal(10) ** int(size_exponent)))


def index(xx: int, yy: int) -> int:
    return (xx % ELL) * EMM + yy % EMM


def coordinate(position: int) -> tuple[int, int]:
    return divmod(position, EMM)


def group_permutation(dx: int, dy: int) -> np.ndarray:
    """Permutation taking ``(x,y)`` to ``(x+dx,y+dy)``."""
    return np.asarray(
        [index(xx + dx, yy + dy) for xx, yy in map(coordinate, range(CELL_COUNT))],
        dtype=np.int64,
    )


def inversion_permutation() -> np.ndarray:
    return np.asarray(
        [index(-xx, -yy) for xx, yy in map(coordinate, range(CELL_COUNT))],
        dtype=np.int64,
    )


def two_half_permutation(bottom: np.ndarray, *, swap_halves: bool = False) -> np.ndarray:
    if swap_halves:
        return np.concatenate([CELL_COUNT + bottom, bottom]).astype(np.int64)
    return np.concatenate([bottom, CELL_COUNT + bottom]).astype(np.int64)


def group_matrix(support: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return the translated-check matrix for one group-ring polynomial."""
    matrix = np.zeros((CELL_COUNT, CELL_COUNT), dtype=np.uint8)
    for xx, yy in map(coordinate, range(CELL_COUNT)):
        for dx, dy in support:
            matrix[index(xx, yy), index(xx + dx, yy + dy)] ^= 1
    return matrix


def build_checks() -> tuple[np.ndarray, np.ndarray]:
    aa = group_matrix(A_SUPPORT)
    bb = group_matrix(B_SUPPORT)
    return np.hstack([aa, bb]), np.hstack([bb.T, aa.T])


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank over GF(2), using Python integers as packed rows."""
    matrix = np.asarray(matrix, dtype=np.uint8)
    pivots: dict[int, int] = {}
    for row in matrix:
        packed = np.packbits(row, bitorder="little").tobytes()
        value = int.from_bytes(packed, "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ``left`` after ``right`` for image-list permutations."""
    return left[right]


def permutation_power(permutation: np.ndarray, exponent: int) -> np.ndarray:
    output = np.arange(len(permutation), dtype=np.int64)
    base = np.asarray(permutation, dtype=np.int64)
    while exponent:
        if exponent & 1:
            output = compose(base, output)
        base = compose(base, base)
        exponent //= 2
    return output


def permutation_order(permutation: np.ndarray) -> int:
    unseen = np.ones(len(permutation), dtype=bool)
    order = 1
    for start in range(len(permutation)):
        if not unseen[start]:
            continue
        current = start
        length = 0
        while unseen[current]:
            unseen[current] = False
            length += 1
            current = int(permutation[current])
        order = math.lcm(order, length)
    return order


def matrix_power(matrix: np.ndarray, exponent: int) -> np.ndarray:
    output = np.eye(len(matrix), dtype=np.uint8)
    base = np.asarray(matrix, dtype=np.uint8)
    while exponent:
        if exponent & 1:
            output = (output @ base) % 2
        base = (base @ base) % 2
        exponent //= 2
    return output


def matrix_order(matrix: np.ndarray, maximum: int = 1024) -> int | None:
    identity = np.eye(len(matrix), dtype=np.uint8)
    power = identity.copy()
    for exponent in range(1, maximum + 1):
        power = (power @ matrix) % 2
        if np.array_equal(power, identity):
            return exponent
    return None


def matrix_key(matrix: np.ndarray) -> bytes:
    return np.packbits(np.asarray(matrix, dtype=np.uint8), bitorder="little").tobytes()


def build_tanner_graph(hx: np.ndarray, hz: np.ndarray) -> pynauty.Graph:
    x_offset = NUM_QUBITS
    z_offset = NUM_QUBITS + len(hx)
    number_of_vertices = NUM_QUBITS + len(hx) + len(hz)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(number_of_vertices)}
    for row, support in enumerate(hx):
        check_vertex = x_offset + row
        for data in np.flatnonzero(support):
            data_vertex = int(data)
            adjacency[check_vertex].append(data_vertex)
            adjacency[data_vertex].append(check_vertex)
    for row, support in enumerate(hz):
        check_vertex = z_offset + row
        for data in np.flatnonzero(support):
            data_vertex = int(data)
            adjacency[check_vertex].append(data_vertex)
            adjacency[data_vertex].append(check_vertex)
    coloring = [
        set(range(NUM_QUBITS)),
        set(range(x_offset, z_offset)),
        set(range(z_offset, number_of_vertices)),
    ]
    return pynauty.Graph(
        number_of_vertices=number_of_vertices,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )


def graph_connected_components(graph: pynauty.Graph) -> list[list[int]]:
    unseen = set(range(graph.number_of_vertices))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = [start]
        component = []
        while queue:
            vertex = queue.pop()
            component.append(vertex)
            for neighbor in graph.adjacency_dict.get(vertex, []):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def generate_group(
    generators: Sequence[np.ndarray],
    *,
    limit: int,
    checkpoint_every: int,
) -> list[np.ndarray] | None:
    """Enumerate the generated data-permutation group when it is small enough."""
    identity = np.arange(NUM_QUBITS, dtype=np.int64)
    seen = {identity.tobytes()}
    elements = [identity]
    queue: deque[np.ndarray] = deque([identity])
    started = time.perf_counter()
    while queue:
        current = queue.popleft()
        for generator in generators:
            product = compose(generator, current)
            key = product.tobytes()
            if key in seen:
                continue
            seen.add(key)
            elements.append(product)
            queue.append(product)
            if len(elements) > limit:
                checkpoint(
                    "group_enumeration_limit",
                    discovered=len(elements),
                    limit=limit,
                    seconds=round(time.perf_counter() - started, 3),
                )
                return None
            if checkpoint_every and len(elements) % checkpoint_every == 0:
                checkpoint(
                    "group_enumeration",
                    discovered=len(elements),
                    frontier=len(queue),
                    seconds=round(time.perf_counter() - started, 3),
                )
    return elements


def logical_bases(code: codes.CSSCode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    pairing = (logical_z @ logical_x.T) % 2
    pairing_inverse = np.asarray(np.linalg.inv(pairing.view(code.field)), dtype=np.uint8)
    return logical_z, logical_x, pairing_inverse


def induced_z_action(
    permutation: np.ndarray,
    logical_z: np.ndarray,
    logical_x: np.ndarray,
    pairing_inverse: np.ndarray,
) -> np.ndarray:
    translated = permute_rows(logical_z, permutation)
    return ((translated @ logical_x.T) @ pairing_inverse) % 2


def action_group(
    order_four: np.ndarray, order_eight: np.ndarray
) -> list[np.ndarray]:
    return [
        (matrix_power(order_four, aa) @ matrix_power(order_eight, bb)) % 2
        for aa in range(TARGET_X_ORDER)
        for bb in range(TARGET_Y_ORDER)
    ]


def cyclic_vector(
    actions: Sequence[np.ndarray], *, trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    dimension = actions[0].shape[0]
    candidates = [np.eye(dimension, dtype=np.uint8)[row] for row in range(dimension)]
    candidates.extend(
        rng.integers(0, 2, size=dimension, dtype=np.uint8) for _ in range(trials)
    )
    for seed in candidates:
        orbit = np.asarray([(seed @ action) % 2 for action in actions], dtype=np.uint8)
        if gf2_rank(orbit) == dimension:
            return seed, orbit
    return None


def summarize_permutation(permutation: np.ndarray) -> dict[str, Any]:
    cycles: Counter[int] = Counter()
    unseen = set(range(len(permutation)))
    while unseen:
        start = min(unseen)
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            length += 1
            current = int(permutation[current])
        cycles[length] += 1
    return {
        "order": permutation_order(permutation),
        "cycle_length_counts": {str(key): value for key, value in sorted(cycles.items())},
    }


def search_regular_grid(
    elements: Sequence[np.ndarray],
    logical_z: np.ndarray,
    logical_x: np.ndarray,
    pairing_inverse: np.ndarray,
    *,
    seed_trials: int,
    random_seed: int,
    checkpoint_every: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    order_histogram: Counter[int] = Counter()
    physical_order_four: list[np.ndarray] = []
    physical_order_eight: list[np.ndarray] = []
    logical_action_cache: dict[bytes, np.ndarray] = {}

    for count, element in enumerate(elements, start=1):
        order = permutation_order(element)
        order_histogram[order] += 1
        if order % TARGET_X_ORDER == 0:
            action = induced_z_action(element, logical_z, logical_x, pairing_inverse)
            logical_action_cache[element.tobytes()] = action
            logical_order = matrix_order(action, maximum=order)
            if logical_order == TARGET_X_ORDER:
                physical_order_four.append(element)
            if logical_order == TARGET_Y_ORDER:
                physical_order_eight.append(element)
        if checkpoint_every and count % checkpoint_every == 0:
            checkpoint(
                "logical_order_screen",
                tested=count,
                logical_order_four=len(physical_order_four),
                logical_order_eight=len(physical_order_eight),
                seconds=round(time.perf_counter() - started, 3),
            )

    profile: dict[str, Any] = {
        "physical_order_histogram": {
            str(key): value for key, value in sorted(order_histogram.items())
        },
        "logical_order_four_elements": len(physical_order_four),
        "logical_order_eight_elements": len(physical_order_eight),
        "commuting_pairs_tested": 0,
        "pairs_with_32_distinct_actions": 0,
        "pairs_with_cyclic_vector": 0,
    }
    if not physical_order_eight:
        return None, profile

    rng = np.random.default_rng(random_seed)
    for xx in physical_order_four:
        action_x = logical_action_cache[xx.tobytes()]
        for yy in physical_order_eight:
            if not np.array_equal(compose(xx, yy), compose(yy, xx)):
                continue
            action_y = logical_action_cache[yy.tobytes()]
            if not np.array_equal(
                (action_x @ action_y) % 2,
                (action_y @ action_x) % 2,
            ):
                raise RuntimeError("commuting physical actions disagree on logical quotient")
            profile["commuting_pairs_tested"] += 1
            actions = action_group(action_x, action_y)
            if len({matrix_key(action) for action in actions}) != 32:
                continue
            profile["pairs_with_32_distinct_actions"] += 1
            witness = cyclic_vector(actions, trials=seed_trials, rng=rng)
            if witness is None:
                continue
            profile["pairs_with_cyclic_vector"] += 1
            seed, orbit = witness
            return {
                "x_physical_permutation": xx.astype(int).tolist(),
                "y_physical_permutation": yy.astype(int).tolist(),
                "x_physical_profile": summarize_permutation(xx),
                "y_physical_profile": summarize_permutation(yy),
                "x_logical_action": action_x.astype(int).tolist(),
                "y_logical_action": action_y.astype(int).tolist(),
                "seed_logical_coordinates": seed.astype(int).tolist(),
                "orbit_logical_coordinates": orbit.astype(int).tolist(),
                "orbit_rank": gf2_rank(orbit),
            }, profile
    return None, profile


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    hx, hz = build_checks()
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    commutator_weight = int(np.count_nonzero((hx @ hz.T) % 2))
    dimension = NUM_QUBITS - rank_x - rank_z
    code = codes.CSSCode(hx, hz)
    logical_z, logical_x, pairing_inverse = logical_bases(code)
    checkpoint(
        "code_reconstructed",
        n=NUM_QUBITS,
        k=dimension,
        rank_x=rank_x,
        rank_z=rank_z,
        commutator_weight=commutator_weight,
        check_weights_x=sorted(set(map(int, np.count_nonzero(hx, axis=1)))),
        check_weights_z=sorted(set(map(int, np.count_nonzero(hz, axis=1)))),
    )

    translation_x = two_half_permutation(group_permutation(1, 0))
    translation_y = two_half_permutation(group_permutation(0, 1))
    action_tx = induced_z_action(translation_x, logical_z, logical_x, pairing_inverse)
    action_ty = induced_z_action(translation_y, logical_z, logical_x, pairing_inverse)

    fold = two_half_permutation(inversion_permutation(), swap_halves=True)
    folded_hx = permute_rows(hx, fold)
    fold_maps_x_to_z = gf2_rank(np.vstack([hz, folded_hx])) == rank_z
    fold_maps_z_to_x = gf2_rank(np.vstack([hx, permute_rows(hz, fold)])) == rank_x

    graph = build_tanner_graph(hx, hz)
    components = graph_connected_components(graph)
    checkpoint(
        "tanner_graph_built",
        vertices=graph.number_of_vertices,
        components=len(components),
        component_sizes=sorted(map(len, components), reverse=True),
    )

    nauty_started = time.perf_counter()
    generators_raw, size_base, size_exponent, orbits, number_of_orbits = pynauty.autgrp(graph)
    nauty_seconds = time.perf_counter() - nauty_started
    group_size = nauty_group_size(size_base, size_exponent)
    data_generators = [
        np.asarray(generator[:NUM_QUBITS], dtype=np.int64)
        for generator in generators_raw
    ]
    checkpoint(
        "nauty_complete",
        generators=len(data_generators),
        group_size=group_size,
        vertex_orbits=number_of_orbits,
        seconds=round(nauty_seconds, 3),
    )

    generator_records = []
    for generator in data_generators:
        maps_x = gf2_rank(np.vstack([hx, permute_rows(hx, generator)])) == rank_x
        maps_z = gf2_rank(np.vstack([hz, permute_rows(hz, generator)])) == rank_z
        action = induced_z_action(generator, logical_z, logical_x, pairing_inverse)
        generator_records.append(
            {
                "permutation": generator.astype(int).tolist(),
                "physical_profile": summarize_permutation(generator),
                "logical_order": matrix_order(action, maximum=permutation_order(generator)),
                "maps_x_rowspace": maps_x,
                "maps_z_rowspace": maps_z,
            }
        )

    if group_size > args.enumeration_limit:
        elements = None
        checkpoint(
            "group_too_large_for_complete_element_screen",
            group_size=group_size,
            enumeration_limit=args.enumeration_limit,
        )
    else:
        elements = generate_group(
            data_generators,
            limit=args.enumeration_limit,
            checkpoint_every=args.checkpoint_every,
        )
        if elements is not None and len(elements) != group_size:
            raise RuntimeError(
                f"data restrictions generated {len(elements)} elements, nauty reports {group_size}"
            )

    grid_witness = None
    grid_profile: dict[str, Any] = {
        "complete_element_screen": elements is not None,
    }
    if elements is not None:
        grid_witness, searched = search_regular_grid(
            elements,
            logical_z,
            logical_x,
            pairing_inverse,
            seed_trials=args.cyclic_seed_trials,
            random_seed=args.random_seed,
            checkpoint_every=args.checkpoint_every,
        )
        grid_profile.update(searched)

    static = {
        "source": "Cruz-Benito et al., arXiv:2606.02418, Table 2 class j",
        "source_url": "https://arxiv.org/abs/2606.02418",
        "lattice": [ELL, EMM],
        "A_support": [list(term) for term in A_SUPPORT],
        "B_support": [list(term) for term in B_SUPPORT],
        "n": NUM_QUBITS,
        "k": dimension,
        "published_distance": 8,
        "distance_status": "published MILP exact; not recomputed by this script",
        "rank_x": rank_x,
        "rank_z": rank_z,
        "commutator_weight": commutator_weight,
        "check_weights_x": sorted(set(map(int, np.count_nonzero(hx, axis=1)))),
        "check_weights_z": sorted(set(map(int, np.count_nonzero(hz, axis=1)))),
        "tanner_components": len(components),
        "tanner_component_sizes": sorted(map(len, components), reverse=True),
        "native_translation_x_physical_order": permutation_order(translation_x),
        "native_translation_y_physical_order": permutation_order(translation_y),
        "native_translation_x_logical_order": matrix_order(action_tx),
        "native_translation_y_logical_order": matrix_order(action_ty),
        "native_translation_actions_commute": bool(
            np.array_equal((action_tx @ action_ty) % 2, (action_ty @ action_tx) % 2)
        ),
        "zx_fold_physical_order": permutation_order(fold),
        "zx_fold_maps_x_to_z": fold_maps_x_to_z,
        "zx_fold_maps_z_to_x": fold_maps_z_to_x,
    }
    automorphisms = {
        "scope": "full color-preserving automorphism group of the canonical translated-check Tanner graph",
        "nauty_generators": len(data_generators),
        "nauty_group_size": group_size,
        "nauty_group_size_scientific": [size_base, size_exponent],
        "nauty_vertex_orbits": number_of_orbits,
        "nauty_seconds": round(nauty_seconds, 6),
        "generator_records": generator_records,
        "grid_search": grid_profile,
        "regular_c4xc8_found": grid_witness is not None,
    }
    atomic_json(args.output_dir / "static-code.json", static)
    atomic_json(args.output_dir / "automorphisms.json", automorphisms)
    if grid_witness is not None:
        atomic_json(args.output_dir / "c4xc8-witness.json", grid_witness)
    np.savez_compressed(
        args.output_dir / "checks-and-native-symmetries.npz",
        matrix_x=hx,
        matrix_z=hz,
        translation_x=translation_x,
        translation_y=translation_y,
        zx_fold=fold,
        logical_z=logical_z,
        logical_x=logical_x,
        translation_x_logical=action_tx,
        translation_y_logical=action_ty,
    )
    summary = {
        **static,
        "tanner_automorphism_group_size": group_size,
        "complete_tanner_element_screen": elements is not None,
        "regular_c4xc8_found": grid_witness is not None,
        "grid_search": grid_profile,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    checkpoint("complete", **summary)


if __name__ == "__main__":
    main()
