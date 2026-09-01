#!/usr/bin/env python3
"""Analyze the verified ``[[288,40,8]]`` BB code for an Ising grid.

The exact Campaign-4 verification record of Cruz-Benito et al. gives

    A = 1 + x^2 + y^4 + x^2 y^4
    B = 1 + x^4 + y^8 + x^4 y^2

over ``F_2[C_16 x C_9]``.  The displayed class-e generators in Table 2 of
arXiv:2606.02418 are duplicated from the following class-f row and reconstruct
as ``[[288,36,8]]``; they are therefore not used here.  The native ``x^2``
translation has physical order eight.  The script reconstructs the verified
code, verifies its standard BB ZX fold, computes the exact color-preserving
Tanner automorphism group, and searches for a commuting logical order-four
partner supporting a 32-dimensional regular ``C_4 x C_8`` submodule inside the
40-dimensional logical space.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pynauty
from qldpc import codes
from qldpc.objects import Pauli


ELL = 16
EMM = 9
CELL_COUNT = ELL * EMM
NUM_QUBITS = 2 * CELL_COUNT
TARGET_GRID_DIMENSION = 32

A_SUPPORT = ((0, 0), (2, 0), (0, 4), (2, 4))
B_SUPPORT = ((0, 0), (4, 0), (0, 8), (4, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enumeration-limit", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed-trials", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=288408)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


def index(xx: int, yy: int) -> int:
    return (xx % ELL) * EMM + yy % EMM


def coordinate(position: int) -> tuple[int, int]:
    return divmod(position, EMM)


def group_permutation(dx: int, dy: int) -> np.ndarray:
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
    matrix = np.asarray(matrix, dtype=np.uint8)
    pivots: dict[int, int] = {}
    for row in matrix:
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
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


def matrix_order(matrix: np.ndarray, maximum: int = 4096) -> int | None:
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
    total = NUM_QUBITS + len(hx) + len(hz)
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
            set(range(NUM_QUBITS)),
            set(range(x_offset, z_offset)),
            set(range(z_offset, total)),
        ],
    )


def graph_components(graph: pynauty.Graph) -> list[list[int]]:
    unseen = set(range(graph.number_of_vertices))
    output: list[list[int]] = []
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
        output.append(sorted(component))
    return output


def nauty_group_size(size_base: float, size_exponent: int) -> int:
    return int(Decimal(str(size_base)) * (Decimal(10) ** int(size_exponent)))


def generate_group(
    generators: Sequence[np.ndarray],
    *,
    limit: int,
    checkpoint_every: int,
) -> list[np.ndarray] | None:
    identity = np.arange(NUM_QUBITS, dtype=np.int64)
    elements = [identity]
    seen = {identity.tobytes()}
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
                checkpoint("enumeration_limit", discovered=len(elements), limit=limit)
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


def action_group(action_x: np.ndarray, action_y: np.ndarray) -> list[np.ndarray]:
    return [
        (matrix_power(action_x, aa) @ matrix_power(action_y, bb)) % 2
        for aa in range(4)
        for bb in range(8)
    ]


def regular_socle(action_x: np.ndarray, action_y: np.ndarray) -> np.ndarray:
    """Socle operator for ``F_2[C_4 x C_8]`` on row vectors."""
    identity = np.eye(len(action_x), dtype=np.uint8)
    nilpotent_x = action_x ^ identity
    nilpotent_y = action_y ^ identity
    return (matrix_power(nilpotent_x, 3) @ matrix_power(nilpotent_y, 7)) % 2


def regular_seed_from_socle(
    action_x: np.ndarray,
    action_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return an exact rank-32 cyclic witness when the socle acts nontrivially."""
    socle = regular_socle(action_x, action_y)
    socle_rank = gf2_rank(socle)
    if socle_rank == 0:
        return None
    dimension = len(action_x)
    actions = action_group(action_x, action_y)
    if len({matrix_key(action) for action in actions}) != 32:
        raise RuntimeError("nonzero regular socle but fewer than 32 group actions")
    for row in range(dimension):
        if not np.any(socle[row]):
            continue
        seed = np.eye(dimension, dtype=np.uint8)[row]
        orbit = np.asarray([(seed @ action) % 2 for action in actions], dtype=np.uint8)
        if gf2_rank(orbit) != TARGET_GRID_DIMENSION:
            raise RuntimeError("nonzero socle seed does not generate a regular orbit")
        return seed, orbit, socle_rank
    raise RuntimeError("nonzero socle has no nonzero row")


def fold_partner_supports(grid_z: np.ndarray, fold: np.ndarray) -> np.ndarray:
    return permute_rows(grid_z, fold)


def search_fold_compatible_seed(
    action_x: np.ndarray,
    action_y: np.ndarray,
    logical_z: np.ndarray,
    fold: np.ndarray,
    *,
    random_seed: int,
    trials: int,
) -> dict[str, Any]:
    """Sample cyclic seeds and score the ZX-fold pairing on the 32-grid."""
    actions = action_group(action_x, action_y)
    socle = regular_socle(action_x, action_y)
    dimension = len(action_x)
    rng = np.random.default_rng(random_seed)
    seeds = [np.eye(dimension, dtype=np.uint8)[row] for row in range(dimension)]
    seeds.extend(rng.integers(0, 2, size=dimension, dtype=np.uint8) for _ in range(trials))
    best = None
    histogram: Counter[int] = Counter()
    cyclic_tested = 0
    for seed in seeds:
        if not np.any((seed @ socle) % 2):
            continue
        classes = np.asarray([(seed @ action) % 2 for action in actions], dtype=np.uint8)
        if gf2_rank(classes) != TARGET_GRID_DIMENSION:
            raise RuntimeError("socle-qualified seed is not cyclic")
        cyclic_tested += 1
        grid_z = (classes @ logical_z) % 2
        grid_x = fold_partner_supports(grid_z, fold)
        pairing = (grid_z @ grid_x.T) % 2
        rank = gf2_rank(pairing)
        histogram[rank] += 1
        row_weights = np.count_nonzero(pairing, axis=1)
        column_weights = np.count_nonzero(pairing, axis=0)
        monomial = bool(np.all(row_weights == 1) and np.all(column_weights == 1))
        score = (rank, monomial, -int(np.max(row_weights, initial=0)), -int(np.sum(row_weights)))
        if best is None or score > best[0]:
            best = (
                score,
                {
                    "seed_logical_coordinates": seed.astype(int).tolist(),
                    "orbit_logical_coordinates": classes.astype(int).tolist(),
                    "grid_z_supports": [
                        np.flatnonzero(row).astype(int).tolist() for row in grid_z
                    ],
                    "grid_z_weights": np.count_nonzero(grid_z, axis=1).astype(int).tolist(),
                    "zx_pairing_rank": rank,
                    "zx_pairing_monomial": monomial,
                    "zx_pairing_row_weights": row_weights.astype(int).tolist(),
                    "zx_pairing_column_weights": column_weights.astype(int).tolist(),
                    "zx_pairing": pairing.astype(int).tolist(),
                },
            )
            if monomial:
                break
    return {
        "cyclic_seeds_tested": cyclic_tested,
        "pairing_rank_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "best": None if best is None else best[1],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    hx, hz = build_checks()
    rank_x = gf2_rank(hx)
    rank_z = gf2_rank(hz)
    dimension = NUM_QUBITS - rank_x - rank_z
    commutator_weight = int(np.count_nonzero((hx @ hz.T) % 2))
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
    native_order_eight = permutation_power(translation_x, 2)
    action_tx = induced_z_action(translation_x, logical_z, logical_x, pairing_inverse)
    action_ty = induced_z_action(translation_y, logical_z, logical_x, pairing_inverse)
    action_native_y8 = induced_z_action(native_order_eight, logical_z, logical_x, pairing_inverse)

    fold = two_half_permutation(inversion_permutation(), swap_halves=True)
    fold_maps_x_to_z = gf2_rank(np.vstack([hz, permute_rows(hx, fold)])) == rank_z
    fold_maps_z_to_x = gf2_rank(np.vstack([hx, permute_rows(hz, fold)])) == rank_x

    graph = build_tanner_graph(hx, hz)
    components = graph_components(graph)
    checkpoint(
        "tanner_graph",
        vertices=graph.number_of_vertices,
        components=len(components),
        component_sizes=sorted(map(len, components), reverse=True),
    )

    nauty_started = time.perf_counter()
    raw_generators, size_base, size_exponent, _orbits, number_of_orbits = pynauty.autgrp(graph)
    nauty_seconds = time.perf_counter() - nauty_started
    group_size = nauty_group_size(size_base, size_exponent)
    generators = [
        np.asarray(generator[:NUM_QUBITS], dtype=np.int64)
        for generator in raw_generators
    ]
    checkpoint(
        "nauty_complete",
        generators=len(generators),
        group_size=group_size,
        vertex_orbits=number_of_orbits,
        seconds=round(nauty_seconds, 3),
    )

    elements = None
    if group_size <= args.enumeration_limit:
        elements = generate_group(
            generators,
            limit=args.enumeration_limit,
            checkpoint_every=args.checkpoint_every,
        )
        if elements is not None and len(elements) != group_size:
            raise RuntimeError(f"generated {len(elements)} data actions, nauty reports {group_size}")
    else:
        checkpoint(
            "group_too_large",
            group_size=group_size,
            enumeration_limit=args.enumeration_limit,
        )

    search_profile: dict[str, Any] = {"complete_group_enumeration": elements is not None}
    witness = None
    if elements is not None:
        actions: dict[bytes, np.ndarray] = {}
        logical_orders: dict[bytes, int | None] = {}
        physical_histogram: Counter[int] = Counter()
        order_four = []
        order_eight = []
        for count, element in enumerate(elements, start=1):
            physical_order = permutation_order(element)
            physical_histogram[physical_order] += 1
            if physical_order % 4:
                continue
            action = induced_z_action(element, logical_z, logical_x, pairing_inverse)
            actions[element.tobytes()] = action
            logical_order = matrix_order(action, maximum=physical_order)
            logical_orders[element.tobytes()] = logical_order
            if logical_order == 4:
                order_four.append(element)
            if logical_order == 8:
                order_eight.append(element)
            if args.checkpoint_every and count % args.checkpoint_every == 0:
                checkpoint(
                    "logical_order_screen",
                    tested=count,
                    order_four=len(order_four),
                    order_eight=len(order_eight),
                )
        actions[native_order_eight.tobytes()] = action_native_y8
        logical_orders[native_order_eight.tobytes()] = matrix_order(action_native_y8, maximum=8)

        counters: Counter[str] = Counter()
        pairs: list[tuple[np.ndarray, np.ndarray, str]] = [
            (xx, native_order_eight, "native_x_squared") for xx in order_four
        ]
        pairs.extend((xx, yy, "all_order_eight") for yy in order_eight for xx in order_four)
        seen_pairs: set[tuple[bytes, bytes]] = set()
        for pair_count, (xx, yy, source) in enumerate(pairs, start=1):
            pair_key = (xx.tobytes(), yy.tobytes())
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            action_x = actions[xx.tobytes()]
            action_y = actions[yy.tobytes()]
            physical_commutes = np.array_equal(compose(xx, yy), compose(yy, xx))
            logical_commutes = np.array_equal(
                (action_x @ action_y) % 2,
                (action_y @ action_x) % 2,
            )
            if not logical_commutes:
                continue
            counters["logically_commuting_pairs"] += 1
            if physical_commutes:
                counters["physically_commuting_pairs"] += 1
            regular = regular_seed_from_socle(action_x, action_y)
            if regular is None:
                continue
            counters["regular_submodule_pairs"] += 1
            seed, orbit, socle_rank = regular
            fold_screen = search_fold_compatible_seed(
                action_x,
                action_y,
                logical_z,
                fold,
                random_seed=args.random_seed + counters["regular_submodule_pairs"],
                trials=args.seed_trials,
            )
            best_fold = fold_screen["best"]
            witness = {
                "pair_source": source,
                "x_physical_permutation": xx.astype(int).tolist(),
                "y_physical_permutation": yy.astype(int).tolist(),
                "x_physical_order": permutation_order(xx),
                "y_physical_order": permutation_order(yy),
                "x_logical_action": action_x.astype(int).tolist(),
                "y_logical_action": action_y.astype(int).tolist(),
                "x_logical_order": matrix_order(action_x),
                "y_logical_order": matrix_order(action_y),
                "physical_actions_commute": physical_commutes,
                "logical_actions_commute": logical_commutes,
                "regular_socle_rank": socle_rank,
                "seed_logical_coordinates": seed.astype(int).tolist(),
                "orbit_logical_coordinates": orbit.astype(int).tolist(),
                "orbit_rank": gf2_rank(orbit),
                "fold_seed_screen": fold_screen,
            }
            atomic_json(args.output_dir / "c4xc8-witness.json", witness)
            checkpoint(
                "regular_grid_found",
                pair_source=source,
                physical_commutes=physical_commutes,
                socle_rank=socle_rank,
                zx_pairing_rank=None if best_fold is None else best_fold["zx_pairing_rank"],
                zx_pairing_monomial=None if best_fold is None else best_fold["zx_pairing_monomial"],
            )
            if best_fold is not None and best_fold["zx_pairing_rank"] == 32:
                break
        search_profile.update(
            {
                "physical_order_histogram": {
                    str(key): value for key, value in sorted(physical_histogram.items())
                },
                "logical_order_four_elements": len(order_four),
                "logical_order_eight_elements": len(order_eight),
                "pairs_considered": len(seen_pairs),
                "counters": dict(counters),
            }
        )

    static = {
        "source": (
            "Cruz-Benito et al., qcode-discovery, "
            "results/campaign4_milp_verified.jsonl line 515"
        ),
        "source_url": (
            "https://github.com/qiskit-community/qcode-discovery/blob/main/"
            "results/campaign4_milp_verified.jsonl"
        ),
        "paper_table_discrepancy": (
            "Table 2 class-e generators duplicate class f and reconstruct as "
            "[[288,36,8]]; run_001 preserves that diagnostic"
        ),
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
        "native_x_physical_order": permutation_order(translation_x),
        "native_y_physical_order": permutation_order(translation_y),
        "native_x_logical_order": matrix_order(action_tx),
        "native_y_logical_order": matrix_order(action_ty),
        "native_x_squared_logical_order": matrix_order(action_native_y8),
        "native_translation_actions_commute": bool(
            np.array_equal((action_tx @ action_ty) % 2, (action_ty @ action_tx) % 2)
        ),
        "zx_fold_physical_order": permutation_order(fold),
        "zx_fold_maps_x_to_z": fold_maps_x_to_z,
        "zx_fold_maps_z_to_x": fold_maps_z_to_x,
    }
    summary = {
        **static,
        "tanner_automorphism_group_size": group_size,
        "tanner_automorphism_generators": len(generators),
        "nauty_seconds": round(nauty_seconds, 6),
        "search": search_profile,
        "regular_c4xc8_found": witness is not None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    np.savez_compressed(
        args.output_dir / "checks-native-and-logicals.npz",
        matrix_x=hx,
        matrix_z=hz,
        translation_x=translation_x,
        translation_y=translation_y,
        native_order_eight=native_order_eight,
        zx_fold=fold,
        logical_z=logical_z,
        logical_x=logical_x,
    )
    checkpoint("complete", **summary)


if __name__ == "__main__":
    main()
