"""Independent verification of the C4 x C8 automorphism-fold [[160,32,6]] code."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cvxpy as cp
import networkx as nx
import numpy as np
import pynauty

import code


Z_MILP_WITNESS = [34, 47, 56, 105, 108, 125]
X_MILP_WITNESS = [84, 87, 91, 110, 119, 126]
Z_ORBIT_SEED = [0, 8, 20, 64, 74, 77]
X_ORBIT_SEED = [0, 14, 16, 32, 41, 44]


def rowspace_basis(matrix: np.ndarray) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return pivots


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        code.gf2_rank(left)
        == code.gf2_rank(right)
        == code.gf2_rank(np.vstack([left, right]))
    )


def permute_columns(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def is_permutation_matrix(matrix: np.ndarray) -> bool:
    return bool(np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1))


def connected(hx: np.ndarray, hz: np.ndarray) -> bool:
    total = code.NUM_QUBITS + hx.shape[0] + hz.shape[0]
    adjacency = [[] for _ in range(total)]
    for offset, checks in ((code.NUM_QUBITS, hx), (code.NUM_QUBITS + hx.shape[0], hz)):
        for row, support in enumerate(checks):
            check = offset + row
            for qubit in np.flatnonzero(support):
                qubit = int(qubit)
                adjacency[check].append(qubit)
                adjacency[qubit].append(check)
    seen = {0}
    queue = deque([0])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == total


def canonical_graph(hx: np.ndarray, hz: np.ndarray, *, merge_css: bool) -> pynauty.Graph:
    xo = code.NUM_QUBITS
    zo = xo + hx.shape[0]
    total = zo + hz.shape[0]
    adjacency = {vertex: [] for vertex in range(total)}
    for offset, checks in ((xo, hx), (zo, hz)):
        for row, support in enumerate(checks):
            check = offset + row
            for qubit in np.flatnonzero(support):
                qubit = int(qubit)
                adjacency[check].append(qubit)
                adjacency[qubit].append(check)
    colors = (
        [set(range(code.NUM_QUBITS)), set(range(xo, total))]
        if merge_css
        else [set(range(code.NUM_QUBITS)), set(range(xo, zo)), set(range(zo, total))]
    )
    return pynauty.Graph(
        number_of_vertices=total,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=colors,
    )


def nauty_group_size(result) -> int:
    return int(round(result[1])) * 10 ** int(result[2])


def column_masks(matrix: np.ndarray) -> list[int]:
    return [
        int.from_bytes(np.packbits(column, bitorder="little").tobytes(), "little")
        for column in np.asarray(matrix, dtype=np.uint8).T
    ]


def low_logical_through_five(
    check: np.ndarray, conjugate_logicals: np.ndarray
) -> dict:
    """Complete syndrome/signature search through weight five."""
    started = time.perf_counter()
    syndromes = column_masks(check)
    logicals = column_masks(conjugate_logicals)
    singles: dict[int, list[int]] = defaultdict(list)
    for qubit, syndrome in enumerate(syndromes):
        singles[syndrome].append(qubit)
        if syndrome == 0 and logicals[qubit] != 0:
            return {"found": True, "weight": 1, "support": [qubit]}

    pairs: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for left in range(code.NUM_QUBITS):
        for right in range(left + 1, code.NUM_QUBITS):
            syndrome = syndromes[left] ^ syndromes[right]
            logical = logicals[left] ^ logicals[right]
            mask = (1 << left) | (1 << right)
            pairs[syndrome].append((left, right, logical, mask))
            if syndrome == 0 and logical != 0:
                return {"found": True, "weight": 2, "support": [left, right]}

    for left in range(code.NUM_QUBITS):
        for right in range(left + 1, code.NUM_QUBITS):
            target = syndromes[left] ^ syndromes[right]
            logical = logicals[left] ^ logicals[right]
            for third in singles.get(target, ()):
                if third > right and logical ^ logicals[third]:
                    return {"found": True, "weight": 3, "support": [left, right, third]}

    for same_syndrome in pairs.values():
        for index, (aa, bb, logical_left, mask_left) in enumerate(same_syndrome):
            for cc, dd, logical_right, mask_right in same_syndrome[index + 1 :]:
                if mask_left & mask_right:
                    continue
                if logical_left ^ logical_right:
                    return {
                        "found": True,
                        "weight": 4,
                        "support": sorted((aa, bb, cc, dd)),
                    }

    for aa in range(code.NUM_QUBITS):
        for bb in range(aa + 1, code.NUM_QUBITS):
            pair_syndrome = syndromes[aa] ^ syndromes[bb]
            pair_logical = logicals[aa] ^ logicals[bb]
            pair_mask = (1 << aa) | (1 << bb)
            for cc in range(bb + 1, code.NUM_QUBITS):
                target = pair_syndrome ^ syndromes[cc]
                triple_logical = pair_logical ^ logicals[cc]
                triple_mask = pair_mask | (1 << cc)
                for dd, ee, other_logical, other_mask in pairs.get(target, ()):
                    if triple_mask & other_mask:
                        continue
                    if triple_logical ^ other_logical:
                        return {
                            "found": True,
                            "weight": 5,
                            "support": sorted((aa, bb, cc, dd, ee)),
                        }
    return {
        "found": False,
        "maximum_weight": 5,
        "seconds": round(time.perf_counter() - started, 6),
    }


def logical_milp(
    check: np.ndarray,
    conjugate_logicals: np.ndarray,
    *,
    maximum_weight: int,
    time_limit: float,
) -> dict:
    error = cp.Variable(code.NUM_QUBITS, boolean=True)
    syndrome_slack = cp.Variable(check.shape[0], integer=True)
    logical_parities = cp.Variable(conjugate_logicals.shape[0], boolean=True)
    logical_slack = cp.Variable(conjugate_logicals.shape[0], integer=True)
    constraints = [
        check.astype(int) @ error == 2 * syndrome_slack,
        syndrome_slack >= 0,
        syndrome_slack <= np.sum(check, axis=1) // 2,
        conjugate_logicals.astype(int) @ error == logical_parities + 2 * logical_slack,
        logical_slack >= 0,
        logical_slack <= np.sum(conjugate_logicals, axis=1) // 2,
        cp.sum(logical_parities) >= 1,
        cp.sum(error) <= maximum_weight,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(error)), constraints)
    started = time.perf_counter()
    value = problem.solve(
        solver="HIGHS", highs_options={"time_limit": float(time_limit)}
    )
    result = {
        "solver": "HIGHS",
        "status": problem.status,
        "maximum_weight": maximum_weight,
        "seconds": round(time.perf_counter() - started, 6),
    }
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return {**result, "infeasible": True, "weight": None, "support": None}
    if problem.status != cp.OPTIMAL or error.value is None or not np.isfinite(value):
        return {**result, "infeasible": False, "weight": None, "support": None}
    solution = np.rint(error.value).astype(np.uint8)
    return {
        **result,
        "infeasible": False,
        "weight": int(np.count_nonzero(solution)),
        "support": np.flatnonzero(solution).astype(int).tolist(),
    }


def translate_support(support: list[int], shift: tuple[int, int]) -> list[int]:
    permutation = code.translation_permutation(shift)
    return sorted(int(permutation[index]) for index in support)


def translated_orbit(support: list[int]) -> list[list[int]]:
    return [translate_support(support, element) for element in code.GROUP]


def maximum_disjoint_collections(supports: list[list[int]]) -> list[list[int]]:
    compatibility = nx.Graph()
    compatibility.add_nodes_from(range(len(supports)))
    sets = list(map(set, supports))
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            if sets[left].isdisjoint(sets[right]):
                compatibility.add_edge(left, right)
    maximal = [sorted(clique) for clique in nx.find_cliques(compatibility)]
    maximum = max(map(len, maximal))
    return [clique for clique in maximal if len(clique) == maximum]


def logical_support_audit(logical_z: np.ndarray, logical_x: np.ndarray) -> dict:
    orbit_z = translated_orbit(Z_ORBIT_SEED)
    orbit_x = translated_orbit(X_ORBIT_SEED)
    vectors_z = np.zeros((len(orbit_z), code.NUM_QUBITS), dtype=np.uint8)
    vectors_x = np.zeros((len(orbit_x), code.NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(orbit_z):
        vectors_z[row, support] = 1
    for row, support in enumerate(orbit_x):
        vectors_x[row, support] = 1
    signatures_z = (vectors_z @ logical_x.T) % 2
    signatures_x = (vectors_x @ logical_z.T) % 2
    collections_z = maximum_disjoint_collections(orbit_z)
    collections_x = maximum_disjoint_collections(orbit_x)
    best_z = collections_z[0]
    best_x = collections_x[0]

    paired = None
    for size in range(len(best_z), 0, -1):
        subsets_z = sorted(
            {
                tuple(sorted(subset))
                for collection in collections_z
                for subset in itertools.combinations(collection, size)
            }
        )
        subsets_x = sorted(
            {
                tuple(sorted(subset))
                for collection in collections_x
                for subset in itertools.combinations(collection, size)
            }
        )
        for zz in subsets_z:
            for xx in subsets_x:
                pairing = (vectors_z[list(zz)] @ vectors_x[list(xx)].T) % 2
                if is_permutation_matrix(pairing):
                    paired = {
                        "size": size,
                        "z_indices": list(zz),
                        "x_indices": list(xx),
                        "z_group_labels": [list(code.GROUP[index]) for index in zz],
                        "x_group_labels": [list(code.GROUP[index]) for index in xx],
                        "z_supports": [orbit_z[index] for index in zz],
                        "x_supports": [orbit_x[index] for index in xx],
                        "pairing": pairing.astype(int).tolist(),
                    }
                    break
            if paired is not None:
                break
        if paired is not None:
            break

    return {
        "orbit_seed_z": Z_ORBIT_SEED,
        "orbit_seed_x": X_ORBIT_SEED,
        "orbit_size_z": len({tuple(support) for support in orbit_z}),
        "orbit_size_x": len({tuple(support) for support in orbit_x}),
        "orbit_logical_rank_z": code.gf2_rank(signatures_z),
        "orbit_logical_rank_x": code.gf2_rank(signatures_x),
        "maximum_disjoint_count_z": len(best_z),
        "maximum_disjoint_count_x": len(best_x),
        "maximum_disjoint_z_indices": best_z,
        "maximum_disjoint_x_indices": best_x,
        "maximum_disjoint_z_group_labels": [list(code.GROUP[index]) for index in best_z],
        "maximum_disjoint_x_group_labels": [list(code.GROUP[index]) for index in best_x],
        "maximum_disjoint_z_supports": [orbit_z[index] for index in best_z],
        "maximum_disjoint_x_supports": [orbit_x[index] for index in best_x],
        "maximum_disjoint_z_logical_rank": code.gf2_rank(signatures_z[best_z]),
        "maximum_disjoint_x_logical_rank": code.gf2_rank(signatures_x[best_x]),
        "best_disjoint_symplectic_pairs": paired,
        "scope": "translated weight-six orbit; no claim over arbitrary stabilizer dressings",
    }


def algebra_audit(hx, hz, logical_z, logical_x) -> dict:
    identity = np.eye(code.GROUP_ORDER, dtype=np.uint8)
    translations = []
    for name, shift, order in (("C4", (1, 0), 4), ("C8", (0, 1), 8)):
        permutation = code.translation_permutation(shift)
        moved_z = code.permute_rows(logical_z, permutation)
        logical_action = moved_z[:, : code.GROUP_ORDER]
        translations.append(
            {
                "name": name,
                "shift": list(shift),
                "order": order,
                "preserves_x_check_space": same_rowspace(permute_columns(hx, permutation), hx),
                "preserves_z_check_space": same_rowspace(permute_columns(hz, permutation), hz),
                "logical_action_is_permutation": is_permutation_matrix(logical_action),
                "exact_action_on_logical_basis": bool(
                    np.array_equal(moved_z, logical_action @ logical_z)
                ),
            }
        )
    fold = code.fold_permutation()
    moved_z = code.permute_rows(logical_z, fold)
    moved_x = code.permute_rows(logical_x, fold)
    z_to_x = (moved_z @ logical_z.T) % 2
    x_to_z = (moved_x @ logical_x.T) % 2
    css_automorphisms = pynauty.autgrp(canonical_graph(hx, hz, merge_css=False))
    merged_automorphisms = pynauty.autgrp(canonical_graph(hx, hz, merge_css=True))
    return {
        "n": int(hx.shape[1]),
        "rank_x": code.gf2_rank(hx),
        "rank_z": code.gf2_rank(hz),
        "k": int(hx.shape[1] - code.gf2_rank(hx) - code.gf2_rank(hz)),
        "css_commutation": bool(not np.any((hx @ hz.T) % 2)),
        "connected": connected(hx, hz),
        "row_weights_x": sorted(set(map(int, hx.sum(axis=1)))),
        "row_weights_z": sorted(set(map(int, hz.sum(axis=1)))),
        "logical_z_cycles": bool(not np.any((logical_z @ hx.T) % 2)),
        "logical_x_cycles": bool(not np.any((logical_x @ hz.T) % 2)),
        "logical_pairing_identity": bool(
            np.array_equal((logical_z @ logical_x.T) % 2, identity)
        ),
        "logical_weights_z": sorted(set(map(int, logical_z.sum(axis=1)))),
        "logical_weights_x": sorted(set(map(int, logical_x.sum(axis=1)))),
        "translations": translations,
        "translations_commute": bool(
            np.array_equal(
                code.translation_permutation((1, 0))[code.translation_permutation((0, 1))],
                code.translation_permutation((0, 1))[code.translation_permutation((1, 0))],
            )
        ),
        "fold_involution": bool(np.array_equal(fold[fold], np.arange(code.NUM_QUBITS))),
        "fold_maps_x_to_z": same_rowspace(permute_columns(hx, fold), hz),
        "fold_maps_z_to_x": same_rowspace(permute_columns(hz, fold), hx),
        "hadamard_z_to_x_is_permutation": is_permutation_matrix(z_to_x),
        "hadamard_x_to_z_is_permutation": is_permutation_matrix(x_to_z),
        "hadamard_actions_are_transposes": bool(np.array_equal(z_to_x, x_to_z.T)),
        "logical_hadamard_permutation": [int(np.argmax(row)) for row in z_to_x],
        "logical_hadamard_formula": "(r,s) -> (r,-2*r-s mod 8)",
        "canonical_css_preserving_tanner_group_order": nauty_group_size(css_automorphisms),
        "canonical_css_merged_tanner_group_order": nauty_group_size(merged_automorphisms),
    }


def witness_valid(check, conjugate_logicals, support) -> bool:
    vector = np.zeros(code.NUM_QUBITS, dtype=np.uint8)
    vector[support] = 1
    return bool(
        not np.any((check @ vector) % 2)
        and np.any((conjugate_logicals @ vector) % 2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-certificates", action="store_true")
    parser.add_argument("--skip-milp", action="store_true")
    parser.add_argument("--milp-seconds", type=float, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = Path(__file__).resolve().parent
    hx, hz = code.build_checks()
    logical_z, logical_x = code.build_logical_bases()
    algebra = algebra_audit(hx, hz, logical_z, logical_x)
    supports = logical_support_audit(logical_z, logical_x)
    enumeration = {
        "Z": low_logical_through_five(hx, logical_x),
        "X": low_logical_through_five(hz, logical_z),
    }
    distance = {
        "parameters": "[[160,32,6]]",
        "meet_in_the_middle": enumeration,
        "stored_witnesses": {
            "Z": {
                "support": Z_MILP_WITNESS,
                "weight": len(Z_MILP_WITNESS),
                "valid": witness_valid(hx, logical_x, Z_MILP_WITNESS),
            },
            "X": {
                "support": X_MILP_WITNESS,
                "weight": len(X_MILP_WITNESS),
                "valid": witness_valid(hz, logical_z, X_MILP_WITNESS),
            },
        },
        "analytic_upper_bound": {
            "value": 6,
            "source": "Theorem D.3 of arXiv:2607.27644v1",
        },
    }
    if not args.skip_milp:
        distance["milp"] = {
            "Z_lower_bound": logical_milp(
                hx, logical_x, maximum_weight=5, time_limit=args.milp_seconds
            ),
            "X_lower_bound": logical_milp(
                hz, logical_z, maximum_weight=5, time_limit=args.milp_seconds
            ),
            "Z_exact": logical_milp(
                hx, logical_x, maximum_weight=6, time_limit=args.milp_seconds
            ),
            "X_exact": logical_milp(
                hz, logical_z, maximum_weight=6, time_limit=args.milp_seconds
            ),
        }
    assert algebra["n"] == 160 and algebra["k"] == 32
    assert algebra["css_commutation"] and algebra["connected"]
    assert algebra["row_weights_x"] == [9] and algebra["row_weights_z"] == [9]
    assert algebra["logical_pairing_identity"]
    assert algebra["canonical_css_preserving_tanner_group_order"] == 32
    assert algebra["canonical_css_merged_tanner_group_order"] == 64
    assert algebra["fold_maps_x_to_z"] and algebra["fold_maps_z_to_x"]
    assert algebra["hadamard_z_to_x_is_permutation"]
    assert algebra["hadamard_x_to_z_is_permutation"]
    assert not enumeration["Z"]["found"] and not enumeration["X"]["found"]
    assert distance["stored_witnesses"]["Z"]["valid"]
    assert distance["stored_witnesses"]["X"]["valid"]
    if "milp" in distance:
        assert distance["milp"]["Z_lower_bound"]["infeasible"]
        assert distance["milp"]["X_lower_bound"]["infeasible"]
        assert distance["milp"]["Z_exact"]["weight"] == 6
        assert distance["milp"]["X_exact"]["weight"] == 6
    if args.write_certificates:
        (directory / "automorphism_certificate.json").write_text(
            json.dumps(algebra, indent=2, sort_keys=True) + "\n"
        )
        (directory / "distance_certificate.json").write_text(
            json.dumps(distance, indent=2, sort_keys=True) + "\n"
        )
        (directory / "logical_supports.json").write_text(
            json.dumps(supports, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({
        "verified": True,
        "algebra": algebra,
        "distance": distance,
        "logical_support_summary": {
            "orbit_rank_z": supports["orbit_logical_rank_z"],
            "orbit_rank_x": supports["orbit_logical_rank_x"],
            "maximum_disjoint_z": supports["maximum_disjoint_count_z"],
            "maximum_disjoint_x": supports["maximum_disjoint_count_x"],
            "best_disjoint_symplectic_pairs": supports["best_disjoint_symplectic_pairs"]["size"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
