"""Joint ZX-pairing rescreen for the earlier weight-16 S3 searches."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from qldpc.objects import Pauli

from .s3_ising import (
    BOTTOM_ORDER,
    ProductMonomial,
    S3L4W16Candidate,
    S3L8W16Candidate,
    _gf2_rank,
    _permute_columns,
    _zx_fold_permutation,
    analyze_translation_logical_seed,
    build_s3_l4_w16_code,
    build_s3_l8_w16_code,
    find_zero_syndrome_by_tanner_search,
    logical_translation_orbit,
)

Family = Literal["l4-w16", "l8-w16"]
Candidate = S3L4W16Candidate | S3L8W16Candidate


def saved_extended_near_miss_candidate() -> S3L8W16Candidate:
    """Return the saved ``[[768,388,6]]`` two-grid near miss."""
    pp = ProductMonomial
    return S3L8W16Candidate(
        tuple(sorted((pp(0, 0, 0), pp(2, 4, 3)))),
        tuple(sorted((pp(0, 7, 3), pp(4, 6, 0)))),
        tuple(sorted((pp(2, 1, 0), pp(4, 0, 3)))),
        tuple(sorted((pp(0, 7, 3), pp(4, 7, 3)))),
    )


def candidate_from_record(record: dict[str, Any], family: Family) -> Candidate:
    """Reconstruct a saved candidate from one search JSON record."""
    data = record["candidate"]

    def entry(name: str) -> tuple[ProductMonomial, ...]:
        return tuple(ProductMonomial(**term) for term in data[name])

    if family == "l4-w16":
        return S3L4W16Candidate(f0=entry("f0"), f1=entry("f1"))
    return S3L8W16Candidate(
        entry("f0"), entry("f1"), entry("f2"), entry("f3")
    )


def build_candidate_code(candidate: Candidate):
    """Build either supported saved-search family."""
    if isinstance(candidate, S3L4W16Candidate):
        return build_s3_l4_w16_code(candidate)
    return build_s3_l8_w16_code(candidate)


def enumerate_translation_grid_orbits(
    code,
    *,
    weight: int = 6,
    max_orbits: int = 500,
    max_nodes: int = 5_000_000,
) -> dict[str, Any]:
    """Exhaustively enumerate graph-supported kernels modulo translation."""
    supports: list[list[int]] = []
    grids: list[dict[str, Any]] = []
    exhaustive = False
    last_search: dict[str, Any] | None = None
    for _index in range(max_orbits):
        last_search = find_zero_syndrome_by_tanner_search(
            code,
            weight=weight,
            max_nodes=max_nodes,
            require_graph_support=True,
            excluded_translation_orbits=supports,
        )
        if last_search is None or last_search.get("support") is None:
            exhaustive = bool(last_search and last_search.get("search_exhaustive"))
            break
        support = list(map(int, last_search["support"]))
        supports.append(support)
        grid = analyze_translation_logical_seed(code, support)
        grids.append(
            {
                "seed_support": support,
                "internal_orbits": grid["internal_orbits"],
                "graph_supported": grid["graph_supported"],
                "pairwise_disjoint": grid["pairwise_disjoint"],
                "orbit_rank_mod_stabilizers": grid[
                    "orbit_rank_mod_stabilizers"
                ],
                "self_zx_pairing_rank": grid["zx_pairing_rank"],
                "is_nontrivial_logical": bool(
                    last_search.get("is_nontrivial_logical")
                ),
            }
        )
    else:
        last_search = None
    return {
        "weight": weight,
        "num_orbits": len(grids),
        "enumeration_exhaustive": exhaustive,
        "truncated": len(grids) >= max_orbits and not exhaustive,
        "last_search": last_search,
        "self_pairing_rank_counts": dict(
            sorted(Counter(grid["self_zx_pairing_rank"] for grid in grids).items())
        ),
        "orbit_rank_counts": dict(
            sorted(
                Counter(
                    grid["orbit_rank_mod_stabilizers"] for grid in grids
                ).items()
            )
        ),
        "grids": grids,
    }


def _logical_grid_arrays(code, grids: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    num_blocks = code.num_qubits // (3 * BOTTOM_ORDER)
    arrays: list[np.ndarray] = []
    for grid in grids:
        seed = np.zeros(code.num_qubits, dtype=np.uint8)
        seed[grid["seed_support"]] = 1
        arrays.append(logical_translation_orbit(seed, num_blocks=num_blocks))
    return arrays


def _permutation_pair_map(pairing: np.ndarray) -> list[list[int]] | None:
    """Return `(source,target,dx,dy)` grid maps for a permutation pairing."""
    if not (
        np.all(np.count_nonzero(pairing, axis=0) == 1)
        and np.all(np.count_nonzero(pairing, axis=1) == 1)
    ):
        return None
    num_grids = pairing.shape[0] // BOTTOM_ORDER
    maps: list[list[int]] = []
    for source in range(num_grids):
        relations: set[tuple[int, int, int]] = set()
        for xx in range(8):
            for yy in range(4):
                row = source * BOTTOM_ORDER + xx * 4 + yy
                column = int(np.flatnonzero(pairing[row])[0])
                target, remainder = divmod(column, BOTTOM_ORDER)
                target_x, target_y = divmod(remainder, 4)
                relations.add(
                    (target, (target_x - xx) % 8, (target_y - yy) % 4)
                )
        if len(relations) != 1:
            return None
        target, delta_x, delta_y = next(iter(relations))
        maps.append([source, target, delta_x, delta_y])
    return maps


def analyze_joint_grid_combinations(
    code,
    grids: Sequence[dict[str, Any]],
    *,
    fold_permutation: Sequence[int] | None = None,
    maximum_grids: int,
) -> dict[str, Any]:
    """Test all disjoint grid combinations through `maximum_grids`."""
    num_blocks = code.num_qubits // (3 * BOTTOM_ORDER)
    fold = np.asarray(
        _zx_fold_permutation(num_blocks)
        if fold_permutation is None
        else fold_permutation,
        dtype=int,
    )
    arrays = _logical_grid_arrays(code, grids)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    coordinates = [(orbit @ logical_x.T) % 2 for orbit in arrays]
    folded = [_permute_columns(orbit, fold) for orbit in arrays]
    cross_pairings = {
        (left, right): (arrays[left] @ folded[right].T) % 2
        for left in range(len(arrays))
        for right in range(len(arrays))
    }
    internal = [set(map(int, grid["internal_orbits"])) for grid in grids]
    summaries: dict[str, Any] = {}
    hits: list[dict[str, Any]] = []
    for size in range(1, maximum_grids + 1):
        tested = 0
        disjoint = 0
        independent = 0
        best_logical_rank = 0
        best_logical_combination: tuple[int, ...] | None = None
        best_pairing_rank_any = 0
        best_pairing_combination_any: tuple[int, ...] | None = None
        best_pairing_rank_independent = 0
        best_independent_combination: tuple[int, ...] | None = None
        full_rank_count = 0
        for indices in itertools.combinations(range(len(grids)), size):
            tested += 1
            used: set[int] = set()
            supports_overlap = False
            for index in indices:
                if used & internal[index]:
                    supports_overlap = True
                    break
                used.update(internal[index])
            if supports_overlap:
                continue
            disjoint += 1
            coordinate_matrix = np.vstack([coordinates[index] for index in indices])
            logical_rank = _gf2_rank(coordinate_matrix, code.field)
            pairing = np.block(
                [
                    [cross_pairings[(left, right)] for right in indices]
                    for left in indices
                ]
            ).astype(np.uint8)
            pairing_rank = _gf2_rank(pairing, code.field)
            if logical_rank > best_logical_rank:
                best_logical_rank = logical_rank
                best_logical_combination = indices
            if pairing_rank > best_pairing_rank_any:
                best_pairing_rank_any = pairing_rank
                best_pairing_combination_any = indices
            if logical_rank != BOTTOM_ORDER * size:
                continue
            independent += 1
            if pairing_rank > best_pairing_rank_independent:
                best_pairing_rank_independent = pairing_rank
                best_independent_combination = indices
            if pairing_rank != BOTTOM_ORDER * size:
                continue
            full_rank_count += 1
            pair_map = _permutation_pair_map(pairing)
            hits.append(
                {
                    "num_grids": size,
                    "indices": list(indices),
                    "seed_supports": [
                        grids[index]["seed_support"] for index in indices
                    ],
                    "internal_orbits": [
                        grids[index]["internal_orbits"] for index in indices
                    ],
                    "combined_rank_mod_stabilizers": logical_rank,
                    "combined_zx_pairing_rank": pairing_rank,
                    "pairing_is_permutation": pair_map is not None,
                    "logical_grid_pair_maps": pair_map,
                }
            )
        summaries[str(size)] = {
            "combinations_tested": tested,
            "disjoint_combinations": disjoint,
            "independent_combinations": independent,
            "best_logical_rank": best_logical_rank,
            "best_logical_combination": (
                list(best_logical_combination)
                if best_logical_combination is not None
                else None
            ),
            "best_pairing_rank_any": best_pairing_rank_any,
            "best_pairing_combination_any": (
                list(best_pairing_combination_any)
                if best_pairing_combination_any is not None
                else None
            ),
            "best_pairing_rank_independent": best_pairing_rank_independent,
            "best_independent_combination": (
                list(best_independent_combination)
                if best_independent_combination is not None
                else None
            ),
            "full_rank_combinations": full_rank_count,
        }
    return {"by_num_grids": summaries, "full_rank_hits": hits}


def saturation_even_weight_obstruction(
    *, num_qubits: int, num_grids: int, logical_weight: int
) -> bool:
    """Whether disjoint even supports would saturate and force singular pairing."""
    return bool(
        logical_weight % 2 == 0
        and num_grids * BOTTOM_ORDER * logical_weight == num_qubits
    )
