"""Orbit-replacement refinement for folded ``[[n,2,*]]`` CSS codes.

The direct Z3 search represents a C2-invariant X-check space by generators whose
translation orbits have rank one or two.  This module keeps that presentation
explicit and replaces one orbit at a time.  Consequently every candidate keeps
the requested C2 action and every displayed generator stays below the requested
check-weight ceiling.
"""

from __future__ import annotations

import functools
import json
import pathlib
import random
from collections import Counter
from typing import Any, Sequence

from searches.inverse_c2_minimum.search import (
    FoldGeometry,
    LogicalGeometry,
    analyze_basis,
    permute_mask,
)


def canonical_basis(rows: Sequence[int], n: int) -> tuple[int, ...]:
    """Return a unique reduced basis for a packed binary row space."""
    pivots = [0] * n
    for original in rows:
        value = int(original)
        while value:
            pivot = value.bit_length() - 1
            if pivots[pivot]:
                value ^= pivots[pivot]
                continue
            for index, known in enumerate(pivots):
                if known >> pivot & 1:
                    pivots[index] ^= value
            pivots[pivot] = value
            break
    return tuple(pivots[index] for index in range(n - 1, -1, -1) if pivots[index])


def expand_generators(
    generators: Sequence[int],
    module_type: Sequence[int],
    translation: Sequence[int],
) -> tuple[int, ...]:
    if len(generators) != len(module_type):
        raise ValueError("one generator is required per C2 module")
    rows: list[int] = []
    for generator, orbit_rank in zip(generators, module_type):
        translated = permute_mask(int(generator), translation)
        if orbit_rank == 1:
            if translated != generator:
                raise ValueError("rank-one module generator is not C2 fixed")
            rows.append(int(generator))
        elif orbit_rank == 2:
            if translated == generator:
                raise ValueError("rank-two module generator is C2 fixed")
            rows.extend((int(generator), translated))
        else:
            raise ValueError("C2 module ranks must be one or two")
    return tuple(rows)


def _dot(left: int, right: int) -> int:
    return (left & right).bit_count() & 1


def _css_orthogonal(rows_x: Sequence[int], fold: Sequence[int]) -> bool:
    rows_z = tuple(permute_mask(row, fold) for row in rows_x)
    return all(not _dot(left, right) for left in rows_x for right in rows_z)


def _connected(rows_x: Sequence[int], fold: Sequence[int], n: int) -> bool:
    rows = tuple(rows_x) + tuple(permute_mask(row, fold) for row in rows_x)
    adjacency = [set() for _ in range(len(rows) + n)]
    for row_index, row in enumerate(rows):
        for qubit in range(n):
            if row >> qubit & 1:
                qnode = len(rows) + qubit
                adjacency[row_index].add(qnode)
                adjacency[qnode].add(row_index)
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == len(adjacency)


@functools.lru_cache(maxsize=None)
def candidate_generators(
    n: int,
    translation: tuple[int, ...],
    logical_z: tuple[int, int],
    orbit_rank: int,
    maximum_check_weight: int,
) -> tuple[int, ...]:
    """Enumerate normalized low-weight generators satisfying fixed constraints."""
    candidates = []
    for mask in range(1, 1 << n):
        weight = mask.bit_count()
        if not 2 <= weight <= maximum_check_weight:
            continue
        if any(_dot(mask, logical) for logical in logical_z):
            continue
        translated = permute_mask(mask, translation)
        if orbit_rank == 1:
            if translated != mask:
                continue
        elif orbit_rank == 2:
            if translated == mask or mask > translated:
                continue
        else:
            raise ValueError("orbit rank must be one or two")
        candidates.append(mask)
    return tuple(candidates)


def seed_from_record(record: dict[str, Any]) -> dict[str, Any]:
    analysis = record["analysis"]
    fold_data = analysis["fold"]
    geometry_data = fold_data["geometry"]
    geometry = LogicalGeometry(
        n=int(geometry_data["n"]),
        logical_weight=int(geometry_data["logical_weight"]),
        slack_transpositions=int(geometry_data["slack_transpositions"]),
        slack_fixed_points=int(geometry_data["slack_fixed_points"]),
    )
    fold = FoldGeometry(
        geometry=geometry,
        index=int(record["fold_index"]),
        permutation=tuple(int(value) for value in fold_data["permutation"]),
        raw_multiplicity=int(fold_data["raw_multiplicity"]),
    )
    module_type = tuple(int(value) for value in record["module_type"])
    rows = tuple(int(value) for value in analysis["basis_x_masks"])
    generators = []
    cursor = 0
    for orbit_rank in module_type:
        generator = rows[cursor]
        generators.append(generator)
        if orbit_rank == 2:
            expected = permute_mask(generator, geometry.translation)
            if rows[cursor + 1] != expected:
                raise ValueError("saved rank-two rows are not a translation orbit")
        cursor += orbit_rank
    if cursor != len(rows):
        raise ValueError("saved basis does not match its module type")
    reconstructed = expand_generators(generators, module_type, geometry.translation)
    if canonical_basis(reconstructed, geometry.n) != canonical_basis(rows, geometry.n):
        raise ValueError("failed to reconstruct saved X-check row space")
    return {
        "geometry": geometry,
        "fold": fold,
        "module_type": module_type,
        "generators": tuple(generators),
    }


def load_seed(path: pathlib.Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    seed = seed_from_record(record)
    seed["source"] = str(path)
    return seed


def recover_low_weight_generators(
    analysis: dict[str, Any],
    module_type: Sequence[int],
    maximum_check_weight: int,
) -> tuple[int, ...]:
    """Recover a low-weight C2-orbit presentation of a saved row space.

    Local-search summaries save an arbitrary basis of the stabilizer space.  That
    basis can be heavier than the actual orbit generators.  Enumerating the 127
    nonzero elements of a rank-seven space is cheap and lets us reconstruct a
    presentation that obeys the requested check-weight ceiling exactly.
    """
    fold_data = analysis["fold"]
    geometry_data = fold_data["geometry"]
    n = int(geometry_data["n"])
    translation = tuple(int(value) for value in geometry_data["translation"])
    target_rows = tuple(int(value) for value in analysis["basis_x_masks"])
    target = canonical_basis(target_rows, n)
    if len(target) != sum(module_type):
        raise ValueError("saved row-space rank does not match module type")

    codewords = {
        functools.reduce(int.__xor__, (row for index, row in enumerate(target_rows) if combination >> index & 1), 0)
        for combination in range(1, 1 << len(target_rows))
    }
    pools: dict[int, tuple[int, ...]] = {}
    fixed = sorted(
        word
        for word in codewords
        if word.bit_count() <= maximum_check_weight
        and permute_mask(word, translation) == word
    )
    paired = sorted(
        word
        for word in codewords
        if word.bit_count() <= maximum_check_weight
        and word < permute_mask(word, translation)
    )
    pools[1] = tuple(fixed)
    pools[2] = tuple(paired)

    failed: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()

    def search(
        index: int,
        generators: tuple[int, ...],
        rows: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        signature = canonical_basis(rows, n)
        if index == len(module_type):
            return generators if signature == target else None
        key = (index, signature, generators[-1:] if generators else ())
        if key in failed:
            return None
        rank = int(module_type[index])
        previous_same_rank = max(
            (generators[prior] for prior in range(index) if module_type[prior] == rank),
            default=-1,
        )
        for candidate in pools[rank]:
            if candidate <= previous_same_rank:
                continue
            expanded = expand_generators((candidate,), (rank,), translation)
            new_rows = rows + expanded
            if len(canonical_basis(new_rows, n)) != len(signature) + rank:
                continue
            result = search(index + 1, generators + (candidate,), new_rows)
            if result is not None:
                return result
        failed.add(key)
        return None

    recovered = search(0, (), ())
    if recovered is None:
        raise ValueError(
            f"no C2-orbit presentation exists at weight <= {maximum_check_weight}"
        )
    return recovered


def analyze_generators(
    generators: Sequence[int], module_type: Sequence[int], fold: FoldGeometry
) -> dict[str, Any]:
    rows = expand_generators(generators, module_type, fold.geometry.translation)
    analysis = analyze_basis(rows, fold)
    analysis["presentation_generators_x_masks"] = list(map(int, generators))
    analysis["presentation_basis_x_masks"] = list(map(int, rows))
    analysis["presentation_row_weights_x"] = [row.bit_count() for row in rows]
    analysis["presentation_maximum_check_weight"] = max(
        analysis["presentation_row_weights_x"], default=0
    )
    analysis["module_type"] = list(map(int, module_type))
    return analysis


def refinement_score(analysis: dict[str, Any]) -> tuple[int, int, int]:
    distance = int(analysis["distance"])
    x_counts = analysis["x_distance"]["weight_counts_through_seven"]
    z_counts = analysis["z_distance"]["weight_counts_through_seven"]

    def count(weight: int) -> int:
        return int(x_counts.get(str(weight), 0)) + int(z_counts.get(str(weight), 0))

    return (
        distance,
        -count(distance),
        -sum(count(weight) for weight in range(1, 6)),
    )


def refine_seed(
    seed: dict[str, Any],
    *,
    rng: random.Random,
    iterations: int,
    attempts_per_replacement: int,
    maximum_check_weight: int,
    replacement_width: int = 1,
    downhill_distance_floor: int = 4,
    downhill_probability: float = 0.0,
    minimum_start_distance: int = 2,
) -> dict[str, Any]:
    geometry: LogicalGeometry = seed["geometry"]
    fold: FoldGeometry = seed["fold"]
    module_type: tuple[int, ...] = seed["module_type"]
    current = tuple(int(value) for value in seed["generators"])
    current_analysis = analyze_generators(current, module_type, fold)
    if current_analysis["distance"] < minimum_start_distance:
        raise ValueError(
            f"local refinement requires a distance-{minimum_start_distance} seed"
        )
    current_score = refinement_score(current_analysis)
    best = current_analysis
    seen = {canonical_basis(current_analysis["presentation_basis_x_masks"], geometry.n)}
    counters: Counter[str] = Counter()
    pools = {
        rank: candidate_generators(
            geometry.n,
            geometry.translation,
            geometry.logical_z,
            rank,
            maximum_check_weight,
        )
        for rank in set(module_type)
    }
    if not 1 <= replacement_width <= len(module_type):
        raise ValueError("replacement width exceeds the number of C2 modules")
    if not 0.0 <= downhill_probability <= 1.0:
        raise ValueError("downhill probability must lie between zero and one")
    for _iteration in range(iterations):
        counters["iterations"] += 1
        replaced_indices = rng.sample(range(len(module_type)), replacement_width)
        neighbor = None
        neighbor_rows: tuple[int, ...] | None = None
        for _attempt in range(attempts_per_replacement):
            counters["replacement_attempts"] += 1
            proposal = list(current)
            for replaced in replaced_indices:
                pool = pools[module_type[replaced]]
                proposal[replaced] = pool[rng.randrange(len(pool))]
            try:
                rows = expand_generators(proposal, module_type, geometry.translation)
            except ValueError:
                continue
            signature = canonical_basis(rows, geometry.n)
            target_rank = (geometry.n - 2) // 2
            if len(signature) != target_rank or signature in seen:
                continue
            if not _css_orthogonal(rows, fold.permutation):
                continue
            seen.add(signature)
            if not _connected(rows, fold.permutation, geometry.n):
                counters["disconnected"] += 1
                continue
            neighbor = tuple(proposal)
            neighbor_rows = rows
            break
        if neighbor is None or neighbor_rows is None:
            counters["replacement_failed"] += 1
            continue
        analysis = analyze_generators(neighbor, module_type, fold)
        distance = int(analysis["distance"])
        counters[f"neighbor_distance_{distance}"] += 1
        score = refinement_score(analysis)
        if score > refinement_score(best):
            best = analysis
            counters["best_improvement"] += 1
        if distance >= 6:
            counters["accepted"] += 1
            return {
                "accepted": True,
                "best": best,
                "counters": dict(sorted(counters.items())),
                "candidate_pool_sizes": {
                    str(rank): len(pool) for rank, pool in sorted(pools.items())
                },
            }
        accept_move = score > current_score or (
            score == current_score and rng.random() < 0.10
        )
        if (
            not accept_move
            and distance >= downhill_distance_floor
            and rng.random() < downhill_probability
        ):
            accept_move = True
            counters["downhill_move_accepted"] += 1
        if accept_move:
            current = neighbor
            current_analysis = analysis
            current_score = score
            counters["move_accepted"] += 1
    return {
        "accepted": False,
        "best": best,
        "counters": dict(sorted(counters.items())),
        "candidate_pool_sizes": {
            str(rank): len(pool) for rank, pool in sorted(pools.items())
        },
    }
