"""Candidate enumeration, light-presentation extraction, and exact distance."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from algebra import (
    CHECK_RANK,
    GROUP_ORDER,
    QUOTIENT_ORDER,
    SparsePolynomial,
    build_stabilizer_basis,
    displacement_subgroup_size,
    fibre_logicals,
    gf8_contains,
    gf8_rref,
    gf8_subspace_key,
    iter_monomial_ideal_thresholds,
    iter_sparse_polynomials,
    monomial_ideal_basis,
    physical_seed_from_sparse,
    quotient_annihilator_basis,
    quotient_ideal_basis,
    validate_stabilizer,
)


@dataclass(frozen=True)
class CatalogEntry:
    polynomial: SparsePolynomial
    ideal_basis: np.ndarray[Any, Any]
    ideal_key: tuple[tuple[int, ...], ...]


@dataclass
class IdealCandidate:
    candidate_id: str
    ideal_basis: np.ndarray[Any, Any]
    origins: list[dict[str, Any]]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({str(origin["family"]) for origin in self.origins}))


def build_sparse_catalog() -> list[CatalogEntry]:
    catalog = []
    for polynomial in iter_sparse_polynomials():
        ideal = quotient_ideal_basis([polynomial.vector])
        catalog.append(CatalogEntry(polynomial, ideal, gf8_subspace_key(ideal)))
    catalog.sort(
        key=lambda entry: (
            entry.polynomial.support_size,
            -len(entry.ideal_basis),
            entry.polynomial.candidate_id,
        )
    )
    return catalog


def enumerate_ideal_candidates(catalog: Sequence[CatalogEntry]) -> tuple[list[IdealCandidate], dict[str, int]]:
    records: dict[tuple[tuple[int, ...], ...], IdealCandidate] = {}
    raw_counts: defaultdict[str, int] = defaultdict(int)

    def include(ideal: np.ndarray[Any, Any], origin: dict[str, Any]) -> None:
        key = gf8_subspace_key(ideal)
        if key not in records:
            digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            records[key] = IdealCandidate(f"ideal-{digest}", ideal, [])
        records[key].origins.append(origin)

    for thresholds in iter_monomial_ideal_thresholds():
        raw_counts["monomial_ideal"] += 1
        ideal = monomial_ideal_basis(thresholds)
        include(
            ideal,
            {
                "family": "monomial_ideal",
                "thresholds": list(map(int, thresholds)),
                "dimension_over_gf8": len(ideal),
            },
        )
    for entry in catalog:
        family = entry.polynomial.family
        raw_counts[family] += 1
        include(
            entry.ideal_basis,
            {
                "family": family,
                "polynomial_id": entry.polynomial.candidate_id,
                "support": [list(term) for term in entry.polynomial.support],
                "dimension_over_gf8": len(entry.ideal_basis),
            },
        )
    return sorted(records.values(), key=lambda item: item.candidate_id), dict(raw_counts)


def _span_entries(entries: Sequence[CatalogEntry]) -> np.ndarray[Any, Any]:
    if not entries:
        return np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    return gf8_rref(np.vstack([entry.ideal_basis for entry in entries]))[0]


def _greedy_generators(target: np.ndarray[Any, Any], eligible: Sequence[CatalogEntry]) -> list[CatalogEntry]:
    selected: list[CatalogEntry] = []
    selected_ids: set[str] = set()
    generated = np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    while len(generated) < len(target):
        best = None
        for entry in eligible:
            if entry.polynomial.candidate_id in selected_ids:
                continue
            expanded = gf8_rref(np.vstack([generated, entry.ideal_basis]))[0]
            gain = len(expanded) - len(generated)
            if gain <= 0:
                continue
            score = (gain, -entry.polynomial.support_size, len(entry.ideal_basis))
            if best is None or score > best[0]:
                best = (score, entry, expanded)
        if best is None:
            break
        selected.append(best[1])
        selected_ids.add(best[1].polynomial.candidate_id)
        generated = best[2]
    return selected


def find_connected_sparse_presentation(
    ideal: np.ndarray[Any, Any],
    annihilator: np.ndarray[Any, Any],
    catalog: Sequence[CatalogEntry],
) -> dict[str, Any]:
    """Find a complete connected presentation from every seed of weight <= 12.

    The algebraic span test uses all eligible sparse seeds, avoiding the RREF
    blind spot encountered in the lifted ``C56 x C4`` audit.
    """
    eligible_plus = [entry for entry in catalog if gf8_contains(ideal, entry.polynomial.vector)]
    eligible_minus = [entry for entry in catalog if gf8_contains(annihilator, entry.polynomial.vector)]
    all_plus_rank = len(_span_entries(eligible_plus))
    all_minus_rank = len(_span_entries(eligible_minus))
    complete = all_plus_rank == len(ideal) and all_minus_rank == len(annihilator)
    all_items = [(entry, False) for entry in eligible_plus] + [(entry, True) for entry in eligible_minus]
    all_seeds = [physical_seed_from_sparse(entry.polynomial, dagger=dagger) for entry, dagger in all_items]
    all_subgroup = displacement_subgroup_size(all_seeds)

    selected_plus = _greedy_generators(ideal, eligible_plus)
    selected_minus = _greedy_generators(annihilator, eligible_minus)
    selected = [(entry, False) for entry in selected_plus] + [(entry, True) for entry in selected_minus]

    def subgroup(items: Sequence[tuple[CatalogEntry, bool]]) -> int:
        return displacement_subgroup_size(
            [physical_seed_from_sparse(entry.polynomial, dagger=dagger) for entry, dagger in items]
        )

    selected_keys = {(entry.polynomial.candidate_id, dagger) for entry, dagger in selected}
    remaining = [
        item
        for item in all_items
        if (item[0].polynomial.candidate_id, item[1]) not in selected_keys
    ]
    while complete and subgroup(selected) < GROUP_ORDER:
        current = subgroup(selected)
        choices = [(subgroup([*selected, item]), item) for item in remaining]
        choices = [choice for choice in choices if choice[0] > current]
        if not choices:
            break
        choices.sort(
            key=lambda choice: (
                -choice[0],
                choice[1][0].polynomial.support_size,
                choice[1][0].polynomial.candidate_id,
                choice[1][1],
            )
        )
        item = choices[0][1]
        selected.append(item)
        item_key = (item[0].polynomial.candidate_id, item[1])
        remaining = [
            other
            for other in remaining
            if (other[0].polynomial.candidate_id, other[1]) != item_key
        ]

    # Remove unnecessary checks while retaining both sector spans and connectivity.
    changed = True
    while changed:
        changed = False
        for item in list(reversed(selected)):
            item_key = (item[0].polynomial.candidate_id, item[1])
            trial = [
                other
                for other in selected
                if (other[0].polynomial.candidate_id, other[1]) != item_key
            ]
            trial_plus = [entry for entry, dagger in trial if not dagger]
            trial_minus = [entry for entry, dagger in trial if dagger]
            if (
                len(_span_entries(trial_plus)) == len(ideal)
                and len(_span_entries(trial_minus)) == len(annihilator)
                and subgroup(trial) == GROUP_ORDER
            ):
                selected = trial
                changed = True

    selected_subgroup = subgroup(selected)

    def item_record(entry: CatalogEntry, dagger: bool) -> dict[str, Any]:
        return {
            "polynomial_id": entry.polynomial.candidate_id,
            "support": [list(term) for term in entry.polynomial.support],
            "quotient_weight": entry.polynomial.support_size,
            "physical_check_weight": 4 * entry.polynomial.support_size,
            "dagger_sector": dagger,
            "orbit_rank_over_gf8": len(entry.ideal_basis),
        }

    return {
        "complete_weight_12_span": complete,
        "all_sparse_displacement_subgroup_size": all_subgroup,
        "connected_weight_12_presentation_exists": complete and all_subgroup == GROUP_ORDER,
        "selected_displacement_subgroup_size": selected_subgroup,
        "eligible_plus_seeds": len(eligible_plus),
        "eligible_minus_seeds": len(eligible_minus),
        "all_plus_span_dimension": all_plus_rank,
        "all_minus_span_dimension": all_minus_rank,
        "selected_generators": [item_record(entry, dagger) for entry, dagger in selected],
        "selected_check_weight": max(
            (4 * entry.polynomial.support_size for entry, _dagger in selected), default=0
        ),
    }


def _syndrome_columns(stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]) -> tuple[list[int], int]:
    target = 1 << len(stabilizer)
    columns = []
    for qubit in range(stabilizer.shape[1]):
        value = sum(int(stabilizer[row, qubit]) << row for row in range(len(stabilizer)))
        if logical[qubit]:
            value |= target
        columns.append(value)
    return columns, target


def find_logical_through_weight_five(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[int | None, list[int] | None, float]:
    started = time.perf_counter()
    columns, target = _syndrome_columns(stabilizer, logical)
    num_qubits = len(columns)
    for qubit, syndrome in enumerate(columns):
        if syndrome == target:
            return 1, [qubit], time.perf_counter() - started
    pair_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pairs = []
    for left in range(num_qubits):
        for right in range(left + 1, num_qubits):
            syndrome = columns[left] ^ columns[right]
            pair_map[syndrome].append((left, right))
            pairs.append((left, right, syndrome))
    if target in pair_map:
        return 2, list(pair_map[target][0]), time.perf_counter() - started
    for qubit, syndrome in enumerate(columns):
        for left, right in pair_map.get(target ^ syndrome, ()):
            if qubit not in (left, right):
                return 3, [left, right, qubit], time.perf_counter() - started
    for left, right, syndrome in pairs:
        for other_left, other_right in pair_map.get(target ^ syndrome, ()):
            if len({left, right, other_left, other_right}) == 4:
                return 4, [left, right, other_left, other_right], time.perf_counter() - started
    for first in range(num_qubits - 2):
        for second in range(first + 1, num_qubits - 1):
            partial = columns[first] ^ columns[second]
            for third in range(second + 1, num_qubits):
                for left, right in pair_map.get(target ^ partial ^ columns[third], ()):
                    if len({first, second, third, left, right}) == 5:
                        return 5, [first, second, third, left, right], time.perf_counter() - started
    return None, None, time.perf_counter() - started


def find_weight_six_logical(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[list[int] | None, float]:
    started = time.perf_counter()
    columns, target = _syndrome_columns(stabilizer, logical)
    triples: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for first in range(len(columns) - 2):
        for second in range(first + 1, len(columns) - 1):
            partial = columns[first] ^ columns[second]
            for third in range(second + 1, len(columns)):
                triples[partial ^ columns[third]].append((first, second, third))
    for syndrome, supports in triples.items():
        complements = triples.get(target ^ syndrome, ())
        for first_support in supports:
            first_set = set(first_support)
            for second_support in complements:
                if first_set.isdisjoint(second_support):
                    return sorted([*first_support, *second_support]), time.perf_counter() - started
    return None, time.perf_counter() - started


def analyze_candidate(candidate: IdealCandidate, catalog: Sequence[CatalogEntry]) -> dict[str, Any]:
    started = time.perf_counter()
    ideal = gf8_rref(candidate.ideal_basis)[0]
    annihilator = quotient_annihilator_basis(list(ideal))
    stabilizer = build_stabilizer_basis(ideal, annihilator)
    fixed = validate_stabilizer(stabilizer)
    presentation = find_connected_sparse_presentation(ideal, annihilator, catalog)

    low_weight, support, distance_seconds = find_logical_through_weight_five(
        stabilizer, fibre_logicals()[0]
    )
    if low_weight is None:
        six_support, six_seconds = find_weight_six_logical(stabilizer, fibre_logicals()[0])
        distance_seconds += six_seconds
        if six_support is None:
            distance, support = 7, np.flatnonzero(fibre_logicals()[0]).astype(int).tolist()
            distance_method = "exact_exclusion_through_six_with_weight_seven_fibre"
        else:
            distance, support = 6, six_support
            distance_method = "exact_meet_in_the_middle_3_plus_3"
    else:
        distance = low_weight
        distance_method = "exact_subset_search_through_five"

    structural = {
        **fixed,
        "target_rank": fixed["rank"] == CHECK_RANK,
        "weight_12_generated": presentation["complete_weight_12_span"],
        "weight_12_connected": presentation["connected_weight_12_presentation_exists"],
    }
    return {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "families": list(candidate.families),
        "origins": candidate.origins,
        "n": GROUP_ORDER,
        "k": GROUP_ORDER - 2 * int(fixed["rank"]),
        "ideal_dimension_over_gf8": len(ideal),
        "annihilator_dimension_over_gf8": len(annihilator),
        "structural_checks": structural,
        "presentation": presentation,
        "distance": distance,
        "distance_support": support,
        "distance_method": distance_method,
        "distance_seconds": round(distance_seconds, 6),
        "target_distance_six": distance >= 6,
        "fully_usable": all(bool(value) for value in structural.values()) and distance >= 6,
        "total_seconds": round(time.perf_counter() - started, 6),
    }
