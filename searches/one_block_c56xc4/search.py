"""Candidate construction, sparse-check screening, and distance certification."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from algebra import (
    GROUP_ORDER,
    QUOTIENT_ORDER,
    SparsePolynomial,
    build_stabilizer_basis,
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
    tanner_connected,
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
    """Precompute principal ideals for all normalized sparse polynomials."""
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


def enumerate_ideal_candidates(
    catalog: Sequence[CatalogEntry],
) -> tuple[list[IdealCandidate], dict[str, int]]:
    """Combine 495 monomial ideals and all sparse principal ideals by row space."""
    records: dict[tuple[tuple[int, ...], ...], IdealCandidate] = {}
    raw_counts: defaultdict[str, int] = defaultdict(int)

    for thresholds in iter_monomial_ideal_thresholds():
        raw_counts["monomial_ideal"] += 1
        ideal = monomial_ideal_basis(thresholds)
        key = gf8_subspace_key(ideal)
        origin = {
            "family": "monomial_ideal",
            "thresholds": list(map(int, thresholds)),
            "dimension_over_gf8": len(ideal),
        }
        if key not in records:
            digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            records[key] = IdealCandidate(f"ideal-{digest}", ideal, [])
        records[key].origins.append(origin)

    for entry in catalog:
        family = entry.polynomial.family
        raw_counts[family] += 1
        origin = {
            "family": family,
            "polynomial_id": entry.polynomial.candidate_id,
            "support": [list(term) for term in entry.polynomial.support],
            "dimension_over_gf8": len(entry.ideal_basis),
        }
        if entry.ideal_key not in records:
            digest = hashlib.sha256(repr(entry.ideal_key).encode()).hexdigest()[:16]
            records[entry.ideal_key] = IdealCandidate(
                f"ideal-{digest}", entry.ideal_basis, []
            )
        records[entry.ideal_key].origins.append(origin)

    candidates = sorted(records.values(), key=lambda candidate: candidate.candidate_id)
    return candidates, dict(raw_counts)


def _span_with(
    basis: np.ndarray[Any, Any], addition: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if not len(basis):
        return gf8_rref(addition)[0]
    return gf8_rref(np.vstack([basis, addition]))[0]


def find_sparse_orbit_generators(
    target_basis: np.ndarray[Any, Any], catalog: Sequence[CatalogEntry]
) -> tuple[list[CatalogEntry], int]:
    """Find weight-at-most-three elements whose principal ideals span a target."""
    target = gf8_rref(target_basis)[0]
    if not len(target):
        return [], 0
    generated = np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    selected = []
    for entry in catalog:
        if not gf8_contains(target, entry.polynomial.vector):
            continue
        expanded = _span_with(generated, entry.ideal_basis)
        if len(expanded) == len(generated):
            continue
        generated = expanded
        selected.append(entry)
        if len(generated) == len(target):
            break
    return selected, len(generated)


def _syndrome_columns(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[list[int], int]:
    rank = len(stabilizer)
    target = 1 << rank
    columns = []
    for qubit in range(stabilizer.shape[1]):
        value = sum(
            int(stabilizer[row, qubit]) << row for row in range(rank)
        )
        if logical[qubit]:
            value |= target
        columns.append(value)
    return columns, target


def find_logical_through_weight_five(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[int | None, list[int] | None, float]:
    """Exactly search one translated logical class through weight five."""
    started = time.perf_counter()
    columns, target = _syndrome_columns(stabilizer, logical)
    num_qubits = len(columns)

    for qubit, syndrome in enumerate(columns):
        if syndrome == target:
            return 1, [qubit], time.perf_counter() - started

    pair_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pairs: list[tuple[int, int, int]] = []
    for left in range(num_qubits):
        for right in range(left + 1, num_qubits):
            syndrome = columns[left] ^ columns[right]
            pair = (left, right)
            pair_map[syndrome].append(pair)
            pairs.append((left, right, syndrome))
    if target in pair_map:
        return 2, list(pair_map[target][0]), time.perf_counter() - started

    for qubit, syndrome in enumerate(columns):
        for left, right in pair_map.get(target ^ syndrome, ()):
            if qubit != left and qubit != right:
                return 3, [left, right, qubit], time.perf_counter() - started

    for left, right, syndrome in pairs:
        for other_left, other_right in pair_map.get(target ^ syndrome, ()):
            if len({left, right, other_left, other_right}) == 4:
                return (
                    4,
                    [left, right, other_left, other_right],
                    time.perf_counter() - started,
                )

    for first in range(num_qubits - 2):
        first_syndrome = columns[first]
        for second in range(first + 1, num_qubits - 1):
            partial = first_syndrome ^ columns[second]
            for third in range(second + 1, num_qubits):
                wanted = target ^ partial ^ columns[third]
                for left, right in pair_map.get(wanted, ()):
                    if len({first, second, third, left, right}) == 5:
                        return (
                            5,
                            [first, second, third, left, right],
                            time.perf_counter() - started,
                        )
    return None, None, time.perf_counter() - started


def find_weight_six_logical(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[list[int] | None, float]:
    """Exactly test for a weight-six vector using a vectorized ``3+3`` split."""
    started = time.perf_counter()
    columns, target = _syndrome_columns(stabilizer, logical)
    num_qubits = len(columns)
    num_triples = num_qubits * (num_qubits - 1) * (num_qubits - 2) // 6
    low_columns = np.asarray(
        [value & ((1 << 64) - 1) for value in columns], dtype=np.uint64
    )
    high_columns = np.asarray([value >> 64 for value in columns], dtype=np.uint64)
    lows = np.empty(num_triples, dtype=np.uint64)
    highs = np.empty(num_triples, dtype=np.uint64)
    packed = np.empty(num_triples, dtype=np.uint32)
    offset = 0
    for first in range(num_qubits - 2):
        for second in range(first + 1, num_qubits - 1):
            thirds = np.arange(second + 1, num_qubits, dtype=np.uint32)
            count = len(thirds)
            selection = slice(offset, offset + count)
            lows[selection] = (
                low_columns[first] ^ low_columns[second] ^ low_columns[thirds]
            )
            highs[selection] = (
                high_columns[first] ^ high_columns[second] ^ high_columns[thirds]
            )
            packed[selection] = (
                np.uint32(first)
                | (np.uint32(second) << np.uint32(8))
                | (thirds << np.uint32(16))
            )
            offset += count
    assert offset == num_triples

    key_dtype = np.dtype([("high", np.uint64), ("low", np.uint64)])
    keys = np.empty(num_triples, dtype=key_dtype)
    keys["high"] = highs
    keys["low"] = lows
    order = np.argsort(keys, order=("high", "low"), kind="stable")
    sorted_keys = keys[order]
    complements = keys.copy()
    complements["high"] ^= np.uint64(target >> 64)
    complements["low"] ^= np.uint64(target & ((1 << 64) - 1))
    left_edges = np.searchsorted(sorted_keys, complements, side="left")
    right_edges = np.searchsorted(sorted_keys, complements, side="right")

    for triple_index in np.flatnonzero(left_edges < right_edges):
        first_packed = int(packed[triple_index])
        first_support = {
            first_packed & 0xFF,
            (first_packed >> 8) & 0xFF,
            (first_packed >> 16) & 0xFF,
        }
        for sorted_index in range(
            int(left_edges[triple_index]), int(right_edges[triple_index])
        ):
            second_packed = int(packed[int(order[sorted_index])])
            second_support = {
                second_packed & 0xFF,
                (second_packed >> 8) & 0xFF,
                (second_packed >> 16) & 0xFF,
            }
            if first_support.isdisjoint(second_support):
                return (
                    sorted(first_support | second_support),
                    time.perf_counter() - started,
                )
    return None, time.perf_counter() - started


def certify_distance_ilp(
    stabilizer: np.ndarray[Any, Any], logical: np.ndarray[Any, Any]
) -> tuple[int, list[int], str, float]:
    """Certify the remaining distance-six-or-seven cases with HiGHS."""
    import cvxpy as cp

    effective_check = np.vstack([stabilizer, logical])
    syndrome = np.zeros(len(effective_check), dtype=int)
    syndrome[-1] = 1
    check = np.asarray(effective_check, dtype=int)
    error = cp.Variable(check.shape[1], boolean=True)
    slack = cp.Variable(check.shape[0], integer=True)
    max_slack = (np.sum(check, axis=1) - syndrome) // 2
    constraints = [
        check @ error == syndrome + 2 * slack,
        slack >= 0,
        slack <= max_slack,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(error)), constraints)
    started = time.perf_counter()
    result = problem.solve(solver="HIGHS")
    elapsed = time.perf_counter() - started
    if problem.status != cp.OPTIMAL or error.value is None or not np.isfinite(result):
        raise RuntimeError(f"exact-distance ILP did not terminate optimally: {problem.status}")
    solution = np.rint(error.value).astype(int)
    if not np.array_equal(check @ solution % 2, syndrome):
        raise RuntimeError("ILP returned a solution with the wrong syndrome")
    return (
        int(np.count_nonzero(solution)),
        np.flatnonzero(solution).astype(int).tolist(),
        str(problem.status),
        elapsed,
    )


def analyze_candidate(
    candidate: IdealCandidate,
    catalog: Sequence[CatalogEntry],
    *,
    exact_distance: bool,
    diagnose_rejections_through_five: bool = True,
) -> dict[str, Any]:
    """Construct, structurally screen, and optionally certify one ideal code."""
    started = time.perf_counter()
    ideal = gf8_rref(candidate.ideal_basis)[0]
    annihilator = quotient_annihilator_basis(list(ideal))
    stabilizer = build_stabilizer_basis(ideal, annihilator)
    fixed_checks = validate_stabilizer(stabilizer)

    ideal_generators, ideal_generated_dimension = find_sparse_orbit_generators(
        ideal, catalog
    )
    annihilator_generators, annihilator_generated_dimension = (
        find_sparse_orbit_generators(annihilator, catalog)
    )
    sparse_generation = (
        ideal_generated_dimension == len(ideal)
        and annihilator_generated_dimension == len(annihilator)
    )
    selected = [
        (entry, False) for entry in ideal_generators
    ] + [
        (entry, True) for entry in annihilator_generators
    ]
    check_weight = (
        max(
            (4 * entry.polynomial.support_size for entry, _dagger in selected),
            default=0,
        )
        if sparse_generation
        else None
    )
    physical_seeds = [
        physical_seed_from_sparse(entry.polynomial, dagger=dagger)
        for entry, dagger in selected
    ]
    connected = sparse_generation and tanner_connected(physical_seeds)

    structural_checks = {
        **fixed_checks,
        "target_rank": fixed_checks["rank"] == 96,
        "sparse_weight_12_generation": sparse_generation and (check_weight or 0) <= 12,
        "tanner_connected": connected,
    }
    structurally_accepted = all(bool(value) for value in structural_checks.values())

    distance = None
    distance_support = None
    distance_method = None
    distance_seconds = 0.0
    should_diagnose = structurally_accepted or diagnose_rejections_through_five
    if should_diagnose:
        low_weight, low_support, low_seconds = find_logical_through_weight_five(
            stabilizer, fibre_logicals()[0]
        )
        distance_seconds += low_seconds
        if low_weight is not None:
            distance = low_weight
            distance_support = low_support
            distance_method = "exact_subset_search_through_five"
        elif exact_distance:
            weight_six_support, elapsed = find_weight_six_logical(
                stabilizer, fibre_logicals()[0]
            )
            distance_seconds += elapsed
            if weight_six_support is not None:
                distance = 6
                distance_support = weight_six_support
                distance_method = "exact_meet_in_the_middle_3_plus_3"
            else:
                distance = 7
                distance_support = np.flatnonzero(fibre_logicals()[0]).tolist()
                distance_method = "exact_exclusion_through_six_with_weight_seven_fibre"

    def generators_to_dict(
        generators: Sequence[CatalogEntry], *, dagger: bool
    ) -> list[dict[str, Any]]:
        return [
            {
                "polynomial_id": entry.polynomial.candidate_id,
                "support": [list(term) for term in entry.polynomial.support],
                "quotient_weight": entry.polynomial.support_size,
                "physical_check_weight": 4 * entry.polynomial.support_size,
                "dagger_sector": dagger,
                "orbit_rank_over_gf8": len(entry.ideal_basis),
            }
            for entry in generators
        ]

    return {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "families": list(candidate.families),
        "origins": candidate.origins,
        "n": GROUP_ORDER,
        "k": 32 if fixed_checks["rank"] == 96 else GROUP_ORDER - 2 * int(fixed_checks["rank"]),
        "ideal_dimension_over_gf8": len(ideal),
        "annihilator_dimension_over_gf8": len(annihilator),
        "structural_checks": structural_checks,
        "structurally_accepted": structurally_accepted,
        "check_weight": check_weight,
        "ideal_generators": generators_to_dict(ideal_generators, dagger=False),
        "annihilator_generators": generators_to_dict(
            annihilator_generators, dagger=True
        ),
        "distance": distance,
        "distance_support": distance_support,
        "distance_method": distance_method,
        "distance_lower_bound": 6 if distance is None and should_diagnose else distance,
        "distance_upper_bound": 7,
        "distance_seconds": round(distance_seconds, 6),
        "target_distance_six": distance is not None and distance >= 6,
        "target_distance_seven": distance == 7,
        "total_seconds": round(time.perf_counter() - started, 6),
    }
