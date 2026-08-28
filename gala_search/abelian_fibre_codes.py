"""Self-dual abelian BB codes with disjoint order-seven logical fibres.

Two compact targets are supported:

* ``C_28 x C_4`` with quotient ``C_4 x C_4`` and target ``[[224,32,7]]``;
* ``C_7 x C_4`` with quotient ``C_4`` and target ``[[56,8,7]]``.

In both cases the first cyclic factor contains an order-seven thickness
subgroup.  Every polynomial is a sum of pairs separated inside that subgroup,
which forces the disjoint order-seven fibre sums into the kernel.
"""

from __future__ import annotations

import functools
import itertools
import math
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from qldpc import abstract, codes

from .core import _solve_minimum_weight_parity_problem
from .packed import _canonical_support, _format_polynomial

SCHEMA_VERSION = 1
THICKNESS = 7
Y_ORDER = 4


def _xor_support(
    terms: Sequence[tuple[int, int]], long_order: int
) -> tuple[tuple[int, int], ...]:
    """Reduce a list of group-ring monomials over ``GF(2)``."""
    parity: Counter[tuple[int, int]] = Counter(
        (xx % long_order, yy % Y_ORDER) for xx, yy in terms
    )
    return tuple(sorted(term for term, count in parity.items() if count % 2))


def _paired_support(
    logical_x_order: int,
    pairs: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int], ...]:
    """Expand ``(base_x, base_y, separation)`` thickness pairs."""
    long_order = logical_x_order * THICKNESS
    terms = []
    for base_x, base_y, separation in pairs:
        terms.extend(
            (
                (base_x, base_y),
                (base_x + logical_x_order * separation, base_y),
            )
        )
    return _xor_support(terms, long_order)


@dataclass(frozen=True, order=True)
class FibreCodeCandidate:
    """One paired polynomial over ``C_(7q) x C_4``."""

    logical_x_order: int
    ansatz: str
    parameters: tuple[int, ...]
    support: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.logical_x_order not in (1, 4):
            raise ValueError("logical_x_order must be 1 or 4")
        if len(self.support) > 6 or len(self.support) % 2:
            raise ValueError("paired supports must have even weight at most six")

    @property
    def long_order(self) -> int:
        return self.logical_x_order * THICKNESS

    @property
    def group_order(self) -> int:
        return self.long_order * Y_ORDER

    @property
    def logicals_per_half(self) -> int:
        return self.logical_x_order * Y_ORDER

    @property
    def num_logicals(self) -> int:
        return 2 * self.logicals_per_half

    @property
    def num_qubits(self) -> int:
        return 2 * self.group_order

    @property
    def target(self) -> str:
        return f"[[{self.num_qubits},{self.num_logicals},7]]"

    @property
    def group_name(self) -> str:
        return f"C{self.long_order} x C4"

    @property
    def candidate_id(self) -> str:
        values = "-".join(map(str, self.parameters))
        return f"fibre-q{self.logical_x_order}-{self.ansatz}-{values}"

    @property
    def polynomial(self) -> str:
        return _format_polynomial(self.support)

    @property
    def equivalence_key(self) -> tuple[tuple[int, int], ...]:
        return _canonical_support(self.support, self.long_order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "abelian_order_seven_fibres",
            "group": self.group_name,
            "target": self.target,
            "logical_x_order": self.logical_x_order,
            "ansatz": self.ansatz,
            "parameters": list(self.parameters),
            "support": [list(term) for term in self.support],
        }


def make_four_term_candidate(r: int, s: int, p: int) -> FibreCodeCandidate:
    """Return ``(1+x^r)+x^p y(1+x^s)`` over ``C_7 x C_4``."""
    support = _paired_support(1, ((0, 0, r), (p, 1, s)))
    return FibreCodeCandidate(1, "weight8", (r, s, p), support)


def make_six_term_candidate(
    logical_x_order: int,
    r0: int,
    r1: int,
    r2: int,
    p1: int,
    p2: int,
    v1: int = 0,
    v2: int = 1,
) -> FibreCodeCandidate:
    """Construct a normalized three-pair candidate.

    For quotient ``C_4 x C_4`` the quotient locations are fixed to
    ``(0,0)``, ``(1,0)``, and ``(0,1)``.  For quotient ``C_4`` the caller
    supplies the last two ``y`` locations.
    """
    if logical_x_order == 4:
        bases = ((0, 0), (1 + 4 * p1, 0), (4 * p2, 1))
        v1, v2 = 0, 1
    elif logical_x_order == 1:
        bases = ((0, 0), (p1, v1), (p2, v2))
    else:
        raise ValueError("logical_x_order must be 1 or 4")
    support = _paired_support(
        logical_x_order,
        tuple(
            (base_x, base_y, separation)
            for (base_x, base_y), separation in zip(bases, (r0, r1, r2))
        ),
    )
    return FibreCodeCandidate(
        logical_x_order,
        "weight12",
        (r0, r1, r2, p1, p2, v1, v2),
        support,
    )


def iter_four_term_candidates(
    *, quotient_symmetries: bool = True
) -> Iterator[FibreCodeCandidate]:
    """Enumerate the normalized four-term ``C_7 x C_4`` family."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r, s in itertools.product(range(1, 4), repeat=2):
        for p in range(THICKNESS):
            candidate = make_four_term_candidate(r, s, p)
            key = candidate.equivalence_key if quotient_symmetries else candidate.support
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def iter_six_term_candidates(
    logical_x_order: int, *, quotient_symmetries: bool = True
) -> Iterator[FibreCodeCandidate]:
    """Enumerate the connected normalized three-pair family."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    separations = range(1, 4)
    quotient_locations = (
        ((0, 1),)
        if logical_x_order == 4
        else tuple(
            (v1, v2)
            for v1, v2 in itertools.product(range(Y_ORDER), repeat=2)
            if math.gcd(v1, v2, Y_ORDER) == 1
        )
    )
    for r0, r1, r2 in itertools.product(separations, repeat=3):
        for p1, p2 in itertools.product(range(THICKNESS), repeat=2):
            for v1, v2 in quotient_locations:
                candidate = make_six_term_candidate(
                    logical_x_order, r0, r1, r2, p1, p2, v1, v2
                )
                if len(candidate.support) != 6:
                    continue
                key = (
                    candidate.equivalence_key
                    if quotient_symmetries
                    else candidate.support
                )
                if key in seen:
                    continue
                seen.add(key)
                yield candidate


@functools.lru_cache(maxsize=None)
def _ring_data(
    logical_x_order: int,
) -> tuple[abstract.GroupRing, abstract.RingMember, abstract.RingMember]:
    ring = abstract.GroupRing(
        abstract.AbelianGroup(logical_x_order * THICKNESS, Y_ORDER)
    )
    xx, yy = ring.generators
    return ring, xx, yy


def polynomial_member(candidate: FibreCodeCandidate) -> abstract.RingMember:
    ring, xx, yy = _ring_data(candidate.logical_x_order)
    return functools.reduce(
        lambda left, right: left + right,
        (xx**power_x * yy**power_y for power_x, power_y in candidate.support),
        ring.zero,
    )


def build_code(candidate: FibreCodeCandidate) -> codes.GALACode:
    polynomial = polynomial_member(candidate)
    return codes.GALACode(
        generators_f=[polynomial],
        generators_g=[polynomial.T],
        num_active_rows=1,
    )


def fibre_logicals(candidate: FibreCodeCandidate) -> np.ndarray[Any, Any]:
    """Return the complete disjoint logical-fibre basis on both halves."""
    ring, xx, yy = _ring_data(candidate.logical_x_order)
    omega = sum(
        (
            xx ** (candidate.logical_x_order * thickness)
            for thickness in range(THICKNESS)
        ),
        ring.zero,
    )
    rows = []
    for half in range(2):
        for logical_x in range(candidate.logical_x_order):
            for logical_y in range(Y_ORDER):
                support = np.asarray(
                    (xx**logical_x * yy**logical_y * omega).to_vector(), dtype=int
                )
                zeros = np.zeros(candidate.group_order, dtype=int)
                rows.append(
                    np.hstack([support, zeros])
                    if half == 0
                    else np.hstack([zeros, support])
                )
    return np.asarray(rows, dtype=int)


def _gf2_rank(matrix: np.ndarray[Any, Any], field: Any) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=int).view(field)))


def _translation_checks(
    candidate: FibreCodeCandidate, logicals: np.ndarray[Any, Any]
) -> dict[str, bool]:
    ring, xx, yy = _ring_data(candidate.logical_x_order)
    x_ok = True
    y_ok = True
    for half in range(2):
        offset = half * candidate.logicals_per_half
        half_slice = slice(
            half * candidate.group_order, (half + 1) * candidate.group_order
        )
        for logical_x in range(candidate.logical_x_order):
            for logical_y in range(Y_ORDER):
                index = offset + logical_x * Y_ORDER + logical_y
                member = abstract.RingMember.from_vector(logicals[index][half_slice], ring)
                shifted_x = np.asarray((xx * member).to_vector(), dtype=int)
                expected_x = logicals[
                    offset
                    + ((logical_x + 1) % candidate.logical_x_order) * Y_ORDER
                    + logical_y
                ][half_slice]
                x_ok &= np.array_equal(shifted_x, expected_x)
                shifted_y = np.asarray((yy * member).to_vector(), dtype=int)
                expected_y = logicals[
                    offset + logical_x * Y_ORDER + (logical_y + 1) % Y_ORDER
                ][half_slice]
                y_ok &= np.array_equal(shifted_y, expected_y)
    return {
        f"x_translation_is_C{candidate.logical_x_order}": bool(x_ok),
        "y_translation_is_C4": bool(y_ok),
    }


def analyze_candidate(
    candidate: FibreCodeCandidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Apply all algebraic and Ising-architecture acceptance tests."""
    started = time.perf_counter()
    code = build_code(candidate)
    logicals_raw = fibre_logicals(candidate)
    logicals = code.field(logicals_raw)
    check = code.matrix_x
    integer_check = np.asarray(check, dtype=int)
    row_weights = np.count_nonzero(integer_check, axis=1)
    column_weights = np.count_nonzero(integer_check, axis=0)
    logical_weights = np.count_nonzero(logicals_raw, axis=1)
    support_multiplicity = np.count_nonzero(logicals_raw, axis=0)
    rank_h = code.code_x.rank
    rank_logicals = _gf2_rank(logicals_raw, code.field)
    rank_gain = _gf2_rank(np.vstack([check, logicals]), code.field) - rank_h
    pairing = logicals @ logicals.T
    expected_rank = candidate.group_order - candidate.logicals_per_half
    translation_checks = _translation_checks(candidate, logicals_raw)
    checks = {
        "self_dual": bool(np.array_equal(code.matrix_x, code.matrix_z)),
        "css_orthogonal": bool(not np.any(code.matrix_x @ code.matrix_z.T)),
        "uniform_check_weight": bool(
            np.all(row_weights == 2 * len(candidate.support))
        ),
        "target_check_rank": rank_h == expected_rank,
        "target_dimension": code.dimension == candidate.num_logicals,
        "fibre_logicals_commute": bool(not np.any(check @ logicals.T)),
        "fibre_logicals_independent": rank_logicals == candidate.num_logicals,
        "fibre_logicals_complete": rank_gain == candidate.num_logicals,
        "fibre_logicals_orthonormal": bool(
            np.array_equal(pairing, np.eye(candidate.num_logicals, dtype=int))
        ),
        "fibre_logicals_weight_7": bool(np.all(logical_weights == THICKNESS)),
        "fibre_logicals_partition_qubits": bool(
            np.all(support_multiplicity == 1)
        ),
        "even_syndrome_parity": bool(np.all(column_weights % 2 == 0)),
        **translation_checks,
    }
    if include_graph_metrics:
        checks["tanner_connected"] = bool(
            nx.is_connected(code.code_x.graph.to_undirected())
        )
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "canonical_support": [list(term) for term in candidate.equivalence_key],
        "polynomial": candidate.polynomial,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_h": rank_h,
        "expected_rank_h": expected_rank,
        "check_weight": int(row_weights.max(initial=0)),
        "qubit_degree": int(column_weights.max(initial=0)),
        "logical_weight": int(logical_weights[0]),
        "num_logicals": len(logicals_raw),
        "distance_upper_bound": THICKNESS if not rejection_reasons else None,
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def certify_distance(
    candidate: FibreCodeCandidate, *, solver: str = "HIGHS"
) -> dict[str, Any]:
    """Certify exact distance using one translated class from each half."""
    analysis = analyze_candidate(candidate)
    if not analysis["accepted"]:
        raise ValueError(
            f"candidate failed algebraic checks: {analysis['rejection_reasons']}"
        )
    code = build_code(candidate)
    logicals = fibre_logicals(candidate)
    indices = [0, candidate.logicals_per_half]
    class_results = []
    for logical_index in indices:
        effective_check = np.vstack([code.matrix_x, logicals[logical_index]])
        syndrome = np.zeros(len(effective_check), dtype=int)
        syndrome[-1] = 1
        weight, support, status, elapsed = _solve_minimum_weight_parity_problem(
            effective_check, syndrome, solver=solver
        )
        class_results.append(
            {
                "logical_index": logical_index,
                "minimum_weight": weight,
                "support": support,
                "solver_status": status,
                "seconds": round(elapsed, 6),
            }
        )
    exact_distance = min(result["minimum_weight"] for result in class_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomial": candidate.polynomial,
        "solver": solver,
        "complete": True,
        "certified_by_translation_symmetry": True,
        "translation_representatives": indices,
        "certified_distance": exact_distance,
        "distance_upper_bound": exact_distance,
        "target_distance": THICKNESS,
        "threshold_passed": exact_distance >= THICKNESS,
        "class_results": class_results,
        "total_seconds": round(
            sum(result["seconds"] for result in class_results), 6
        ),
    }
