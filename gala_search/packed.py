"""Packed Ising-lattice BB codes with a ``C_8 x C_4`` logical quotient.

The physical group is ``C_(8m) x C_4``.  The order-``m`` subgroup generated
by ``x^8`` supplies code thickness, while the quotient supplies the desired
logical translations.  The weight-eight polynomial family is

    a = (1 + x^(8r)) + x^u y^v (1 + x^(8s)).

Taking the second BB generator to be the antipode of ``a`` makes the two CSS
check matrices identical.  Each physical half then has 32 candidate logical
fibres, one for every element of ``C_8 x C_4``.
"""

from __future__ import annotations

import functools
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np

from qldpc import abstract, codes

from .core import _solve_minimum_weight_parity_problem

PACKED_SCHEMA_VERSION = 1
LOGICAL_X_ORDER = 8
LOGICAL_Y_ORDER = 4
LOGICALS_PER_HALF = LOGICAL_X_ORDER * LOGICAL_Y_ORDER
NUM_LOGICALS = 2 * LOGICALS_PER_HALF


def _canonical_support(
    support: Sequence[tuple[int, int]], long_order: int
) -> tuple[tuple[int, int], ...]:
    """Canonicalize a support under translations and axis reflections."""
    orbit: list[tuple[tuple[int, int], ...]] = []
    for sign_x, sign_y in itertools.product((-1, 1), repeat=2):
        reflected = tuple(
            (
                (sign_x * xx) % long_order,
                (sign_y * yy) % LOGICAL_Y_ORDER,
            )
            for xx, yy in support
        )
        for anchor_x, anchor_y in reflected:
            orbit.append(
                tuple(
                    sorted(
                        (
                            (xx - anchor_x) % long_order,
                            (yy - anchor_y) % LOGICAL_Y_ORDER,
                        )
                        for xx, yy in reflected
                    )
                )
            )
    return min(orbit)


def _format_polynomial(support: Sequence[tuple[int, int]]) -> str:
    terms: list[str] = []
    for xx, yy in support:
        if xx == 0 and yy == 0:
            terms.append("1")
        elif xx == 0:
            terms.append(f"y^{yy}")
        elif yy == 0:
            terms.append(f"x^{xx}")
        else:
            terms.append(f"x^{xx} y^{yy}")
    return " + ".join(terms)


@dataclass(frozen=True, order=True)
class PackedCandidate:
    """A four-term polynomial in the packed weight-eight ansatz."""

    m: int
    r: int
    s: int
    u: int
    v: int

    def __post_init__(self) -> None:
        if self.m < 3 or self.m % 2 == 0:
            raise ValueError(f"m must be odd and at least 3, got {self.m}")
        if not 0 < self.r < self.m or not 0 < self.s < self.m:
            raise ValueError("r and s must be nonzero reduced exponents modulo m")
        if not 0 <= self.u < self.long_order:
            raise ValueError(f"u must be reduced modulo {self.long_order}")
        if not 0 <= self.v < LOGICAL_Y_ORDER:
            raise ValueError(f"v must be reduced modulo {LOGICAL_Y_ORDER}")

    @property
    def long_order(self) -> int:
        return LOGICAL_X_ORDER * self.m

    @property
    def group_order(self) -> int:
        return self.long_order * LOGICAL_Y_ORDER

    @property
    def candidate_id(self) -> str:
        return f"packed-m{self.m}-r{self.r}-s{self.s}-u{self.u}-v{self.v}"

    @property
    def support(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                {
                    (0, 0),
                    ((LOGICAL_X_ORDER * self.r) % self.long_order, 0),
                    (self.u, self.v),
                    (
                        (self.u + LOGICAL_X_ORDER * self.s) % self.long_order,
                        self.v,
                    ),
                }
            )
        )

    @property
    def polynomial(self) -> str:
        return _format_polynomial(self.support)

    @property
    def equivalence_key(self) -> tuple[tuple[int, int], ...]:
        """Canonical support under translations and axis reflections.

        These operations preserve the thickness subgroup and the logical
        ``C_8 x C_4`` quotient, up to inversion of either logical coordinate.
        It is sufficient to translate each support point to the origin rather
        than scan every group translation.
        """
        return _canonical_support(self.support, self.long_order)

    def to_dict(self) -> dict[str, Any]:
        return {"family": "packed_weight8", **asdict(self)}


@dataclass(frozen=True, order=True)
class PackedWeightTwelveCandidate:
    """Three paired terms whose quotient offsets generate ``C_8 x C_4``.

    The normalized polynomial is

        (1 + x^(8 r0))
        + x^(1 + 8 p1) (1 + x^(8 r1))
        + x^(8 p2) y (1 + x^(8 r2)).

    The three quotient locations are ``(0,0)``, ``(1,0)``, and ``(0,1)``.
    Their differences generate the complete logical translation quotient, in
    contrast to the necessarily cyclic quotient support of the two-pair ansatz.
    """

    m: int
    r0: int
    r1: int
    r2: int
    p1: int
    p2: int

    def __post_init__(self) -> None:
        if self.m < 3 or self.m % 2 == 0:
            raise ValueError(f"m must be odd and at least 3, got {self.m}")
        if any(not 0 < value < self.m for value in (self.r0, self.r1, self.r2)):
            raise ValueError("r0, r1, and r2 must be nonzero modulo m")
        if any(not 0 <= value < self.m for value in (self.p1, self.p2)):
            raise ValueError("p1 and p2 must be reduced modulo m")

    @property
    def long_order(self) -> int:
        return LOGICAL_X_ORDER * self.m

    @property
    def group_order(self) -> int:
        return self.long_order * LOGICAL_Y_ORDER

    @property
    def candidate_id(self) -> str:
        return (
            f"packed12-m{self.m}-r{self.r0}-{self.r1}-{self.r2}"
            f"-p{self.p1}-{self.p2}"
        )

    @property
    def support(self) -> tuple[tuple[int, int], ...]:
        first_x = 1 + LOGICAL_X_ORDER * self.p1
        second_x = LOGICAL_X_ORDER * self.p2
        return tuple(
            sorted(
                {
                    (0, 0),
                    ((LOGICAL_X_ORDER * self.r0) % self.long_order, 0),
                    (first_x % self.long_order, 0),
                    (
                        (first_x + LOGICAL_X_ORDER * self.r1) % self.long_order,
                        0,
                    ),
                    (second_x % self.long_order, 1),
                    (
                        (second_x + LOGICAL_X_ORDER * self.r2) % self.long_order,
                        1,
                    ),
                }
            )
        )

    @property
    def polynomial(self) -> str:
        return _format_polynomial(self.support)

    @property
    def equivalence_key(self) -> tuple[tuple[int, int], ...]:
        return _canonical_support(self.support, self.long_order)

    def to_dict(self) -> dict[str, Any]:
        return {"family": "packed_weight12", **asdict(self)}


PackedSearchCandidate = PackedCandidate | PackedWeightTwelveCandidate


def iter_packed_candidates(
    m: int, *, quotient_symmetries: bool = True
) -> Iterator[PackedCandidate]:
    """Enumerate distinct normalized weight-eight packed candidates."""
    if m < 3 or m % 2 == 0:
        raise ValueError(f"m must be odd and at least 3, got {m}")

    seen: set[tuple[tuple[int, int], ...]] = set()
    # Each thickness pair is unordered, so only the shorter separation is
    # needed.  The complete relative placement is retained through u and v.
    for r in range(1, (m + 1) // 2):
        for s in range(1, (m + 1) // 2):
            for u in range(LOGICAL_X_ORDER * m):
                for v in range(LOGICAL_Y_ORDER):
                    candidate = PackedCandidate(m=m, r=r, s=s, u=u, v=v)
                    if len(candidate.support) != 4:
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


def iter_packed_weight_twelve_candidates(
    m: int, *, quotient_symmetries: bool = True
) -> Iterator[PackedWeightTwelveCandidate]:
    """Enumerate the normalized connected weight-twelve search family."""
    if m < 3 or m % 2 == 0:
        raise ValueError(f"m must be odd and at least 3, got {m}")

    seen: set[tuple[tuple[int, int], ...]] = set()
    separations = range(1, (m + 1) // 2)
    for r0, r1, r2 in itertools.product(separations, repeat=3):
        for p1, p2 in itertools.product(range(m), repeat=2):
            candidate = PackedWeightTwelveCandidate(
                m=m, r0=r0, r1=r1, r2=r2, p1=p1, p2=p2
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
    m: int,
) -> tuple[abstract.GroupRing, abstract.RingMember, abstract.RingMember]:
    ring = abstract.GroupRing(
        abstract.AbelianGroup(LOGICAL_X_ORDER * m, LOGICAL_Y_ORDER)
    )
    xx, yy = ring.generators
    return ring, xx, yy


def polynomial_member(candidate: PackedSearchCandidate) -> abstract.RingMember:
    ring, xx, yy = _ring_data(candidate.m)
    return functools.reduce(
        lambda left, right: left + right,
        (xx**power_x * yy**power_y for power_x, power_y in candidate.support),
        ring.zero,
    )


def build_packed_code(candidate: PackedSearchCandidate) -> codes.GALACode:
    aa = polynomial_member(candidate)
    return codes.GALACode(
        generators_f=[aa],
        generators_g=[aa.T],
        num_active_rows=1,
    )


def packed_logicals(candidate: PackedSearchCandidate) -> np.ndarray[Any, Any]:
    """Return the 64 disjoint logical-fibre supports on both BB halves."""
    ring, xx, yy = _ring_data(candidate.m)
    omega = sum(
        (xx ** (LOGICAL_X_ORDER * thickness) for thickness in range(candidate.m)),
        ring.zero,
    )
    rows: list[np.ndarray[Any, Any]] = []
    for half in range(2):
        for logical_x in range(LOGICAL_X_ORDER):
            for logical_y in range(LOGICAL_Y_ORDER):
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


def _gf2_rank(matrix: np.ndarray[Any, Any], field: type[np.ndarray[Any, Any]]) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=int).view(field)))


def _logical_index(half: int, logical_x: int, logical_y: int) -> int:
    return half * LOGICALS_PER_HALF + logical_x * LOGICAL_Y_ORDER + logical_y


def _translation_action_is_regular(
    candidate: PackedSearchCandidate, logicals: np.ndarray[Any, Any]
) -> tuple[bool, bool]:
    ring, xx, yy = _ring_data(candidate.m)
    for half in range(2):
        for logical_x in range(LOGICAL_X_ORDER):
            for logical_y in range(LOGICAL_Y_ORDER):
                index = _logical_index(half, logical_x, logical_y)
                row = logicals[index]
                support = row[
                    half * candidate.group_order : (half + 1) * candidate.group_order
                ]
                member = abstract.RingMember.from_vector(support, ring)

                translated_x = np.asarray((xx * member).to_vector(), dtype=int)
                expected_x = logicals[
                    _logical_index(
                        half, (logical_x + 1) % LOGICAL_X_ORDER, logical_y
                    )
                ][half * candidate.group_order : (half + 1) * candidate.group_order]
                if not np.array_equal(translated_x, expected_x):
                    return False, False

                translated_y = np.asarray((yy * member).to_vector(), dtype=int)
                expected_y = logicals[
                    _logical_index(
                        half, logical_x, (logical_y + 1) % LOGICAL_Y_ORDER
                    )
                ][half * candidate.group_order : (half + 1) * candidate.group_order]
                if not np.array_equal(translated_y, expected_y):
                    return True, False
    return True, True


def _tanner_connected(code: codes.GALACode) -> bool:
    return bool(nx.is_connected(code.code_x.graph.to_undirected()))


@functools.lru_cache(maxsize=None)
def _group_action_permutations(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index permutations for x, y, and group inversion."""
    ring, xx, yy = _ring_data(m)
    group_order = LOGICAL_X_ORDER * m * LOGICAL_Y_ORDER

    def multiplication_permutation(member: abstract.RingMember) -> np.ndarray:
        permutation: list[int] = []
        for index in range(group_order):
            vector = np.zeros(group_order, dtype=int)
            vector[index] = 1
            basis = abstract.RingMember.from_vector(vector, ring)
            translated = np.asarray((member * basis).to_vector(), dtype=int)
            permutation.append(int(np.flatnonzero(translated)[0]))
        return np.asarray(permutation, dtype=int)

    inversion: list[int] = []
    for index in range(group_order):
        vector = np.zeros(group_order, dtype=int)
        vector[index] = 1
        basis = abstract.RingMember.from_vector(vector, ring)
        inverted = np.asarray(basis.T.to_vector(), dtype=int)
        inversion.append(int(np.flatnonzero(inverted)[0]))
    return (
        multiplication_permutation(xx),
        multiplication_permutation(yy),
        np.asarray(inversion, dtype=int),
    )


def packed_automorphism_checks(candidate: PackedSearchCandidate) -> dict[str, bool]:
    """Verify physical permutations that underpin the logical gate schedule."""
    code = build_packed_code(candidate)
    check = np.asarray(code.matrix_x, dtype=int)
    group_order = candidate.group_order
    perm_x, perm_y, inversion = _group_action_permutations(candidate.m)

    def packed_row_set(matrix: np.ndarray[Any, Any]) -> set[bytes]:
        return {
            np.packbits(np.asarray(row, dtype=np.uint8)).tobytes() for row in matrix
        }

    original_rows = packed_row_set(check)

    def preserves_stabilizers(permutation: np.ndarray) -> bool:
        transformed = np.zeros_like(check)
        transformed[:, permutation] = check
        return packed_row_set(transformed) == original_rows

    return {
        "x_translation_stabilizer_automorphism": preserves_stabilizers(
            np.hstack([perm_x, group_order + perm_x])
        ),
        "y_translation_stabilizer_automorphism": preserves_stabilizers(
            np.hstack([perm_y, group_order + perm_y])
        ),
        "half_swap_inversion_automorphism": preserves_stabilizers(
            np.hstack([group_order + inversion, inversion])
        ),
    }


def analyze_packed_candidate(
    candidate: PackedSearchCandidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Apply every algebraic acceptance test before distance certification."""
    started = time.perf_counter()
    code = build_packed_code(candidate)
    logicals_raw = packed_logicals(candidate)
    logicals = code.field(logicals_raw)
    check = code.matrix_x

    integer_check = np.asarray(check, dtype=int)
    row_weights = np.count_nonzero(integer_check, axis=1)
    column_weights = np.count_nonzero(integer_check, axis=0)
    logical_weights = np.count_nonzero(logicals_raw, axis=1)
    rank_h = code.code_x.rank
    rank_logicals = _gf2_rank(logicals_raw, code.field)
    rank_gain = _gf2_rank(np.vstack([check, logicals]), code.field) - rank_h
    pairing = logicals @ logicals.T
    x_regular, y_regular = _translation_action_is_regular(candidate, logicals_raw)
    support_multiplicity = np.count_nonzero(logicals_raw, axis=0)
    expected_rank = candidate.group_order - LOGICALS_PER_HALF

    expected_terms = 4 if isinstance(candidate, PackedCandidate) else 6
    expected_check_weight = 2 * expected_terms
    checks = {
        f"{expected_terms}_distinct_terms": len(candidate.support) == expected_terms,
        "self_dual": bool(np.array_equal(code.matrix_x, code.matrix_z)),
        "css_orthogonal": bool(not np.any(code.matrix_x @ code.matrix_z.T)),
        f"uniform_check_weight_{expected_check_weight}": bool(
            np.all(row_weights == expected_check_weight)
        ),
        "rank_is_group_order_minus_32": rank_h == expected_rank,
        "dimension_is_64": code.dimension == NUM_LOGICALS,
        "candidate_logicals_commute": bool(not np.any(check @ logicals.T)),
        "candidate_logicals_rank_64": rank_logicals == NUM_LOGICALS,
        "candidate_logicals_complete": rank_gain == NUM_LOGICALS,
        "candidate_logicals_orthonormal": bool(
            np.array_equal(pairing, np.eye(NUM_LOGICALS, dtype=int))
        ),
        "candidate_logicals_weight_m": bool(np.all(logical_weights == candidate.m)),
        "candidate_logicals_partition_qubits": bool(
            np.all(support_multiplicity == 1)
        ),
        "even_syndrome_parity": bool(np.all(column_weights % 2 == 0)),
        "x_translation_is_C8": x_regular,
        "y_translation_is_C4": y_regular,
    }
    if include_graph_metrics:
        checks["tanner_connected"] = _tanner_connected(code)

    rejection_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": PACKED_SCHEMA_VERSION,
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
        "distance_upper_bound": candidate.m if not rejection_reasons else None,
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def certify_packed_distance(
    candidate: PackedSearchCandidate,
    *,
    solver: str = "HIGHS",
    target_distance: int | None = None,
    logical_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Certify distance using translation representatives from both halves.

    All 32 logical fibres in a fixed half are related by physical translations,
    so their constrained minimum weights agree.  Every nontrivial logical class
    anticommutes with at least one fibre because the fibres form a complete
    orthonormal logical basis.  It is therefore enough to solve one class from
    each physical half (indices 0 and 32).  Supplying ``logical_indices`` is
    mainly useful for independent audits.
    """
    analysis = analyze_packed_candidate(candidate)
    if not analysis["accepted"]:
        raise ValueError(
            f"candidate failed algebraic checks: {analysis['rejection_reasons']}"
        )
    if target_distance is not None and target_distance < 1:
        raise ValueError("target_distance must be positive")

    indices = [0, LOGICALS_PER_HALF] if logical_indices is None else list(logical_indices)
    if not indices or any(index not in range(NUM_LOGICALS) for index in indices):
        raise ValueError(f"logical indices must be selected from range({NUM_LOGICALS})")

    code = build_packed_code(candidate)
    logicals = packed_logicals(candidate)
    class_results: list[dict[str, Any]] = []
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
        if target_distance is not None and weight < target_distance:
            break

    both_halves_complete = set(indices[:2]) == {0, LOGICALS_PER_HALF} and len(
        class_results
    ) == 2
    exact_distance = (
        min(result["minimum_weight"] for result in class_results)
        if both_halves_complete
        else None
    )
    threshold_passed = (
        target_distance is not None
        and both_halves_complete
        and exact_distance is not None
        and exact_distance >= target_distance
    )
    return {
        "schema_version": PACKED_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "canonical_support": [list(term) for term in candidate.equivalence_key],
        "polynomial": candidate.polynomial,
        "solver": solver,
        "logical_indices": indices,
        "translation_representatives": [0, LOGICALS_PER_HALF],
        "complete": both_halves_complete,
        "certified_by_translation_symmetry": both_halves_complete,
        "target_distance": target_distance,
        "threshold_passed": threshold_passed,
        "distance_upper_bound": min(
            result["minimum_weight"] for result in class_results
        ),
        "certified_distance": exact_distance,
        "class_results": class_results,
        "total_seconds": round(
            sum(result["seconds"] for result in class_results), 6
        ),
    }
