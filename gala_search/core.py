"""Construction and analysis of paired-polynomial GALA codes.

The weight-eight family is

    a(x, y) = 1 + y**r + x*y**s + x*y**(s+t)

over GF(2)[C_4 x C_m], with m odd.  The associated minimal GALA code is

    GALACode(generators_f=[a], generators_g=[a.T], num_active_rows=1).

The module deliberately keeps randomized upper bounds and certified distances in
separate fields.  A large randomized upper bound is evidence, not a proof.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np

from qldpc import abstract, codes
from qldpc.objects import Pauli

SEARCH_SCHEMA_VERSION = 1
ELL = 4


@dataclass(frozen=True, order=True)
class WeightEightCandidate:
    """Exponent data for a normalized four-term paired polynomial."""

    m: int
    r: int
    s: int
    t: int

    def __post_init__(self) -> None:
        if self.m < 3 or self.m % 2 == 0:
            raise ValueError(f"m must be odd and at least 3, got {self.m}")
        normalized = (self.r % self.m, self.s % self.m, self.t % self.m)
        if normalized != (self.r, self.s, self.t):
            raise ValueError("candidate exponents must already be reduced modulo m")
        if self.r == 0 or self.t == 0:
            raise ValueError("r and t must be nonzero modulo m")

    @property
    def family(self) -> str:
        return "weight8"

    @property
    def candidate_id(self) -> str:
        return f"w8-m{self.m}-r{self.r}-s{self.s}-t{self.t}"

    @property
    def support(self) -> tuple[tuple[int, int], ...]:
        """Sorted polynomial support as (x exponent, y exponent) pairs."""
        return tuple(
            sorted(
                {
                    (0, 0),
                    (0, self.r),
                    (1, self.s),
                    (1, (self.s + self.t) % self.m),
                }
            )
        )

    @property
    def polynomial(self) -> str:
        terms = []
        for xx, yy in self.support:
            if xx == 0 and yy == 0:
                terms.append("1")
            elif xx == 0:
                terms.append(f"y^{yy}")
            elif yy == 0:
                terms.append("x")
            else:
                terms.append(f"x y^{yy}")
        return " + ".join(terms)

    @property
    def equivalence_key(self) -> tuple[tuple[int, int], ...]:
        """Canonical support under torus translations and axis reflections.

        These transformations preserve the C_4 and C_m directions rather than
        mixing them, so they retain the intended logical-chain interpretation.
        """
        orbit: list[tuple[tuple[int, int], ...]] = []
        for sign_x, sign_y in itertools.product((-1, 1), repeat=2):
            reflected = [
                ((sign_x * xx) % ELL, (sign_y * yy) % self.m) for xx, yy in self.support
            ]
            for shift_x in range(ELL):
                for shift_y in range(self.m):
                    transformed = tuple(
                        sorted(
                            (
                                (xx + shift_x) % ELL,
                                (yy + shift_y) % self.m,
                            )
                            for xx, yy in reflected
                        )
                    )
                    orbit.append(transformed)
        return min(orbit)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, **asdict(self)}


def iter_weight_eight_candidates(
    m: int, *, quotient_symmetries: bool = True
) -> Iterator[WeightEightCandidate]:
    """Yield weight-eight candidates without duplicate polynomial supports.

    The two y positions in the x sector form an unordered pair, so combinations
    avoid generating the same polynomial twice.  Optionally keep only one
    representative under torus translations and independent axis reflections.
    """
    if m < 3 or m % 2 == 0:
        raise ValueError(f"m must be odd and at least 3, got {m}")

    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(1, m):
        for y_one, y_two in itertools.combinations(range(m), 2):
            candidate = WeightEightCandidate(m=m, r=r, s=y_one, t=(y_two - y_one) % m)
            key = candidate.equivalence_key if quotient_symmetries else candidate.support
            if key in seen:
                continue
            seen.add(key)
            yield candidate


@functools.lru_cache(maxsize=None)
def _ring_data(m: int) -> tuple[abstract.GroupRing, abstract.RingMember, abstract.RingMember]:
    ring = abstract.GroupRing(abstract.AbelianGroup(ELL, m))
    xx, yy = ring.generators
    return ring, xx, yy


def polynomial_member(candidate: WeightEightCandidate) -> abstract.RingMember:
    """Build the group-ring polynomial for a candidate."""
    ring, xx, yy = _ring_data(candidate.m)
    return functools.reduce(
        lambda left, right: left + right,
        (xx**power_x * yy**power_y for power_x, power_y in candidate.support),
        ring.zero,
    )


def build_code(candidate: WeightEightCandidate) -> codes.GALACode:
    """Build Q_{4,m}(a) through qLDPC's GALACode implementation."""
    aa = polynomial_member(candidate)
    return codes.GALACode(
        generators_f=[aa],
        generators_g=[aa.T],
        num_active_rows=1,
    )


def candidate_logicals(candidate: WeightEightCandidate) -> np.ndarray[Any, Any]:
    """Return the eight disjoint x^i omega_m supports on the L and R halves."""
    ring, xx, yy = _ring_data(candidate.m)
    omega = sum((yy**power for power in range(candidate.m)), ring.zero)
    half_size = ELL * candidate.m
    rows: list[np.ndarray[Any, Any]] = []
    for half in range(2):
        for logical_index in range(ELL):
            support = np.asarray((xx**logical_index * omega).to_vector(), dtype=int)
            zeros = np.zeros(half_size, dtype=int)
            rows.append(np.hstack([support, zeros]) if half == 0 else np.hstack([zeros, support]))
    return np.asarray(rows, dtype=int)


def _gf2_rank(matrix: np.ndarray[Any, Any], field: type[np.ndarray[Any, Any]]) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=int).view(field)))


def _count_four_cycles(check_matrix: np.ndarray[Any, Any]) -> int:
    integer_matrix = np.asarray(check_matrix, dtype=np.int64)
    overlaps = integer_matrix @ integer_matrix.T
    upper = overlaps[np.triu_indices_from(overlaps, k=1)]
    return int(np.sum(upper * (upper - 1) // 2))


def _tanner_metrics(code: codes.GALACode) -> tuple[bool, int | None, int]:
    graph = code.code_x.graph.to_undirected()
    connected = nx.is_connected(graph)
    girth = int(nx.girth(graph)) if graph.number_of_edges() else None
    return connected, girth, _count_four_cycles(code.matrix_x)


def _shift_action_is_two_four_cycles(candidate: WeightEightCandidate, logicals: np.ndarray) -> bool:
    ring, xx, _yy = _ring_data(candidate.m)
    half_size = ELL * candidate.m
    for half in range(2):
        for logical_index in range(ELL):
            row = logicals[half * ELL + logical_index]
            support = row[half * half_size : (half + 1) * half_size]
            member = abstract.RingMember.from_vector(support, ring)
            translated = np.asarray((xx * member).to_vector(), dtype=int)
            expected_row = logicals[half * ELL + (logical_index + 1) % ELL]
            expected = expected_row[half * half_size : (half + 1) * half_size]
            if not np.array_equal(translated, expected):
                return False
    return True


def deterministic_seed(candidate: WeightEightCandidate, base_seed: int = 0) -> int:
    """Derive a stable per-candidate random seed."""
    payload = f"{base_seed}:{candidate.candidate_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def analyze_candidate(
    candidate: WeightEightCandidate,
    *,
    distance_trials: int = 0,
    base_seed: int = 0,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Construct and algebraically screen one candidate."""
    started = time.perf_counter()
    code = build_code(candidate)
    logicals_raw = candidate_logicals(candidate)
    logicals = code.field(logicals_raw)
    check = code.matrix_x

    self_dual = bool(np.array_equal(code.matrix_x, code.matrix_z))
    css_orthogonal = bool(not np.any(code.matrix_x @ code.matrix_z.T))
    integer_check = np.asarray(check, dtype=int)
    row_weights = np.count_nonzero(integer_check, axis=1)
    column_weights = np.count_nonzero(integer_check, axis=0)
    rank_h = code.code_x.rank
    rank_logicals = _gf2_rank(logicals_raw, code.field)
    rank_with_logicals = _gf2_rank(np.vstack([check, logicals]), code.field)
    rank_gain = rank_with_logicals - rank_h
    logical_commutes = bool(not np.any(check @ logicals.T))
    pairing = logicals @ logicals.T
    orthonormal_logicals = bool(np.array_equal(pairing, np.eye(2 * ELL, dtype=int)))
    logical_weights = np.count_nonzero(logicals_raw, axis=1)
    disjoint_logicals = bool(np.all(np.count_nonzero(logicals_raw, axis=0) <= 1))
    even_syndrome_parity = bool(np.all(column_weights % 2 == 0))
    shift_two_cycles = _shift_action_is_two_four_cycles(candidate, logicals_raw)

    if include_graph_metrics:
        tanner_connected, tanner_girth, num_four_cycles = _tanner_metrics(code)
    else:
        tanner_connected, tanner_girth, num_four_cycles = True, None, -1

    expected_rank = ELL * candidate.m - ELL
    checks = {
        "four_distinct_terms": len(candidate.support) == 4,
        "self_dual": self_dual,
        "css_orthogonal": css_orthogonal,
        "uniform_check_weight_8": bool(np.all(row_weights == 8)),
        "rank_is_4m_minus_4": rank_h == expected_rank,
        "dimension_is_8": code.dimension == 2 * ELL,
        "candidate_logicals_commute": logical_commutes,
        "candidate_logicals_rank_8": rank_logicals == 2 * ELL,
        "candidate_logicals_complete": rank_gain == 2 * ELL,
        "candidate_logicals_orthonormal": orthonormal_logicals,
        "candidate_logicals_weight_m": bool(np.all(logical_weights == candidate.m)),
        "candidate_logicals_disjoint": disjoint_logicals,
        "even_syndrome_parity": even_syndrome_parity,
        "shift_is_two_four_cycles": shift_two_cycles,
        "tanner_connected": tanner_connected,
    }
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    accepted = not rejection_reasons

    seed = deterministic_seed(candidate, base_seed)
    decoder_bound: int | None = None
    distance_upper_bound: int | None = candidate.m if accepted else None
    distance_seconds = 0.0
    if accepted and distance_trials > 0:
        distance_started = time.perf_counter()
        random.seed(seed)
        np.random.seed(seed)
        decoder_bound = int(
            code.get_distance_bound_with_decoder(Pauli.X, num_trials=distance_trials)
        )
        distance_upper_bound = min(candidate.m, decoder_bound)
        distance_seconds = time.perf_counter() - distance_started

    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "canonical_support": [list(term) for term in candidate.equivalence_key],
        "polynomial": candidate.polynomial,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_h": rank_h,
        "expected_rank_h": expected_rank,
        "check_weight": int(row_weights.max(initial=0)),
        "qubit_degree": int(column_weights.max(initial=0)),
        "logical_weight": int(logical_weights[0]),
        "tanner_girth": tanner_girth,
        "num_four_cycles": num_four_cycles,
        "distance_upper_bound": distance_upper_bound,
        "decoder_distance_upper_bound": decoder_bound,
        "distance_trials": distance_trials,
        "random_seed": seed,
        "distance_seconds": round(distance_seconds, 6),
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def _solve_minimum_weight_parity_problem(
    check_matrix: np.ndarray[Any, Any],
    syndrome: np.ndarray[Any, Any],
    *,
    solver: str,
) -> tuple[int, list[int], str, float]:
    """Solve min |e| subject to check_matrix @ e == syndrome (mod 2)."""
    import cvxpy as cp

    check = np.asarray(check_matrix, dtype=int)
    target = np.asarray(syndrome, dtype=int)
    num_checks, num_qubits = check.shape
    error = cp.Variable(num_qubits, boolean=True)
    slack = cp.Variable(num_checks, integer=True)
    max_slack = (np.sum(check, axis=1) - target) // 2
    constraints = [
        check @ error == target + 2 * slack,
        slack >= 0,
        slack <= max_slack,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(error)), constraints)
    started = time.perf_counter()
    result = problem.solve(solver=solver)
    elapsed = time.perf_counter() - started
    if problem.status != cp.OPTIMAL or error.value is None or not np.isfinite(result):
        raise RuntimeError(f"exact-distance ILP did not terminate optimally: {problem.status}")

    solution = np.rint(error.value).astype(int)
    if not np.array_equal(check @ solution % 2, target):
        raise RuntimeError("ILP returned a solution with the wrong binary syndrome")
    return int(np.count_nonzero(solution)), np.flatnonzero(solution).astype(int).tolist(), problem.status, elapsed


def certify_distance(
    candidate: WeightEightCandidate,
    *,
    solver: str = "HIGHS",
    logical_indices: Sequence[int] | None = None,
    stop_at: int | None = None,
) -> dict[str, Any]:
    """Certify the exact distance with at most eight minimum-weight ILPs.

    For a zero-syndrome vector e, L @ e is its logical class in the known
    orthonormal logical basis L.  It is nontrivial iff at least one component is
    one.  Therefore the distance is the minimum, over logical basis rows L_i, of
    min |e| subject to H @ e = 0 and L_i @ e = 1.
    """
    analysis = analyze_candidate(candidate, distance_trials=0)
    if not analysis["accepted"]:
        raise ValueError(f"candidate failed algebraic checks: {analysis['rejection_reasons']}")

    code = build_code(candidate)
    logicals = candidate_logicals(candidate)
    indices = list(range(2 * ELL)) if logical_indices is None else list(logical_indices)
    if not indices or any(index not in range(2 * ELL) for index in indices):
        raise ValueError("logical indices must be selected from range(8)")
    if stop_at is not None and stop_at < 1:
        raise ValueError("stop_at must be positive")

    class_results: list[dict[str, Any]] = []
    for logical_index in indices:
        effective_check = np.vstack([code.matrix_x, logicals[logical_index]])
        syndrome = np.zeros(len(effective_check), dtype=int)
        syndrome[-1] = 1
        weight, support, status, elapsed = _solve_minimum_weight_parity_problem(
            effective_check,
            syndrome,
            solver=solver,
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
        if stop_at is not None and weight <= stop_at:
            break

    complete = len(class_results) == 2 * ELL
    ilp_upper_bound = min(result["minimum_weight"] for result in class_results)

    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "canonical_support": [list(term) for term in candidate.equivalence_key],
        "polynomial": candidate.polynomial,
        "solver": solver,
        "logical_indices": indices,
        "completed_logical_indices": [result["logical_index"] for result in class_results],
        "complete": complete,
        "stop_at": stop_at,
        "threshold_met": stop_at is not None and ilp_upper_bound <= stop_at,
        "ilp_distance_upper_bound": ilp_upper_bound,
        "certified_distance": ilp_upper_bound if complete else None,
        "class_results": class_results,
        "total_seconds": round(sum(result["seconds"] for result in class_results), 6),
    }
