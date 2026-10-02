"""Unconstrained, distance-first search over CSS stabilizer subspaces.

The search deliberately does not prescribe logical representatives, a physical
translation, or a ZX fold.  A state is just a pair of commuting binary row
spaces of equal rank.  Ising structure is a post-processing question once a
distance-six code has been found.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TARGET_DISTANCE = 6


def canonical_basis(rows: Iterable[int], n: int) -> tuple[int, ...]:
    """Return the unique reduced basis of a packed binary row space."""
    pivots = [0] * n
    for original in rows:
        value = int(original) & ((1 << n) - 1)
        for pivot in range(n - 1, -1, -1):
            if value >> pivot & 1 and pivots[pivot]:
                value ^= pivots[pivot]
        if not value:
            continue
        pivot = value.bit_length() - 1
        for index, known in enumerate(pivots):
            if known >> pivot & 1:
                pivots[index] ^= value
        pivots[pivot] = value
    return tuple(pivots[index] for index in range(n - 1, -1, -1) if pivots[index])


def span(basis: Sequence[int]) -> tuple[int, ...]:
    output = [0]
    for row in basis:
        output.extend(value ^ int(row) for value in tuple(output))
    return tuple(output)


def nullspace_basis(rows: Sequence[int], n: int) -> tuple[int, ...]:
    """Return a basis for vectors orthogonal to every supplied row."""
    reduced = canonical_basis(rows, n)
    pivot_bits = {row.bit_length() - 1 for row in reduced}
    output: list[int] = []
    for free_bit in range(n):
        if free_bit in pivot_bits:
            continue
        vector = 1 << free_bit
        for row in reduced:
            pivot = row.bit_length() - 1
            if (row & vector).bit_count() & 1:
                vector ^= 1 << pivot
        output.append(vector)
    return canonical_basis(output, n)


def orthogonal(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(not ((a & b).bit_count() & 1) for a in left for b in right)


def masks_to_supports(rows: Sequence[int], n: int) -> list[list[int]]:
    return [[bit for bit in range(n) if row >> bit & 1] for row in rows]


def minimum_weight_basis(rows: Sequence[int], n: int) -> tuple[int, ...]:
    """Find a minimum-weight generating basis by binary-matroid greediness."""
    target = canonical_basis(rows, n)
    candidates = sorted(
        (word for word in span(target) if word),
        key=lambda word: (word.bit_count(), word),
    )
    selected_rows: list[int] = []
    selected_space: tuple[int, ...] = ()
    for candidate in candidates:
        extended = canonical_basis((*selected_space, candidate), n)
        if len(extended) == len(selected_space) + 1:
            selected_rows.append(candidate)
            selected_space = extended
        if len(selected_space) == len(target):
            return tuple(selected_rows)
    raise RuntimeError("failed to recover a stabilizer basis")


def logical_spectrum(
    commuting_checks: Sequence[int],
    stabilizers: Sequence[int],
    n: int,
    *,
    count_through: int = 7,
) -> dict[str, Any]:
    """Enumerate one CSS logical sector exactly."""
    kernel = span(nullspace_basis(commuting_checks, n))
    stabilizer_space = set(span(canonical_basis(stabilizers, n)))
    counts: Counter[int] = Counter()
    distance = n + 1
    witness = 0
    logicals = 0
    for operator in kernel:
        if operator in stabilizer_space:
            continue
        logicals += 1
        weight = operator.bit_count()
        if weight <= count_through:
            counts[weight] += 1
        if weight < distance:
            distance = weight
            witness = operator
    return {
        "distance": distance,
        "witness": witness,
        "witness_support": [bit for bit in range(n) if witness >> bit & 1],
        "logical_vectors": logicals,
        "weight_counts_through_seven": {
            str(weight): counts[weight] for weight in range(1, count_through + 1)
        },
    }


@dataclass(frozen=True)
class CSSState:
    n: int
    basis_x: tuple[int, ...]
    basis_z: tuple[int, ...]

    def __post_init__(self) -> None:
        canonical_x = canonical_basis(self.basis_x, self.n)
        canonical_z = canonical_basis(self.basis_z, self.n)
        if canonical_x != self.basis_x or canonical_z != self.basis_z:
            raise ValueError("CSSState bases must be canonical")
        if not orthogonal(self.basis_x, self.basis_z):
            raise ValueError("CSS stabilizer spaces do not commute")

    @property
    def rank_x(self) -> int:
        return len(self.basis_x)

    @property
    def rank_z(self) -> int:
        return len(self.basis_z)

    @property
    def k(self) -> int:
        return self.n - self.rank_x - self.rank_z

    @property
    def key(self) -> str:
        payload = ",".join(map(str, (*self.basis_x, -1, *self.basis_z))).encode()
        return hashlib.sha256(payload).hexdigest()[:20]


def analyze_css(state: CSSState) -> dict[str, Any]:
    x_logicals = logical_spectrum(state.basis_z, state.basis_x, state.n)
    z_logicals = logical_spectrum(state.basis_x, state.basis_z, state.n)
    displayed_x = minimum_weight_basis(state.basis_x, state.n)
    displayed_z = minimum_weight_basis(state.basis_z, state.n)
    distance_x = int(x_logicals["distance"])
    distance_z = int(z_logicals["distance"])
    return {
        "n": state.n,
        "k": state.k,
        "rank_x": state.rank_x,
        "rank_z": state.rank_z,
        "css_orthogonal": orthogonal(state.basis_x, state.basis_z),
        "distance": min(distance_x, distance_z),
        "distance_x": distance_x,
        "distance_z": distance_z,
        "x_logicals": x_logicals,
        "z_logicals": z_logicals,
        "basis_x_masks": list(state.basis_x),
        "basis_z_masks": list(state.basis_z),
        "basis_x": masks_to_supports(state.basis_x, state.n),
        "basis_z": masks_to_supports(state.basis_z, state.n),
        "minimum_weight_basis_x_masks": list(displayed_x),
        "minimum_weight_basis_z_masks": list(displayed_z),
        "minimum_weight_basis_x": masks_to_supports(displayed_x, state.n),
        "minimum_weight_basis_z": masks_to_supports(displayed_z, state.n),
        "minimum_basis_weights_x": [row.bit_count() for row in displayed_x],
        "minimum_basis_weights_z": [row.bit_count() for row in displayed_z],
        "minimum_maximum_check_weight": max(
            [row.bit_count() for row in (*displayed_x, *displayed_z)] + [0]
        ),
        "state_key": state.key,
    }


def distance_score(analysis: dict[str, Any]) -> tuple[int, ...]:
    """Lexicographically prefer removing the lightest logical operators."""
    x_counts = analysis["x_logicals"]["weight_counts_through_seven"]
    z_counts = analysis["z_logicals"]["weight_counts_through_seven"]
    counts = [int(x_counts[str(w)]) + int(z_counts[str(w)]) for w in range(1, 8)]
    return (
        int(analysis["distance"]),
        *(-counts[weight - 1] for weight in range(1, TARGET_DISTANCE)),
        -sum(counts),
        -int(analysis["minimum_maximum_check_weight"]),
    )


def scalar_penalty(analysis: dict[str, Any]) -> int:
    """A smoother penalty used only for controlled downhill moves."""
    total = 0
    for sector in ("x_logicals", "z_logicals"):
        counts = analysis[sector]["weight_counts_through_seven"]
        for weight in range(1, TARGET_DISTANCE):
            total += int(counts[str(weight)]) * 16 ** (TARGET_DISTANCE - weight)
    return total


def _random_independent_vectors(
    container_basis: Sequence[int], rank: int, n: int, rng: random.Random
) -> tuple[int, ...]:
    words = span(container_basis)[1:]
    selected: tuple[int, ...] = ()
    while len(selected) < rank:
        candidate = words[rng.randrange(len(words))]
        extended = canonical_basis((*selected, candidate), n)
        if len(extended) == len(selected) + 1:
            selected = extended
    return selected


def random_css_state(n: int, rank: int, rng: random.Random) -> CSSState:
    """Sample a CSS pair with ranks `(rank, rank)` and no logical ansatz."""
    if 2 * rank != n - 2:
        raise ValueError("this search currently targets k=2")
    ambient = tuple(1 << bit for bit in range(n))
    basis_x = _random_independent_vectors(ambient, rank, n, rng)
    kernel_x = nullspace_basis(basis_x, n)
    basis_z = _random_independent_vectors(kernel_x, rank, n, rng)
    return CSSState(n, basis_x, basis_z)


def mutate_state(state: CSSState, sector: str, rng: random.Random) -> CSSState:
    """Replace one stabilizer direction while preserving CSS and both ranks."""
    if sector == "x":
        current, opposite = state.basis_x, state.basis_z
    elif sector == "z":
        current, opposite = state.basis_z, state.basis_x
    else:
        raise ValueError("sector must be x or z")
    removed = rng.randrange(len(current))
    remaining = current[:removed] + current[removed + 1 :]
    candidates = span(nullspace_basis(opposite, state.n))[1:]
    rng.shuffle(candidates := list(candidates))
    replacement_basis = None
    for candidate in candidates:
        proposed = canonical_basis((*remaining, candidate), state.n)
        if len(proposed) == len(current) and proposed != current:
            replacement_basis = proposed
            break
    if replacement_basis is None:
        return state
    if sector == "x":
        return CSSState(state.n, replacement_basis, state.basis_z)
    return CSSState(state.n, state.basis_x, replacement_basis)


def _compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return analysis


def run_restart(
    *,
    n: int,
    rank: int,
    seed: int,
    iterations: int,
    proposals_per_iteration: int,
    downhill_probability: float = 0.02,
) -> dict[str, Any]:
    """Run one stochastic hill-climb restart and return its exact best state."""
    rng = random.Random(seed)
    current = random_css_state(n, rank, rng)
    current_analysis = analyze_css(current)
    current_score = distance_score(current_analysis)
    current_penalty = scalar_penalty(current_analysis)
    best = current
    best_analysis = current_analysis
    best_score = current_score
    counters: Counter[str] = Counter()
    counters[f"initial_distance_{current_analysis['distance']}"] += 1

    for iteration in range(iterations):
        sector = "x" if rng.randrange(2) == 0 else "z"
        proposals: list[tuple[tuple[int, ...], int, CSSState, dict[str, Any]]] = []
        seen: set[str] = set()
        for _ in range(proposals_per_iteration):
            candidate = mutate_state(current, sector, rng)
            if candidate.key in seen or candidate == current:
                continue
            seen.add(candidate.key)
            analysis = analyze_css(candidate)
            proposals.append(
                (distance_score(analysis), scalar_penalty(analysis), candidate, analysis)
            )
            counters["evaluations"] += 1
        if not proposals:
            counters["empty_iterations"] += 1
            continue
        proposal_score, proposal_penalty, proposal, proposal_analysis = max(
            proposals, key=lambda item: (item[0], -item[1])
        )
        accept = proposal_score >= current_score
        if not accept and rng.random() < downhill_probability:
            delta = max(0, proposal_penalty - current_penalty)
            scale = max(1, current_penalty // 8)
            accept = rng.random() < math.exp(-delta / scale)
        if accept:
            current = proposal
            current_analysis = proposal_analysis
            current_score = proposal_score
            current_penalty = proposal_penalty
            counters["accepted_moves"] += 1
        else:
            counters["rejected_iterations"] += 1
        if proposal_score > best_score:
            best = proposal
            best_analysis = proposal_analysis
            best_score = proposal_score
            counters["new_best"] += 1
        if int(best_analysis["distance"]) >= TARGET_DISTANCE:
            counters["target_hit"] += 1
            return {
                "seed": seed,
                "iterations_completed": iteration + 1,
                "target_hit": True,
                "counters": dict(counters),
                "best": _compact_analysis(best_analysis),
            }
    return {
        "seed": seed,
        "iterations_completed": iterations,
        "target_hit": False,
        "counters": dict(counters),
        "best": _compact_analysis(best_analysis),
    }
