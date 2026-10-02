"""Compact faithful-GL(2,2) GALA models with an x-reflected ZX fold.

This module is the 32-qubit analogue of ``twisted_halfswap_model``.  The top
group is represented by its faithful two-dimensional binary representation,
which has no nonzero globally fixed vector.  Binary transpose is handled at
the represented-matrix level rather than identified with group inversion.
"""

from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np
from qldpc import codes

from .model import (
    GRID_ORDER,
    GRID_X_ORDER,
    GRID_Y_ORDER,
    Monomial,
    _permute_columns,
    _tanner_connected,
    binary_transpose_terms,
    bottom_support_generates,
    gf2_rank,
    lift_block_permutation,
    ring_polynomial,
    translation_permutations,
    zx_fold_permutation,
)

SCHEMA_VERSION = 1
TOP_REPRESENTATION = "s3-linear"
TOP_ORDER = 6
TOP_LIFT_DIMENSION = 2


def reflect_x_terms(terms: tuple[Monomial, ...]) -> tuple[Monomial, ...]:
    return tuple(
        sorted(
            Monomial(term.top, (-term.x) % GRID_X_ORDER, term.y)
            for term in terms
        )
    )


def faithful_twisted_transpose_terms(
    terms: tuple[Monomial, ...],
) -> tuple[Monomial, ...]:
    """Apply x reflection after actual represented binary transpose."""
    return reflect_x_terms(binary_transpose_terms(TOP_REPRESENTATION, terms))


@functools.lru_cache(maxsize=1)
def faithful_x_reflection_permutation() -> np.ndarray:
    """Return x reflection on one 16-coordinate represented lift block."""
    px, py = translation_permutations(TOP_REPRESENTATION)
    unseen = set(range(len(px)))
    reflection = np.empty(len(px), dtype=int)
    while unseen:
        anchor = min(unseen)
        coordinates: dict[tuple[int, int], int] = {}
        for xx in range(GRID_X_ORDER):
            for yy in range(GRID_Y_ORDER):
                coordinate = anchor
                for _ in range(xx):
                    coordinate = int(px[coordinate])
                for _ in range(yy):
                    coordinate = int(py[coordinate])
                coordinates[xx, yy] = coordinate
        unseen -= set(coordinates.values())
        for (xx, yy), coordinate in coordinates.items():
            reflection[coordinate] = coordinates[(-xx) % GRID_X_ORDER, yy]
    return reflection


@dataclass(frozen=True)
class FaithfulTwistedHalfSwapCandidate:
    f_terms: tuple[Monomial, ...]
    g_terms: tuple[Monomial, ...]
    label: str = ""

    def __post_init__(self) -> None:
        for name, terms in (("F", self.f_terms), ("G", self.g_terms)):
            if not terms:
                raise ValueError(f"{name} support must be nonempty")
            if tuple(sorted(terms)) != terms or len(set(terms)) != len(terms):
                raise ValueError(f"{name} support must be sorted and distinct")
            if any(term.top >= TOP_ORDER for term in terms):
                raise ValueError(f"{name} has an invalid GL(2,2) monomial")
            if faithful_twisted_transpose_terms(terms) != terms:
                raise ValueError(
                    f"{name} support must obey the faithful twisted transpose"
                )

    @property
    def top_representation(self) -> str:
        return TOP_REPRESENTATION

    @property
    def entries(self) -> tuple[tuple[Monomial, ...], ...]:
        return self.f_terms, self.g_terms

    @property
    def half_blocks(self) -> int:
        return 1

    @property
    def num_blocks(self) -> int:
        return 2

    @property
    def top_lift_dimension(self) -> int:
        return TOP_LIFT_DIMENSION

    @property
    def block_size(self) -> int:
        return TOP_LIFT_DIMENSION * GRID_ORDER

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def nominal_check_weight(self) -> int:
        return len(self.f_terms) + len(self.g_terms)

    @property
    def zx_fold_kind(self) -> str:
        return "twisted_half_swap"

    @property
    def zx_internal_permutation(self) -> np.ndarray:
        return faithful_x_reflection_permutation()

    @property
    def candidate_id(self) -> str:
        payload = self.f_terms, self.g_terms
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()
        return (
            "c4xc2-s3-linear-l2-j1-xreflect-halfswap-"
            f"w{self.nominal_check_weight}-{digest[:16]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "family": "faithful-independent-x-reflected-FG",
            "top_representation": self.top_representation,
            "f_terms": [asdict(term) for term in self.f_terms],
            "g_terms": [asdict(term) for term in self.g_terms],
            "label": self.label,
            "num_blocks": self.num_blocks,
            "num_qubits": self.num_qubits,
            "nominal_check_weight": self.nominal_check_weight,
            "zx_relation": (
                "H_Z = row_reflect_x(H_X data_reflect_x_half_swap)"
            ),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "FaithfulTwistedHalfSwapCandidate":
        return cls(
            f_terms=tuple(sorted(Monomial(**term) for term in data["f_terms"])),
            g_terms=tuple(sorted(Monomial(**term) for term in data["g_terms"])),
            label=data.get("label", ""),
        )


def faithful_twisted_check_matrices(
    candidate: FaithfulTwistedHalfSwapCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(
        ring_polynomial(TOP_REPRESENTATION, candidate.f_terms).lift(),
        dtype=np.uint8,
    )
    bb = np.asarray(
        ring_polynomial(TOP_REPRESENTATION, candidate.g_terms).lift(),
        dtype=np.uint8,
    )
    return np.hstack([aa, bb]), np.hstack([bb.T, aa.T])


def build_faithful_twisted_half_swap_code(
    candidate: FaithfulTwistedHalfSwapCandidate,
    *,
    skip_validation: bool = False,
) -> codes.CSSCode:
    del skip_validation
    hx, hz = faithful_twisted_check_matrices(candidate)
    return codes.CSSCode(hx, hz, promise_equal_distance_xz=True)


@functools.lru_cache(maxsize=1)
def faithful_twisted_transpose_orbits() -> tuple[
    tuple[Monomial, ...], tuple[tuple[Monomial, Monomial], ...]
]:
    universe = tuple(
        Monomial(top, xx, yy)
        for top in range(TOP_ORDER)
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    )
    unseen = set(universe)
    fixed: list[Monomial] = []
    pairs: list[tuple[Monomial, Monomial]] = []
    while unseen:
        term = min(unseen)
        partner = faithful_twisted_transpose_terms((term,))[0]
        unseen.remove(term)
        if partner == term:
            fixed.append(term)
        else:
            unseen.remove(partner)
            pairs.append(tuple(sorted((term, partner))))
    return tuple(sorted(fixed)), tuple(sorted(pairs))


def _sample_faithful_twisted_support(
    rng: random.Random, weight: int, *, force_identity: bool
) -> tuple[Monomial, ...]:
    fixed, pairs = faithful_twisted_transpose_orbits()
    identity = Monomial(0, 0, 0)
    required = (identity,) if force_identity else ()
    available_fixed = tuple(term for term in fixed if term not in required)
    remaining = weight - len(required)
    feasible_pair_counts = [
        pair_count
        for pair_count in range(len(pairs) + 1)
        if 0 <= remaining - 2 * pair_count <= len(available_fixed)
    ]
    if not feasible_pair_counts:
        raise ValueError(f"cannot form faithful twisted support of weight {weight}")
    pair_count = rng.choice(feasible_pair_counts)
    fixed_count = remaining - 2 * pair_count
    selected = [*required, *rng.sample(available_fixed, fixed_count)]
    for pair in rng.sample(pairs, pair_count):
        selected.extend(pair)
    return tuple(sorted(selected))


def random_faithful_twisted_half_swap_candidates(
    *, f_weight: int, g_weight: int, count: int, seed: int
) -> Iterator[FaithfulTwistedHalfSwapCandidate]:
    if count < 0 or f_weight < 1 or g_weight < 1:
        raise ValueError("candidate counts and support weights must be positive")
    rng = random.Random(seed)
    seen: set[tuple[tuple[Monomial, ...], tuple[Monomial, ...]]] = set()
    attempts = 0
    maximum_attempts = max(100, 200 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        ff = _sample_faithful_twisted_support(
            rng, f_weight, force_identity=True
        )
        gg = _sample_faithful_twisted_support(
            rng, g_weight, force_identity=False
        )
        key = ff, gg
        if key in seen:
            continue
        seen.add(key)
        yield FaithfulTwistedHalfSwapCandidate(ff, gg)


def _bottom_polynomial_terms(
    support: tuple[tuple[int, int], ...],
) -> tuple[Monomial, ...]:
    """Lift a bottom-group polynomial as scalar multiples of the identity."""
    return tuple(sorted(Monomial(0, xx, yy) for xx, yy in support))


def faithful_commutant_g_terms(
    f_terms: tuple[Monomial, ...],
    *,
    p_support: tuple[tuple[int, int], ...],
    q_support: tuple[tuple[int, int], ...],
) -> tuple[Monomial, ...]:
    """Return the sparse group-ring support of ``p I + q F``.

    The bottom group is central in the direct product, so the represented
    matrices of ``F`` and ``p I + q F`` commute identically.  Addition is over
    GF(2), hence repeated monomials cancel.
    """
    terms = set(_bottom_polynomial_terms(p_support))
    for qx, qy in q_support:
        for term in f_terms:
            shifted = Monomial(
                term.top,
                (term.x + qx) % GRID_X_ORDER,
                (term.y + qy) % GRID_Y_ORDER,
            )
            if shifted in terms:
                terms.remove(shifted)
            else:
                terms.add(shifted)
    return tuple(sorted(terms))


def random_faithful_commutant_candidates(
    *,
    f_weight: int,
    p_weight: int,
    q_weight: int,
    check_weight_ceiling: int,
    count: int,
    seed: int,
) -> Iterator[FaithfulTwistedHalfSwapCandidate]:
    """Sample the exact-CSS ansatz ``G = p I + q F``.

    Here ``p`` and ``q`` are bottom-group polynomials.  The reflected-
    transpose involution fixes every bottom monomial, so an invariant ``F``
    produces an invariant ``G`` as well.
    """
    if min(f_weight, q_weight, count) < 1 or p_weight < 0:
        raise ValueError(
            "F, q, and count must be positive; p may also have weight zero"
        )
    bottom = tuple(
        (xx, yy)
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    )
    if p_weight > len(bottom) or q_weight > len(bottom):
        raise ValueError("bottom-polynomial weight exceeds C4 x C2 order")
    rng = random.Random(seed)
    seen: set[tuple[tuple[Monomial, ...], tuple[Monomial, ...]]] = set()
    attempts = 0
    maximum_attempts = max(1000, 1000 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        ff = _sample_faithful_twisted_support(
            rng, f_weight, force_identity=True
        )
        pp = tuple(sorted(rng.sample(bottom, p_weight)))
        qq = tuple(sorted(rng.sample(bottom, q_weight)))
        gg = faithful_commutant_g_terms(
            ff, p_support=pp, q_support=qq
        )
        if not gg or len(ff) + len(gg) > check_weight_ceiling:
            continue
        key = ff, gg
        if key in seen:
            continue
        seen.add(key)
        yield FaithfulTwistedHalfSwapCandidate(
            ff,
            gg,
            label=f"commutant-p{p_weight}-q{q_weight}",
        )


def analyze_faithful_twisted_half_swap_structure(
    candidate: FaithfulTwistedHalfSwapCandidate,
    *,
    check_weight_ceiling: int = 16,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "n_at_most_200": candidate.num_qubits <= 200,
        "nominal_check_weight_at_most_ceiling": (
            candidate.nominal_check_weight <= check_weight_ceiling
        ),
        "F_obeys_faithful_x_reflected_transpose": (
            faithful_twisted_transpose_terms(candidate.f_terms)
            == candidate.f_terms
        ),
        "G_obeys_faithful_x_reflected_transpose": (
            faithful_twisted_transpose_terms(candidate.g_terms)
            == candidate.g_terms
        ),
        "bottom_support_generates_C4xC2": bottom_support_generates(candidate),
    }
    hx, hz = faithful_twisted_check_matrices(candidate)
    active_css = not bool(np.any((hx @ hz.T) % 2))
    checks["active_css_orthogonality"] = active_css
    if not active_css:
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "checks": checks,
            "rejection_reasons": [
                name for name, passed in checks.items() if not passed
            ],
        }

    code = build_faithful_twisted_half_swap_code(candidate)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    col_weights_x = np.count_nonzero(hx, axis=0)
    col_weights_z = np.count_nonzero(hz, axis=0)
    px, py = translation_permutations(TOP_REPRESENTATION)
    full_px = lift_block_permutation(px, candidate.num_blocks)
    full_py = lift_block_permutation(py, candidate.num_blocks)
    zx_fold = zx_fold_permutation(candidate)
    row_reflection = faithful_x_reflection_permutation()
    folded_hx = _permute_columns(hx, zx_fold)
    folded_hz = _permute_columns(hz, zx_fold)

    def translation_is_automorphism(permutation: np.ndarray) -> bool:
        return bool(
            gf2_rank(
                np.vstack([hx, _permute_columns(hx, permutation)]), code.field
            )
            == code.code_x.rank
            and gf2_rank(
                np.vstack([hz, _permute_columns(hz, permutation)]), code.field
            )
            == code.code_z.rank
        )

    checks.update(
        {
            "actual_check_weight_at_most_ceiling": bool(
                row_weights_x.max(initial=0) <= check_weight_ceiling
                and row_weights_z.max(initial=0) <= check_weight_ceiling
            ),
            "no_zero_checks": bool(
                row_weights_x.min(initial=1) > 0
                and row_weights_z.min(initial=1) > 0
            ),
            "k_at_least_8": code.dimension >= 8,
            "tanner_x_connected": _tanner_connected(hx),
            "tanner_z_connected": _tanner_connected(hz),
            "C4_translation_automorphism": translation_is_automorphism(full_px),
            "C2_translation_automorphism": translation_is_automorphism(full_py),
            "strict_ZX_faithful_x_reflected_half_swap": bool(
                np.array_equal(folded_hx[row_reflection], hz)
                and np.array_equal(folded_hz[row_reflection], hx)
            ),
        }
    )
    rejected = [name for name, passed in checks.items() if not passed]
    return {
        "candidate": candidate.to_dict(),
        "accepted": not rejected,
        "checks": checks,
        "rejection_reasons": rejected,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": code.code_x.rank,
        "rank_z": code.code_z.rank,
        "rate": code.dimension / code.num_qubits,
        "useful_rate": GRID_ORDER / code.num_qubits,
        "check_weights_x": sorted(set(map(int, row_weights_x))),
        "check_weights_z": sorted(set(map(int, row_weights_z))),
        "column_degrees_x": sorted(set(map(int, col_weights_x))),
        "column_degrees_z": sorted(set(map(int, col_weights_z))),
        "zx_fold_is_identity": bool(
            np.array_equal(zx_fold, np.arange(candidate.num_qubits))
        ),
        "zx_fold_permutation": zx_fold.astype(int).tolist(),
        "zx_row_reflection": row_reflection.astype(int).tolist(),
    }
