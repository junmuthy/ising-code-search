"""Compact independent-``F/G`` GALA models with an x-reflected ZX fold.

This family again uses the natural permutation lift of ``S3 x C4 x C2`` and
``L=2,J=1,n=48``.  It replaces individual binary-transpose invariance by

``F^T = alpha(F)`` and ``G^T = alpha(G)``,

where ``alpha`` reflects the ``C4`` coordinate, ``x -> x^-1``.  Transversal
Hadamard followed by that physical reflection and a swap of the two 24-qubit
halves therefore exchanges the X and Z stabilizer spaces.  This condition is
strictly broader than inverse-closed support: x exponents no longer have to
occur in inverse pairs.
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
TOP_REPRESENTATION = "s3-natural"
TOP_ORDER = 6


def reflect_x_terms(terms: tuple[Monomial, ...]) -> tuple[Monomial, ...]:
    """Apply the bottom-group automorphism ``x -> x^-1, y -> y``."""
    return tuple(
        sorted(
            Monomial(term.top, (-term.x) % GRID_X_ORDER, term.y)
            for term in terms
        )
    )


def twisted_transpose_terms(
    terms: tuple[Monomial, ...],
) -> tuple[Monomial, ...]:
    """Apply ``alpha`` after represented binary transpose.

    A support fixed by this involution obeys ``support^dagger=alpha(support)``.
    """
    return reflect_x_terms(binary_transpose_terms(TOP_REPRESENTATION, terms))


@functools.lru_cache(maxsize=1)
def x_reflection_permutation() -> np.ndarray:
    """Return the physical x-reflection on one 24-coordinate lift block."""
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
                coordinates[(xx, yy)] = coordinate
        orbit = set(coordinates.values())
        if len(orbit) != GRID_ORDER:
            raise RuntimeError("bottom translations did not produce an eight-site orbit")
        unseen -= orbit
        for (xx, yy), coordinate in coordinates.items():
            reflection[coordinate] = coordinates[((-xx) % GRID_X_ORDER, yy)]
    if len(set(map(int, reflection))) != len(reflection):
        raise RuntimeError("x reflection is not a permutation")
    if not np.array_equal(reflection[reflection], np.arange(len(reflection))):
        raise RuntimeError("x reflection is not involutive")
    reflection.flags.writeable = False
    return reflection


@dataclass(frozen=True)
class TwistedHalfSwapCandidate:
    """One independent ``F/G`` candidate with an x-reflected ZX fold."""

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
                raise ValueError(f"{name} has an invalid S3 monomial")
            if twisted_transpose_terms(terms) != terms:
                raise ValueError(f"{name} support must obey the twisted transpose fold")

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
        return 3

    @property
    def block_size(self) -> int:
        return self.top_lift_dimension * GRID_ORDER

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
        return x_reflection_permutation()

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr((self.f_terms, self.g_terms)).encode()).hexdigest()
        return (
            "c4xc2-s3-natural-l2-j1-xreflect-halfswap-"
            f"w{self.nominal_check_weight}-{digest[:16]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "family": "independent-x-reflected-FG",
            "top_representation": self.top_representation,
            "f_terms": [asdict(term) for term in self.f_terms],
            "g_terms": [asdict(term) for term in self.g_terms],
            "label": self.label,
            "num_blocks": self.num_blocks,
            "num_qubits": self.num_qubits,
            "nominal_check_weight": self.nominal_check_weight,
            "zx_relation": "H_Z = row_reflect_x(H_X data_reflect_x_half_swap)",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwistedHalfSwapCandidate":
        return cls(
            f_terms=tuple(sorted(Monomial(**term) for term in data["f_terms"])),
            g_terms=tuple(sorted(Monomial(**term) for term in data["g_terms"])),
            label=data.get("label", ""),
        )


def build_twisted_half_swap_code(
    candidate: TwistedHalfSwapCandidate, *, skip_validation: bool = False
) -> codes.CSSCode:
    ff = ring_polynomial(TOP_REPRESENTATION, candidate.f_terms)
    gg = ring_polynomial(TOP_REPRESENTATION, candidate.g_terms)
    return codes.GALACode(
        [ff], [gg], num_active_rows=1, skip_validation=skip_validation
    )


@functools.lru_cache(maxsize=1)
def twisted_transpose_orbits() -> tuple[
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
        partner = twisted_transpose_terms((term,))[0]
        unseen.remove(term)
        if partner == term:
            fixed.append(term)
        else:
            unseen.remove(partner)
            pairs.append(tuple(sorted((term, partner))))
    return tuple(sorted(fixed)), tuple(sorted(pairs))


@functools.lru_cache(maxsize=1)
def twisted_support_orbits() -> tuple[tuple[Monomial, ...], ...]:
    """Return all one- and two-term orbit supports in deterministic order."""
    fixed, pairs = twisted_transpose_orbits()
    return tuple((term,) for term in fixed) + tuple(pairs)


def _support_orbit_ids(terms: tuple[Monomial, ...]) -> frozenset[int]:
    lookup = {
        term: orbit_id
        for orbit_id, orbit in enumerate(twisted_support_orbits())
        for term in orbit
    }
    orbit_ids = frozenset(lookup[term] for term in terms)
    reconstructed = tuple(
        sorted(
            term
            for orbit_id in orbit_ids
            for term in twisted_support_orbits()[orbit_id]
        )
    )
    if reconstructed != terms:
        raise ValueError("support is not a union of twisted-transpose orbits")
    return orbit_ids


def twisted_support_neighbors(
    candidate: TwistedHalfSwapCandidate,
) -> Iterator[tuple[TwistedHalfSwapCandidate, dict[str, Any]]]:
    """Enumerate all one-orbit swaps that preserve both support weights.

    CSS commutation and the logical translation algebra are intentionally
    checked by the caller.  Keeping only candidates that pass those exact
    gates turns this raw neighborhood into a commutation- and symmetry-
    preserving search graph.
    """
    orbits = twisted_support_orbits()
    identity = Monomial(0, 0, 0)
    for side in ("F", "G"):
        original = candidate.f_terms if side == "F" else candidate.g_terms
        occupied = _support_orbit_ids(original)
        for removed_id in sorted(occupied):
            removed = orbits[removed_id]
            if side == "F" and identity in removed:
                continue
            for added_id, added in enumerate(orbits):
                if added_id in occupied or len(added) != len(removed):
                    continue
                updated_ids = (occupied - {removed_id}) | {added_id}
                updated = tuple(
                    sorted(
                        term
                        for orbit_id in updated_ids
                        for term in orbits[orbit_id]
                    )
                )
                neighbor = (
                    TwistedHalfSwapCandidate(updated, candidate.g_terms)
                    if side == "F"
                    else TwistedHalfSwapCandidate(candidate.f_terms, updated)
                )
                yield neighbor, {
                    "side": side,
                    "removed_orbit": [asdict(term) for term in removed],
                    "added_orbit": [asdict(term) for term in added],
                }


def sample_twisted_double_neighbors(
    candidate: TwistedHalfSwapCandidate,
    *,
    count: int,
    seed: int,
) -> Iterator[tuple[TwistedHalfSwapCandidate, dict[str, Any]]]:
    """Sample final supports at exact two-orbit replacement distance.

    The two swaps are composed before any structural or distance gate is
    applied.  This permits the search to cross a bad one-swap intermediate.
    Final candidates must replace exactly two occupied support orbits by two
    previously unoccupied orbits, distributed arbitrarily between F and G.
    Every yielded support therefore retains its term weight, the
    twisted-transpose constraint, and the normalized identity anchor in F.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return
    rng = random.Random(seed)
    orbits = twisted_support_orbits()
    identity = Monomial(0, 0, 0)
    original_ids = {
        "F": _support_orbit_ids(candidate.f_terms),
        "G": _support_orbit_ids(candidate.g_terms),
    }
    removable = {
        "F": tuple(
            orbit_id
            for orbit_id in sorted(original_ids["F"])
            if identity not in orbits[orbit_id]
        ),
        "G": tuple(sorted(original_ids["G"])),
    }
    available = {
        side: tuple(
            orbit_id
            for orbit_id in range(len(orbits))
            if orbit_id not in original_ids[side]
        )
        for side in ("F", "G")
    }
    distributions = tuple(
        (f_count, 2 - f_count)
        for f_count in range(3)
        if len(removable["F"]) >= f_count
        and len(removable["G"]) >= 2 - f_count
        and len(available["F"]) >= f_count
        and len(available["G"]) >= 2 - f_count
    )
    if not distributions:
        return
    seen: set[str] = set()
    attempts = 0
    maximum_attempts = max(1000, 200 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        f_count, g_count = rng.choice(distributions)
        removed_ids = {
            "F": tuple(sorted(rng.sample(removable["F"], f_count))),
            "G": tuple(sorted(rng.sample(removable["G"], g_count))),
        }
        added_ids = {
            "F": tuple(sorted(rng.sample(available["F"], f_count))),
            "G": tuple(sorted(rng.sample(available["G"], g_count))),
        }
        if any(
            sum(len(orbits[orbit_id]) for orbit_id in removed_ids[side])
            != sum(len(orbits[orbit_id]) for orbit_id in added_ids[side])
            for side in ("F", "G")
        ):
            continue
        final_ids = {
            side: (original_ids[side] - set(removed_ids[side]))
            | set(added_ids[side])
            for side in ("F", "G")
        }
        final_terms = {
            side: tuple(
                sorted(
                    term
                    for orbit_id in final_ids[side]
                    for term in orbits[orbit_id]
                )
            )
            for side in ("F", "G")
        }
        final = TwistedHalfSwapCandidate(final_terms["F"], final_terms["G"])
        if final.candidate_id in seen:
            continue
        seen.add(final.candidate_id)
        yield final, {
            "kind": "coordinated_two_orbit",
            "removed_orbits": {
                side: [
                    [asdict(term) for term in orbits[orbit_id]]
                    for orbit_id in removed_ids[side]
                ]
                for side in ("F", "G")
            },
            "added_orbits": {
                side: [
                    [asdict(term) for term in orbits[orbit_id]]
                    for orbit_id in added_ids[side]
                ]
                for side in ("F", "G")
            },
        }


def _sample_twisted_support(
    rng: random.Random, weight: int, *, force_identity: bool
) -> tuple[Monomial, ...]:
    fixed, pairs = twisted_transpose_orbits()
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
        raise ValueError(f"cannot form twisted support of weight {weight}")
    pair_count = rng.choice(feasible_pair_counts)
    fixed_count = remaining - 2 * pair_count
    selected = [*required, *rng.sample(available_fixed, fixed_count)]
    for pair in rng.sample(pairs, pair_count):
        selected.extend(pair)
    return tuple(sorted(selected))


def random_twisted_half_swap_candidates(
    *, f_weight: int, g_weight: int, count: int, seed: int
) -> Iterator[TwistedHalfSwapCandidate]:
    if count < 0 or f_weight < 1 or g_weight < 1:
        raise ValueError("candidate counts and support weights must be positive")
    rng = random.Random(seed)
    seen: set[tuple[tuple[Monomial, ...], tuple[Monomial, ...]]] = set()
    attempts = 0
    maximum_attempts = max(100, 200 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        ff = _sample_twisted_support(rng, f_weight, force_identity=True)
        gg = _sample_twisted_support(rng, g_weight, force_identity=False)
        key = ff, gg
        if key in seen:
            continue
        seen.add(key)
        yield TwistedHalfSwapCandidate(ff, gg)


def analyze_twisted_half_swap_structure(
    candidate: TwistedHalfSwapCandidate, *, check_weight_ceiling: int = 16
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "n_at_most_200": candidate.num_qubits <= 200,
        "nominal_check_weight_at_most_ceiling": (
            candidate.nominal_check_weight <= check_weight_ceiling
        ),
        "F_obeys_x_reflected_transpose": (
            binary_transpose_terms(TOP_REPRESENTATION, candidate.f_terms)
            == reflect_x_terms(candidate.f_terms)
        ),
        "G_obeys_x_reflected_transpose": (
            binary_transpose_terms(TOP_REPRESENTATION, candidate.g_terms)
            == reflect_x_terms(candidate.g_terms)
        ),
        "bottom_support_generates_C4xC2": bottom_support_generates(candidate),
    }
    if not all(checks.values()):
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "checks": checks,
            "rejection_reasons": [name for name, passed in checks.items() if not passed],
        }
    try:
        code = build_twisted_half_swap_code(candidate)
    except ValueError as error:
        checks["active_css_orthogonality"] = False
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "checks": checks,
            "rejection_reasons": ["active_css_orthogonality"],
            "construction_error": str(error),
        }

    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    col_weights_x = np.count_nonzero(hx, axis=0)
    col_weights_z = np.count_nonzero(hz, axis=0)
    px, py = translation_permutations(TOP_REPRESENTATION)
    full_px = lift_block_permutation(px, candidate.num_blocks)
    full_py = lift_block_permutation(py, candidate.num_blocks)
    zx_fold = zx_fold_permutation(candidate)
    row_reflection = x_reflection_permutation()
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
            "active_css_orthogonality": not bool(np.any((hx @ hz.T) % 2)),
            "strict_ZX_x_reflected_half_swap": bool(
                np.array_equal(folded_hx[row_reflection], hz)
                and np.array_equal(folded_hz[row_reflection], hx)
            ),
            "k_at_least_8": code.dimension >= 8,
            "actual_check_weight_at_most_ceiling": bool(
                row_weights_x.max(initial=0) <= check_weight_ceiling
                and row_weights_z.max(initial=0) <= check_weight_ceiling
            ),
            "no_zero_checks": bool(
                np.all(row_weights_x > 0) and np.all(row_weights_z > 0)
            ),
            "C4_translation_automorphism": translation_is_automorphism(full_px),
            "C2_translation_automorphism": translation_is_automorphism(full_py),
            "tanner_x_connected": _tanner_connected(hx),
            "tanner_z_connected": _tanner_connected(hz),
        }
    )
    return {
        "candidate": candidate.to_dict(),
        "accepted": all(checks.values()),
        "checks": checks,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
        "n": code.num_qubits,
        "k": code.dimension,
        "rate": code.dimension / code.num_qubits,
        "useful_rate": GRID_ORDER / code.num_qubits,
        "rank_x": code.code_x.rank,
        "rank_z": code.code_z.rank,
        "check_weights_x": sorted(set(map(int, row_weights_x))),
        "check_weights_z": sorted(set(map(int, row_weights_z))),
        "column_degrees_x": sorted(set(map(int, col_weights_x))),
        "column_degrees_z": sorted(set(map(int, col_weights_z))),
        "zx_fold_is_identity": False,
        "zx_fold_permutation": zx_fold.astype(int).tolist(),
        "zx_row_reflection": row_reflection.astype(int).tolist(),
    }
