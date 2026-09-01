"""Independent-``F/G`` compact GALA models with an exact half-swap ZX fold.

The family fixes ``L=2`` and ``J=1`` over the natural permutation lift of
``S3 x C4 x C2``.  If the two independently searched group-ring elements
``F`` and ``G`` have transpose-invariant binary lifts, then

``H_X = [F | G]`` and ``H_Z = [G | F]``.

Consequently, transversal Hadamard followed by swapping the two 24-qubit
halves exchanges the two check matrices exactly.  CSS commutation is *not*
automatic: it is the active nonabelian constraint ``FG + GF = 0`` and is
checked explicitly for every candidate.
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


@dataclass(frozen=True)
class HalfSwapCandidate:
    """One inverse-closed, independent-``F/G`` compact GALA candidate."""

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
            if binary_transpose_terms(TOP_REPRESENTATION, terms) != terms:
                raise ValueError(f"{name} support must be binary-transpose invariant")

    @property
    def top_representation(self) -> str:
        return TOP_REPRESENTATION

    @property
    def entries(self) -> tuple[tuple[Monomial, ...], ...]:
        """Combined entries used by common bottom-generation diagnostics."""
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
        return "half_swap"

    @property
    def candidate_id(self) -> str:
        payload = (self.f_terms, self.g_terms)
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()
        return (
            "c4xc2-s3-natural-l2-j1-halfswap-"
            f"w{self.nominal_check_weight}-{digest[:16]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "family": "independent-inverse-closed-FG",
            "top_representation": self.top_representation,
            "f_terms": [asdict(term) for term in self.f_terms],
            "g_terms": [asdict(term) for term in self.g_terms],
            "label": self.label,
            "num_blocks": self.num_blocks,
            "num_qubits": self.num_qubits,
            "nominal_check_weight": self.nominal_check_weight,
            "zx_relation": "H_Z = half_swap(H_X)",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HalfSwapCandidate":
        return cls(
            f_terms=tuple(sorted(Monomial(**term) for term in data["f_terms"])),
            g_terms=tuple(sorted(Monomial(**term) for term in data["g_terms"])),
            label=data.get("label", ""),
        )


def build_half_swap_code(
    candidate: HalfSwapCandidate, *, skip_validation: bool = False
) -> codes.CSSCode:
    """Build the ``L=2,J=1`` GALA code for an independent ``F/G`` pair."""
    ff = ring_polynomial(TOP_REPRESENTATION, candidate.f_terms)
    gg = ring_polynomial(TOP_REPRESENTATION, candidate.g_terms)
    return codes.GALACode(
        [ff], [gg], num_active_rows=1, skip_validation=skip_validation
    )


@functools.lru_cache(maxsize=1)
def inverse_orbits() -> tuple[
    tuple[Monomial, ...], tuple[tuple[Monomial, Monomial], ...]
]:
    """Partition represented group monomials into transpose orbits."""
    universe = tuple(
        Monomial(top, xx, yy)
        for top in range(TOP_ORDER)
        for xx in range(4)
        for yy in range(2)
    )
    unseen = set(universe)
    fixed: list[Monomial] = []
    pairs: list[tuple[Monomial, Monomial]] = []
    while unseen:
        term = min(unseen)
        partner = binary_transpose_terms(TOP_REPRESENTATION, (term,))[0]
        unseen.remove(term)
        if partner == term:
            fixed.append(term)
        else:
            unseen.remove(partner)
            pairs.append(tuple(sorted((term, partner))))
    return tuple(sorted(fixed)), tuple(sorted(pairs))


def _sample_inverse_closed_support(
    rng: random.Random, weight: int, *, force_identity: bool
) -> tuple[Monomial, ...]:
    fixed, pairs = inverse_orbits()
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
        raise ValueError(
            f"cannot form inverse-closed support of weight {weight}"
            + (" containing identity" if force_identity else "")
        )
    pair_count = rng.choice(feasible_pair_counts)
    fixed_count = remaining - 2 * pair_count
    selected = [*required, *rng.sample(available_fixed, fixed_count)]
    for pair in rng.sample(pairs, pair_count):
        selected.extend(pair)
    return tuple(sorted(selected))


def random_half_swap_candidates(
    *, f_weight: int, g_weight: int, count: int, seed: int
) -> Iterator[HalfSwapCandidate]:
    """Sample normalized independent transpose-invariant pairs reproducibly."""
    if count < 0 or f_weight < 1 or g_weight < 1:
        raise ValueError("candidate counts and support weights must be positive")
    rng = random.Random(seed)
    seen: set[tuple[tuple[Monomial, ...], tuple[Monomial, ...]]] = set()
    attempts = 0
    maximum_attempts = max(100, 200 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        ff = _sample_inverse_closed_support(rng, f_weight, force_identity=True)
        gg = _sample_inverse_closed_support(rng, g_weight, force_identity=False)
        key = ff, gg
        if key in seen:
            continue
        seen.add(key)
        yield HalfSwapCandidate(ff, gg)


def analyze_half_swap_structure(
    candidate: HalfSwapCandidate, *, check_weight_ceiling: int = 16
) -> dict[str, Any]:
    """Run exact cheap gates for the independent-``F/G`` half-swap family."""
    checks: dict[str, bool] = {
        "n_at_most_200": candidate.num_qubits <= 200,
        "nominal_check_weight_at_most_ceiling": (
            candidate.nominal_check_weight <= check_weight_ceiling
        ),
        "F_binary_transpose_invariant": (
            binary_transpose_terms(TOP_REPRESENTATION, candidate.f_terms)
            == candidate.f_terms
        ),
        "G_binary_transpose_invariant": (
            binary_transpose_terms(TOP_REPRESENTATION, candidate.g_terms)
            == candidate.g_terms
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
        code = build_half_swap_code(candidate)
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
            "strict_ZX_half_swap": bool(
                np.array_equal(_permute_columns(hx, zx_fold), hz)
                and np.array_equal(_permute_columns(hz, zx_fold), hx)
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
    }
