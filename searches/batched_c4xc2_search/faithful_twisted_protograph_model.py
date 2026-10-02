"""Faithful `GL(2,2)` x-reflected GALA protographs at `n=64,96,...`."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import numpy as np
from qldpc import codes

from .faithful_twisted_halfswap_model import (
    TOP_LIFT_DIMENSION,
    TOP_ORDER,
    TOP_REPRESENTATION,
    _sample_faithful_twisted_support,
    faithful_commutant_g_terms,
    faithful_twisted_transpose_terms,
    faithful_x_reflection_permutation,
)
from .model import (
    GRID_ORDER,
    GRID_X_ORDER,
    GRID_Y_ORDER,
    Monomial,
    _permute_columns,
    _tanner_connected,
    bottom_support_generates,
    gf2_rank,
    lift_block_permutation,
    ring_polynomial,
    translation_permutations,
    zx_fold_permutation,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FaithfulTwistedProtographCandidate:
    """One-row GALA candidate with several blocks in each CSS half."""

    f_entries: tuple[tuple[Monomial, ...], ...]
    g_entries: tuple[tuple[Monomial, ...], ...]
    coupling_shift: int
    label: str = ""

    def __post_init__(self) -> None:
        if not self.f_entries or len(self.f_entries) != len(self.g_entries):
            raise ValueError("F and G must have the same positive entry count")
        if not 0 <= self.coupling_shift < len(self.f_entries):
            raise ValueError("coupling shift lies outside the protograph cycle")
        for name, entries in (("F", self.f_entries), ("G", self.g_entries)):
            for terms in entries:
                if not terms:
                    raise ValueError(f"{name} entries must be nonempty")
                if tuple(sorted(terms)) != terms or len(set(terms)) != len(terms):
                    raise ValueError(f"{name} supports must be sorted and distinct")
                if any(term.top >= TOP_ORDER for term in terms):
                    raise ValueError(f"{name} has an invalid GL(2,2) monomial")
                if faithful_twisted_transpose_terms(terms) != terms:
                    raise ValueError(
                        f"{name} entries must obey the faithful twisted transpose"
                    )

    @property
    def top_representation(self) -> str:
        return TOP_REPRESENTATION

    @property
    def entries(self) -> tuple[tuple[Monomial, ...], ...]:
        return self.f_entries + self.g_entries

    @property
    def half_blocks(self) -> int:
        return len(self.f_entries)

    @property
    def num_blocks(self) -> int:
        return 2 * self.half_blocks

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
        return sum(map(len, self.entries))

    @property
    def zx_fold_kind(self) -> str:
        return "twisted_protograph"

    @property
    def zx_internal_permutation(self) -> np.ndarray:
        return faithful_x_reflection_permutation()

    @property
    def candidate_id(self) -> str:
        payload = self.f_entries, self.g_entries, self.coupling_shift
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()
        return (
            f"c4xc2-s3-linear-l{self.num_blocks}-j1-xreflect-"
            f"w{self.nominal_check_weight}-{digest[:16]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "family": "faithful-twisted-protograph-centralizer",
            "top_representation": self.top_representation,
            "f_entries": [
                [asdict(term) for term in entry] for entry in self.f_entries
            ],
            "g_entries": [
                [asdict(term) for term in entry] for entry in self.g_entries
            ],
            "coupling_shift": self.coupling_shift,
            "label": self.label,
            "num_blocks": self.num_blocks,
            "num_qubits": self.num_qubits,
            "nominal_check_weight": self.nominal_check_weight,
            "zx_relation": (
                "H_Z = row_reflect_x(H_X data_reflect_x_reverse_halfswap)"
            ),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "FaithfulTwistedProtographCandidate":
        return cls(
            f_entries=tuple(
                tuple(sorted(Monomial(**term) for term in entry))
                for entry in data["f_entries"]
            ),
            g_entries=tuple(
                tuple(sorted(Monomial(**term) for term in entry))
                for entry in data["g_entries"]
            ),
            coupling_shift=int(data["coupling_shift"]),
            label=data.get("label", ""),
        )


def faithful_twisted_protograph_check_matrices(
    candidate: FaithfulTwistedProtographCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    ff = [
        np.asarray(ring_polynomial(TOP_REPRESENTATION, entry).lift(), dtype=np.uint8)
        for entry in candidate.f_entries
    ]
    gg = [
        np.asarray(ring_polynomial(TOP_REPRESENTATION, entry).lift(), dtype=np.uint8)
        for entry in candidate.g_entries
    ]
    size = candidate.half_blocks
    hx = np.hstack([*ff, *gg])
    hz = np.hstack(
        [
            *(gg[(-index) % size].T for index in range(size)),
            *(ff[(-index) % size].T for index in range(size)),
        ]
    )
    return hx, hz


def build_faithful_twisted_protograph_code(
    candidate: FaithfulTwistedProtographCandidate,
) -> codes.CSSCode:
    return codes.CSSCode(
        *faithful_twisted_protograph_check_matrices(candidate),
        promise_equal_distance_xz=True,
    )


def random_faithful_twisted_protograph_candidates(
    *,
    f_weights: Sequence[int],
    p_weights: Sequence[int],
    coupling_shift: int,
    check_weight_ceiling: int,
    count: int,
    seed: int,
) -> Iterator[FaithfulTwistedProtographCandidate]:
    """Sample `G_i = p_i I + q F_(i+a)` with a central monomial `q`."""
    if not f_weights or len(f_weights) != len(p_weights):
        raise ValueError("F and p profiles must have the same positive length")
    if any(weight < 1 for weight in f_weights) or any(
        weight < 0 for weight in p_weights
    ):
        raise ValueError("F weights are positive and p weights nonnegative")
    if not 0 <= coupling_shift < len(f_weights):
        raise ValueError("coupling shift lies outside the protograph cycle")
    if count < 1:
        raise ValueError("candidate count must be positive")
    bottom = tuple(
        (xx, yy)
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    )
    if any(weight > len(bottom) for weight in p_weights):
        raise ValueError("p-polynomial weight exceeds the bottom-group order")
    rng = random.Random(seed)
    seen: set[
        tuple[tuple[tuple[Monomial, ...], ...], tuple[tuple[Monomial, ...], ...]]
    ] = set()
    attempts = 0
    maximum_attempts = max(1000, 2000 * count)
    while len(seen) < count and attempts < maximum_attempts:
        attempts += 1
        ff = tuple(
            _sample_faithful_twisted_support(
                rng, int(weight), force_identity=index == 0
            )
            for index, weight in enumerate(f_weights)
        )
        q_support = (rng.choice(bottom),)
        pp = tuple(
            tuple(sorted(rng.sample(bottom, int(weight))))
            for weight in p_weights
        )
        gg = tuple(
            faithful_commutant_g_terms(
                ff[(index + coupling_shift) % len(ff)],
                p_support=pp[index],
                q_support=q_support,
            )
            for index in range(len(ff))
        )
        if any(not entry for entry in gg):
            continue
        if sum(map(len, (*ff, *gg))) > check_weight_ceiling:
            continue
        key = ff, gg
        if key in seen:
            continue
        seen.add(key)
        yield FaithfulTwistedProtographCandidate(
            ff,
            gg,
            coupling_shift=coupling_shift,
            label=(
                f"centralizer-f{','.join(map(str, f_weights))}-"
                f"p{','.join(map(str, p_weights))}-a{coupling_shift}"
            ),
        )


def analyze_faithful_twisted_protograph_structure(
    candidate: FaithfulTwistedProtographCandidate,
    *,
    check_weight_ceiling: int = 16,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "n_at_most_200": candidate.num_qubits <= 200,
        "nominal_check_weight_at_most_ceiling": (
            candidate.nominal_check_weight <= check_weight_ceiling
        ),
        "bottom_support_generates_C4xC2": bottom_support_generates(candidate),
    }
    hx, hz = faithful_twisted_protograph_check_matrices(candidate)
    checks["active_css_orthogonality"] = not bool(np.any((hx @ hz.T) % 2))
    if not checks["active_css_orthogonality"]:
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "checks": checks,
            "rejection_reasons": [
                name for name, passed in checks.items() if not passed
            ],
        }

    code = build_faithful_twisted_protograph_code(candidate)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    col_weights_x = np.count_nonzero(hx, axis=0)
    col_weights_z = np.count_nonzero(hz, axis=0)
    px, py = translation_permutations(TOP_REPRESENTATION)
    full_px = lift_block_permutation(px, candidate.num_blocks)
    full_py = lift_block_permutation(py, candidate.num_blocks)
    zx_fold = zx_fold_permutation(candidate)
    row_reflection = faithful_x_reflection_permutation()

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

    folded_hx = _permute_columns(hx, zx_fold)
    folded_hz = _permute_columns(hz, zx_fold)
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
            "strict_ZX_faithful_x_reflected_reverse_halfswap": bool(
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
        "zx_fold_permutation": zx_fold.astype(int).tolist(),
        "zx_row_reflection": row_reflection.astype(int).tolist(),
    }
