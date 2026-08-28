"""Weight-12 active-GALA search over ``S_3 x C_8 x C_4``.

The four-grid pilot uses ``L=12`` physical block columns, ``J=3`` active
block rows, and six entries in each of the F and G block-circulants.  Taking
``G_j = F_{-j}.T`` makes the active binary check matrices exactly equal.  The
active row differences are ``0, +/-1, +/-2``; offset three remains latent and
is required to be nonorthogonal.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

import networkx as nx
import numpy as np

from qldpc import abstract, codes

from .s3_ising import (
    BOTTOM_ORDER,
    ProductMonomial,
    TOP_DEGREE,
    _bottom_subgroup_size,
    _gf2_rank,
    _lift_block_permutation,
    _permute_columns,
    _polynomial_product_support,
    _translation_permutations,
    _transpose_terms,
    has_odd_logical_support,
    ring_polynomial,
)

S3_MANY_COPY_SCHEMA_VERSION = 1
NUM_BLOCKS = 12
NUM_RING_ENTRIES = NUM_BLOCKS // 2
NUM_ACTIVE_ROWS = 3
CHECK_WEIGHT = 12
NUM_TARGET_GRIDS = 4
NUM_TARGET_LOGICALS = 32 * NUM_TARGET_GRIDS

CandidateVariant = Literal[
    "uniform_degree3",
    "even_a_002004",
    "even_b_002022",
    "even_c_011013",
    "even_d_011211",
    "even_e_020202",
]

UNIFORM_TERM_WEIGHTS = (1, 1, 1, 1, 1, 1)
EVEN_TERM_WEIGHT_PATTERNS: dict[CandidateVariant, tuple[int, ...]] = {
    "even_a_002004": (0, 0, 2, 0, 0, 4),
    "even_b_002022": (0, 0, 2, 0, 2, 2),
    "even_c_011013": (0, 1, 1, 0, 1, 3),
    "even_d_011211": (0, 1, 1, 2, 1, 1),
    "even_e_020202": (0, 2, 0, 2, 0, 2),
}


@dataclass(frozen=True, order=True)
class S3L12W12Candidate:
    """One exact-self-dual ``L=12, J=3`` candidate."""

    entries: tuple[tuple[ProductMonomial, ...], ...]
    variant: CandidateVariant

    def __post_init__(self) -> None:
        if len(self.entries) != NUM_RING_ENTRIES:
            raise ValueError(f"expected {NUM_RING_ENTRIES} F entries")
        expected = (
            UNIFORM_TERM_WEIGHTS
            if self.variant == "uniform_degree3"
            else EVEN_TERM_WEIGHT_PATTERNS[self.variant]
        )
        if tuple(map(len, self.entries)) != expected:
            raise ValueError(
                f"entry weights for {self.variant} must be {expected}"
            )
        for entry in self.entries:
            if len(set(entry)) != len(entry):
                raise ValueError("terms within each polynomial must be distinct")
            if tuple(sorted(entry)) != entry:
                raise ValueError("polynomial terms must be sorted")

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr((self.variant, self.entries)).encode()).hexdigest()[:16]
        return f"s3-l12-j3-w12-{self.variant}-{digest}"

    @property
    def num_qubits(self) -> int:
        return NUM_BLOCKS * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, ...]:
        def format_entry(entry: Sequence[ProductMonomial]) -> str:
            if not entry:
                return "0"
            labels = ("e", "tau_0", "tau_1", "sigma", "sigma^-1", "tau_2")
            terms: list[str] = []
            for term in entry:
                factors = [] if term.top == 0 else [labels[term.top]]
                if term.x:
                    factors.append("x" if term.x == 1 else f"x^{term.x}")
                if term.y:
                    factors.append("y" if term.y == 1 else f"y^{term.y}")
                terms.append(" ".join(factors) if factors else "1")
            return " + ".join(terms)

        return tuple(format_entry(entry) for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l12_j3_self_dual_weight12",
            "variant": self.variant,
            "entries": [
                [
                    {"top": term.top, "x": term.x, "y": term.y}
                    for term in entry
                ]
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, order=True)
class S3L12FoldW12Candidate:
    """Uniform-degree-three candidate with a translation-type ZX fold."""

    entries: tuple[tuple[ProductMonomial, ...], ...]

    def __post_init__(self) -> None:
        S3L12W12Candidate(self.entries, "uniform_degree3")

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l12-j3-w12-fold-{digest}"

    @property
    def num_qubits(self) -> int:
        return NUM_BLOCKS * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, ...]:
        return S3L12W12Candidate(
            self.entries, "uniform_degree3"
        ).polynomial_text

    def to_dict(self) -> dict[str, Any]:
        data = S3L12W12Candidate(
            self.entries, "uniform_degree3"
        ).to_dict()
        data["family"] = "s3_l12_j3_fold_weight12"
        return data


@dataclass(frozen=True, order=True)
class S3L12VertexFoldW12Candidate:
    """Uniform-degree-three candidate with a fixed-block ZX reflection."""

    entries: tuple[tuple[ProductMonomial, ...], ...]

    def __post_init__(self) -> None:
        S3L12W12Candidate(self.entries, "uniform_degree3")

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l12-j3-w12-vertex-fold-{digest}"

    @property
    def num_qubits(self) -> int:
        return NUM_BLOCKS * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, ...]:
        return S3L12W12Candidate(
            self.entries, "uniform_degree3"
        ).polynomial_text

    def to_dict(self) -> dict[str, Any]:
        data = S3L12W12Candidate(
            self.entries, "uniform_degree3"
        ).to_dict()
        data["family"] = "s3_l12_j3_vertex_fold_weight12"
        return data


FoldCandidate = S3L12FoldW12Candidate | S3L12VertexFoldW12Candidate


def _fold_shift(candidate: FoldCandidate) -> int:
    return 0 if isinstance(candidate, S3L12VertexFoldW12Candidate) else 3


def s3_l12_zx_fold_permutation(candidate: FoldCandidate) -> np.ndarray:
    """Physical data-qubit permutation implementing the candidate's ZX fold."""
    block_size = TOP_DEGREE * BOTTOM_ORDER
    shift = _fold_shift(candidate)
    reflection_axis = (NUM_ACTIVE_ROWS - 1 - shift) % NUM_RING_ENTRIES
    half_permutation = tuple(
        (reflection_axis - block) % NUM_RING_ENTRIES
        for block in range(NUM_RING_ENTRIES)
    )
    block_permutation = (
        *half_permutation,
        *(NUM_RING_ENTRIES + block for block in half_permutation),
    )
    return np.hstack(
        [block * block_size + np.arange(block_size) for block in block_permutation]
    )


def s3_l12_generators(
    candidate: S3L12W12Candidate,
) -> tuple[tuple[abstract.RingMember, ...], tuple[abstract.RingMember, ...]]:
    """Return ``F`` and ``G`` with ``G_j = F_{-j}.T``."""
    ff = tuple(ring_polynomial(entry) for entry in candidate.entries)
    gg = tuple(ff[(-index) % NUM_RING_ENTRIES].T for index in range(NUM_RING_ENTRIES))
    return ff, gg


def s3_l12_fold_generators(
    candidate: FoldCandidate,
) -> tuple[tuple[abstract.RingMember, ...], tuple[abstract.RingMember, ...]]:
    """Return generators with ``G_(i+s) = F_i.T`` for fold shift ``s``."""
    ff = tuple(ring_polynomial(entry) for entry in candidate.entries)
    gg_list: list[abstract.RingMember | None] = [None] * NUM_RING_ENTRIES
    shift = _fold_shift(candidate)
    for index, member in enumerate(ff):
        gg_list[(index + shift) % NUM_RING_ENTRIES] = member.T
    assert all(member is not None for member in gg_list)
    return ff, tuple(gg_list)  # type: ignore[arg-type,return-value]


def build_s3_l12_code(candidate: S3L12W12Candidate) -> codes.GALACode:
    ff, gg = s3_l12_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=NUM_ACTIVE_ROWS)


def build_s3_l12_fold_code(candidate: FoldCandidate) -> codes.GALACode:
    ff, gg = s3_l12_fold_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=NUM_ACTIVE_ROWS)


def _xor_product_supports(
    pairs: Sequence[
        tuple[Sequence[ProductMonomial], Sequence[ProductMonomial]]
    ],
) -> frozenset[ProductMonomial]:
    support: frozenset[ProductMonomial] = frozenset()
    for left, right in pairs:
        support ^= _polynomial_product_support(left, right)
    return support


def _row_correlation_support(
    candidate: S3L12W12Candidate, offset: int
) -> frozenset[ProductMonomial]:
    ff = candidate.entries
    gg = tuple(
        _transpose_terms(ff[(-index) % NUM_RING_ENTRIES])
        for index in range(NUM_RING_ENTRIES)
    )
    pairs: list[
        tuple[Sequence[ProductMonomial], Sequence[ProductMonomial]]
    ] = []
    for entries in (ff, gg):
        for index, entry in enumerate(entries):
            other = entries[(index - offset) % NUM_RING_ENTRIES]
            pairs.append((entry, _transpose_terms(other)))
    return _xor_product_supports(pairs)


def s3_l12_active_orthogonality_data(
    candidate: S3L12W12Candidate,
) -> dict[str, Any]:
    """Check active offsets and retain a nonzero offset-three latent row."""
    active_offsets = (0, 1, 2, 4, 5)
    active_sizes = {
        str(offset): len(_row_correlation_support(candidate, offset))
        for offset in active_offsets
    }
    latent_size = len(_row_correlation_support(candidate, 3))
    return {
        "active_offsets_zero": not any(active_sizes.values()),
        "latent_offset_3_nonzero": latent_size > 0,
        "active_support_sizes": active_sizes,
        "latent_support_size": latent_size,
    }


def _fold_row_correlation_support(
    candidate: FoldCandidate, offset: int
) -> frozenset[ProductMonomial]:
    ff = candidate.entries
    gg_list: list[tuple[ProductMonomial, ...] | None] = [None] * NUM_RING_ENTRIES
    shift = _fold_shift(candidate)
    for index, entry in enumerate(ff):
        gg_list[(index + shift) % NUM_RING_ENTRIES] = (
            _transpose_terms(entry)
        )
    assert all(entry is not None for entry in gg_list)
    gg = tuple(gg_list)  # type: ignore[arg-type]
    pairs = tuple(
        (ff[index], gg[(offset - index) % NUM_RING_ENTRIES])
        for index in range(NUM_RING_ENTRIES)
    )
    support: frozenset[ProductMonomial] = frozenset()
    for left, right in pairs:
        support ^= _polynomial_product_support(left, right)
        support ^= _polynomial_product_support(right, left)
    return support


def s3_l12_fold_active_orthogonality_data(
    candidate: FoldCandidate,
) -> dict[str, Any]:
    active_offsets = (0, 1, 2, 4, 5)
    active_sizes = {
        str(offset): len(_fold_row_correlation_support(candidate, offset))
        for offset in active_offsets
    }
    latent_size = len(_fold_row_correlation_support(candidate, 3))
    return {
        "active_offsets_zero": not any(active_sizes.values()),
        "latent_offset_3_nonzero": latent_size > 0,
        "active_support_sizes": active_sizes,
        "latent_support_size": latent_size,
    }


def _bottom_support_generates(candidate: S3L12W12Candidate) -> bool:
    points = [term.bottom for entry in candidate.entries for term in entry]
    return _bottom_subgroup_size(points) == BOTTOM_ORDER


def analyze_s3_l12_candidate(
    candidate: S3L12W12Candidate,
    *,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Apply algebraic, lifted, ZX, degree, rank, and translation filters."""
    started = time.perf_counter()
    orthogonality = s3_l12_active_orthogonality_data(candidate)
    cheap_checks = {
        "active_offsets_zero": bool(orthogonality["active_offsets_zero"]),
        "latent_offset_3_nonzero": bool(
            orthogonality["latent_offset_3_nonzero"]
        ),
        "bottom_support_generates_C8xC4": _bottom_support_generates(candidate),
    }
    if not all(cheap_checks.values()):
        return {
            "schema_version": S3_MANY_COPY_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "polynomials": candidate.polynomial_text,
            "accepted": False,
            "rejection_reasons": [
                name for name, passed in cheap_checks.items() if not passed
            ],
            "checks": cheap_checks,
            "orthogonality": orthogonality,
            "total_seconds": round(time.perf_counter() - started, 6),
        }

    code = build_s3_l12_code(candidate)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)
    expected_degrees = (
        {3} if candidate.variant == "uniform_degree3" else {2, 4}
    )
    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, NUM_BLOCKS)
    translation_y = _lift_block_permutation(perm_y, NUM_BLOCKS)
    rank_x = code.code_x.rank
    rank_z = code.code_z.rank

    def translation_is_automorphism(permutation: np.ndarray) -> bool:
        return bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, permutation)]), code.field)
            == rank_x
            and _gf2_rank(
                np.vstack([hz, _permute_columns(hz, permutation)]), code.field
            )
            == rank_z
        )

    checks = {
        **cheap_checks,
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "exact_ZX_self_dual": bool(np.array_equal(hx, hz)),
        "SWEL_odd_logical_exists": has_odd_logical_support(code),
        "uniform_check_weight_12": bool(
            np.all(row_weights_x == CHECK_WEIGHT)
            and np.all(row_weights_z == CHECK_WEIGHT)
        ),
        "doubly_even_stabilizer_generators": bool(
            np.all(row_weights_x % 4 == 0) and np.all(row_weights_z % 4 == 0)
        ),
        "column_degrees_match_variant": bool(
            set(column_weights_x.tolist()) == expected_degrees
            and set(column_weights_z.tolist()) == expected_degrees
        ),
        "dimension_can_hold_four_grids": code.dimension >= NUM_TARGET_LOGICALS,
        "x_translation_stabilizer_automorphism": translation_is_automorphism(
            translation_x
        ),
        "y_translation_stabilizer_automorphism": translation_is_automorphism(
            translation_y
        ),
    }
    if include_graph_metrics:
        checks["tanner_x_connected"] = bool(
            nx.is_connected(code.code_x.graph.to_undirected())
        )
        checks["tanner_z_connected"] = bool(
            nx.is_connected(code.code_z.graph.to_undirected())
        )
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": S3_MANY_COPY_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomials": candidate.polynomial_text,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "orthogonality": orthogonality,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight_x": int(row_weights_x.max(initial=0)),
        "check_weight_z": int(row_weights_z.max(initial=0)),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def analyze_s3_l12_fold_candidate(
    candidate: FoldCandidate,
    *,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Analyze the uniform-degree-three translation-ZX-fold family."""
    started = time.perf_counter()
    orthogonality = s3_l12_fold_active_orthogonality_data(candidate)
    cheap_checks = {
        "active_offsets_zero": bool(orthogonality["active_offsets_zero"]),
        "latent_offset_3_nonzero": bool(
            orthogonality["latent_offset_3_nonzero"]
        ),
        "bottom_support_generates_C8xC4": _bottom_support_generates(
            S3L12W12Candidate(candidate.entries, "uniform_degree3")
        ),
    }
    if not all(cheap_checks.values()):
        return {
            "schema_version": S3_MANY_COPY_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "polynomials": candidate.polynomial_text,
            "accepted": False,
            "rejection_reasons": [
                name for name, passed in cheap_checks.items() if not passed
            ],
            "checks": cheap_checks,
            "orthogonality": orthogonality,
            "total_seconds": round(time.perf_counter() - started, 6),
        }

    code = build_s3_l12_fold_code(candidate)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)
    block_size = TOP_DEGREE * BOTTOM_ORDER
    shift = _fold_shift(candidate)
    zx_fold = s3_l12_zx_fold_permutation(candidate)
    row_order = np.hstack(
        [
            block * block_size + np.arange(block_size)
            for block in range(NUM_ACTIVE_ROWS - 1, -1, -1)
        ]
    )
    folded_hx = _permute_columns(hx, zx_fold)
    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, NUM_BLOCKS)
    translation_y = _lift_block_permutation(perm_y, NUM_BLOCKS)
    rank_x = code.code_x.rank
    rank_z = code.code_z.rank

    def translation_is_automorphism(permutation: np.ndarray) -> bool:
        return bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, permutation)]), code.field)
            == rank_x
            and _gf2_rank(
                np.vstack([hz, _permute_columns(hz, permutation)]), code.field
            )
            == rank_z
        )

    checks = {
        **cheap_checks,
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "translation_ZX_fold": bool(np.array_equal(folded_hx[row_order], hz)),
        "uniform_check_weight_12": bool(
            np.all(row_weights_x == CHECK_WEIGHT)
            and np.all(row_weights_z == CHECK_WEIGHT)
        ),
        "uniform_column_degree_3": bool(
            np.all(column_weights_x == 3) and np.all(column_weights_z == 3)
        ),
        "doubly_even_stabilizer_generators": bool(
            np.all(row_weights_x % 4 == 0) and np.all(row_weights_z % 4 == 0)
        ),
        "dimension_can_hold_four_grids": code.dimension >= NUM_TARGET_LOGICALS,
        "x_translation_stabilizer_automorphism": translation_is_automorphism(
            translation_x
        ),
        "y_translation_stabilizer_automorphism": translation_is_automorphism(
            translation_y
        ),
    }
    if include_graph_metrics:
        checks["tanner_x_connected"] = bool(
            nx.is_connected(code.code_x.graph.to_undirected())
        )
        checks["tanner_z_connected"] = bool(
            nx.is_connected(code.code_z.graph.to_undirected())
        )
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": S3_MANY_COPY_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomials": candidate.polynomial_text,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "orthogonality": orthogonality,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight_x": int(row_weights_x.max(initial=0)),
        "check_weight_z": int(row_weights_z.max(initial=0)),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "even_syndrome_parity": False,
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def random_s3_l12_candidates(
    num_samples: int,
    *,
    variant: CandidateVariant,
    seed: int = 0,
) -> Iterator[S3L12W12Candidate]:
    """Sample normalized candidates with reproducible random monomials."""
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    rng = random.Random(seed)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top, xx, yy in itertools.product(range(6), range(8), range(4))
    ]
    anchor = ProductMonomial(0, 0, 0)
    weights = (
        UNIFORM_TERM_WEIGHTS
        if variant == "uniform_degree3"
        else EVEN_TERM_WEIGHT_PATTERNS[variant]
    )
    anchor_index = next(index for index, weight in enumerate(weights) if weight)
    seen: set[S3L12W12Candidate] = set()
    attempts = 0
    max_attempts = max(100, 20 * num_samples)
    while len(seen) < num_samples and attempts < max_attempts:
        attempts += 1
        entries: list[tuple[ProductMonomial, ...]] = []
        for index, weight in enumerate(weights):
            if not weight:
                entries.append(())
            elif index == anchor_index:
                extras = rng.sample(
                    [term for term in all_terms if term != anchor], weight - 1
                )
                entries.append(tuple(sorted((anchor, *extras))))
            else:
                entries.append(tuple(sorted(rng.sample(all_terms, weight))))
        candidate = S3L12W12Candidate(tuple(entries), variant)
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def random_s3_l12_fold_candidates(
    num_samples: int, *, seed: int = 0, vertex_reflection: bool = False
) -> Iterator[FoldCandidate]:
    for candidate in random_s3_l12_candidates(
        num_samples, variant="uniform_degree3", seed=seed
    ):
        yield (
            S3L12VertexFoldW12Candidate(candidate.entries)
            if vertex_reflection
            else S3L12FoldW12Candidate(candidate.entries)
        )
