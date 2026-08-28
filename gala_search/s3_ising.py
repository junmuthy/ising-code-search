"""S3 polynomial GALA searches tailored to the packed Ising architecture.

The bottom group ``C_8 x C_4`` is the logical translation group.  The top
``S_3`` uses its natural degree-three permutation representation.  The first
pilot family has ``L=4`` and ``J=1`` with term weights ``(2, 4)`` in the two
F entries.  Its G entries are fixed by the nontrivial translation-type ZX
fold, ``G_1 = F_0.T`` and ``G_0 = F_1.T``.

The module separates cheap group-algebra filters from lifted binary-matrix
analysis.  In particular, active CSS orthogonality is ``Psi_0 = 0`` while
``Psi_1 != 0`` witnesses genuinely latent (active-orthogonality) structure.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np

from qldpc import abstract, codes

S3_ISING_SCHEMA_VERSION = 1
BOTTOM_X_ORDER = 8
BOTTOM_Y_ORDER = 4
BOTTOM_ORDER = BOTTOM_X_ORDER * BOTTOM_Y_ORDER
TOP_DEGREE = 3


@dataclass(frozen=True, order=True)
class ProductMonomial:
    """One monomial ``h x^u y^v`` in ``F_2[S_3 x C_8 x C_4]``."""

    top: int
    x: int
    y: int

    def __post_init__(self) -> None:
        if not 0 <= self.top < 6:
            raise ValueError("top index must lie in range(6)")
        if not 0 <= self.x < BOTTOM_X_ORDER:
            raise ValueError("x exponent must lie in range(8)")
        if not 0 <= self.y < BOTTOM_Y_ORDER:
            raise ValueError("y exponent must lie in range(4)")

    @property
    def bottom(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True, order=True)
class S3L4Candidate:
    """An ``L=4, J=1`` weight-12 polynomial GALA candidate."""

    f0: tuple[ProductMonomial, ProductMonomial]
    f1: tuple[ProductMonomial, ProductMonomial, ProductMonomial, ProductMonomial]

    def __post_init__(self) -> None:
        if len(set(self.f0)) != 2 or len(set(self.f1)) != 4:
            raise ValueError("terms within each polynomial must be distinct")
        if tuple(sorted(self.f0)) != self.f0 or tuple(sorted(self.f1)) != self.f1:
            raise ValueError("polynomial terms must be sorted")

    @property
    def candidate_id(self) -> str:
        payload = repr((self.f0, self.f1)).encode()
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"s3-l4-j1-w12-{digest}"

    @property
    def num_blocks(self) -> int:
        return 4

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, str]:
        return _format_polynomial(self.f0), _format_polynomial(self.f1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l4_j1_weight12",
            "f0": [asdict(term) for term in self.f0],
            "f1": [asdict(term) for term in self.f1],
        }


@dataclass(frozen=True, order=True)
class S3L4W16Candidate:
    """An ``L=4, J=1`` candidate with two weight-four F entries."""

    f0: tuple[
        ProductMonomial, ProductMonomial, ProductMonomial, ProductMonomial
    ]
    f1: tuple[
        ProductMonomial, ProductMonomial, ProductMonomial, ProductMonomial
    ]

    def __post_init__(self) -> None:
        for entry in self.entries:
            if len(set(entry)) != 4:
                raise ValueError("each L=4 weight-16 F entry must have four terms")
            if tuple(sorted(entry)) != entry:
                raise ValueError("polynomial terms must be sorted")

    @property
    def entries(self) -> tuple[
        tuple[ProductMonomial, ...], tuple[ProductMonomial, ...]
    ]:
        return self.f0, self.f1

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l4-j1-w16-{digest}"

    @property
    def num_blocks(self) -> int:
        return 4

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, str]:
        return _format_polynomial(self.f0), _format_polynomial(self.f1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l4_j1_weight16",
            "f0": [asdict(term) for term in self.f0],
            "f1": [asdict(term) for term in self.f1],
        }


class S3L4SelfDualW16Candidate(S3L4W16Candidate):
    """An ``L=4, J=1`` weight-16 candidate with ``G_i = F_i.T``.

    This generator relation makes the two active CSS check matrices identical.
    It is distinct from :class:`S3L4W16Candidate`'s translation-type fold,
    which instead reverses the two G entries.
    """

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l4-j1-self-dual-w16-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l4_j1_self_dual_weight16",
            "f0": [asdict(term) for term in self.f0],
            "f1": [asdict(term) for term in self.f1],
        }


@dataclass(frozen=True, order=True)
class S3L8Candidate:
    """An ``L=8, J=2`` weight-12 polynomial GALA candidate.

    The four F-entry term weights are ``(1, 1, 1, 3)``.  The G sequence is
    fixed by the translation-type ZX relation ``G_(i+2) = F_i.T``.
    """

    f0: tuple[ProductMonomial]
    f1: tuple[ProductMonomial]
    f2: tuple[ProductMonomial]
    f3: tuple[ProductMonomial, ProductMonomial, ProductMonomial]

    def __post_init__(self) -> None:
        entries = (self.f0, self.f1, self.f2, self.f3)
        if tuple(map(len, entries)) != (1, 1, 1, 3):
            raise ValueError("L=8 entry weights must be (1, 1, 1, 3)")
        if any(tuple(sorted(entry)) != entry for entry in entries):
            raise ValueError("polynomial terms must be sorted")
        if len(set(self.f3)) != 3:
            raise ValueError("the trinomial terms must be distinct")

    @property
    def entries(self) -> tuple[tuple[ProductMonomial, ...], ...]:
        return self.f0, self.f1, self.f2, self.f3

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l8-j2-w12-{digest}"

    @property
    def num_blocks(self) -> int:
        return 8

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, str, str, str]:
        return tuple(_format_polynomial(entry) for entry in self.entries)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l8_j2_weight12",
            **{
                f"f{index}": [asdict(term) for term in entry]
                for index, entry in enumerate(self.entries)
            },
        }


@dataclass(frozen=True, order=True)
class S3L8W16Candidate:
    """An ``L=8, J=2`` candidate with four binomial F entries.

    All eight data-block columns consequently have degree four in each CSS
    Tanner graph.  This removes the degree-two columns responsible for the
    systematic weight-two and weight-four logicals in the weight-12 pilot.
    """

    f0: tuple[ProductMonomial, ProductMonomial]
    f1: tuple[ProductMonomial, ProductMonomial]
    f2: tuple[ProductMonomial, ProductMonomial]
    f3: tuple[ProductMonomial, ProductMonomial]

    def __post_init__(self) -> None:
        for entry in self.entries:
            if len(set(entry)) != 2:
                raise ValueError("each weight-16 F entry must be a binomial")
            if tuple(sorted(entry)) != entry:
                raise ValueError("polynomial terms must be sorted")

    @property
    def entries(self) -> tuple[tuple[ProductMonomial, ProductMonomial], ...]:
        return self.f0, self.f1, self.f2, self.f3

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha256(repr(self.entries).encode()).hexdigest()[:16]
        return f"s3-l8-j2-w16-{digest}"

    @property
    def num_blocks(self) -> int:
        return 8

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * TOP_DEGREE * BOTTOM_ORDER

    @property
    def polynomial_text(self) -> tuple[str, str, str, str]:
        return tuple(_format_polynomial(entry) for entry in self.entries)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "s3_l8_j2_weight16",
            **{
                f"f{index}": [asdict(term) for term in entry]
                for index, entry in enumerate(self.entries)
            },
        }


def _top_label(index: int) -> str:
    return ("e", "tau_0", "tau_1", "sigma", "sigma^-1", "tau_2")[index]


def _format_polynomial(terms: Sequence[ProductMonomial]) -> str:
    output: list[str] = []
    for term in terms:
        factors = [] if term.top == 0 else [_top_label(term.top)]
        if term.x:
            factors.append("x" if term.x == 1 else f"x^{term.x}")
        if term.y:
            factors.append("y" if term.y == 1 else f"y^{term.y}")
        output.append(" ".join(factors) if factors else "1")
    return " + ".join(output)


@functools.lru_cache(maxsize=1)
def _group_data() -> tuple[
    abstract.GroupRing,
    tuple[abstract.GroupMember, ...],
    abstract.GroupMember,
    abstract.GroupMember,
]:
    top = abstract.SymmetricGroup(3).with_natural_lift()
    bottom = abstract.AbelianGroup(BOTTOM_X_ORDER, BOTTOM_Y_ORDER)
    product = abstract.Group.tensor_product(top, bottom)
    ring = abstract.GroupRing(product)
    top_members = tuple(sorted(top.generate()))
    xx, yy = bottom.generators
    return ring, top_members, xx, yy


@functools.lru_cache(maxsize=None)
def ring_monomial(term: ProductMonomial) -> abstract.RingMember:
    ring, top_members, xx, yy = _group_data()
    product_member = top_members[term.top] @ (xx**term.x * yy**term.y)
    return abstract.RingMember(ring, product_member)


@functools.lru_cache(maxsize=None)
def ring_polynomial(terms: tuple[ProductMonomial, ...]) -> abstract.RingMember:
    ring, _top_members, _xx, _yy = _group_data()
    return functools.reduce(
        lambda left, right: left + right,
        (ring_monomial(term) for term in terms),
        ring.zero,
    )


def candidate_generators(
    candidate: S3L4Candidate | S3L4W16Candidate,
) -> tuple[
    tuple[abstract.RingMember, abstract.RingMember],
    tuple[abstract.RingMember, abstract.RingMember],
]:
    """Return F and the nontrivial translation-ZX partner G.

    With two entries per half, ``r_1`` exchanges indices zero and one.  Taking
    ``G_1 = F_0.T`` and ``G_0 = F_1.T`` gives the sector-fold relation used by
    the search.
    """
    ff = (ring_polynomial(candidate.f0), ring_polynomial(candidate.f1))
    gg = (ff[1].T, ff[0].T)
    return ff, gg


def self_dual_candidate_generators(
    candidate: S3L4SelfDualW16Candidate,
) -> tuple[
    tuple[abstract.RingMember, abstract.RingMember],
    tuple[abstract.RingMember, abstract.RingMember],
]:
    """Return generators satisfying ``G_i = F_i.T``.

    For the active ``J=1`` row this gives
    ``H_X = H_Z = [F_0, F_1, F_0.T, F_1.T]``.
    """
    ff = (ring_polynomial(candidate.f0), ring_polynomial(candidate.f1))
    return ff, (ff[0].T, ff[1].T)


def l8_candidate_generators(
    candidate: S3L8Candidate | S3L8W16Candidate,
) -> tuple[tuple[abstract.RingMember, ...], tuple[abstract.RingMember, ...]]:
    ff = tuple(ring_polynomial(entry) for entry in candidate.entries)
    gg_list: list[abstract.RingMember | None] = [None] * 4
    for index, member in enumerate(ff):
        gg_list[(index + 2) % 4] = member.T
    assert all(member is not None for member in gg_list)
    return ff, tuple(gg_list)  # type: ignore[arg-type,return-value]


@functools.lru_cache(maxsize=1)
def _top_multiplication_data() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Multiplication and inverse tables for the sorted ``S_3`` elements."""
    _ring, top_members, _xx, _yy = _group_data()
    indices = {member: index for index, member in enumerate(top_members)}
    multiplication = tuple(
        tuple(indices[left * right] for right in top_members)
        for left in top_members
    )
    inverses = tuple(
        next(
            right
            for right in range(6)
            if multiplication[left][right] == 0
            and multiplication[right][left] == 0
        )
        for left in range(6)
    )
    return multiplication, inverses


def _transpose_terms(
    terms: Sequence[ProductMonomial],
) -> tuple[ProductMonomial, ...]:
    _multiplication, inverses = _top_multiplication_data()
    return tuple(
        sorted(
            ProductMonomial(
                inverses[term.top],
                (-term.x) % BOTTOM_X_ORDER,
                (-term.y) % BOTTOM_Y_ORDER,
            )
            for term in terms
        )
    )


def _polynomial_product_support(
    left: Sequence[ProductMonomial], right: Sequence[ProductMonomial]
) -> frozenset[ProductMonomial]:
    """Support of a polynomial product over GF(2), including cancellations."""
    multiplication, _inverses = _top_multiplication_data()
    support: set[ProductMonomial] = set()
    for left_term in left:
        for right_term in right:
            product = ProductMonomial(
                multiplication[left_term.top][right_term.top],
                (left_term.x + right_term.x) % BOTTOM_X_ORDER,
                (left_term.y + right_term.y) % BOTTOM_Y_ORDER,
            )
            if product in support:
                support.remove(product)
            else:
                support.add(product)
    return frozenset(support)


def _commutator_support(
    left: Sequence[ProductMonomial], right: Sequence[ProductMonomial]
) -> frozenset[ProductMonomial]:
    return _polynomial_product_support(left, right) ^ _polynomial_product_support(
        right, left
    )


def _sum_commutator_supports(
    pairs: Sequence[
        tuple[Sequence[ProductMonomial], Sequence[ProductMonomial]]
    ],
) -> frozenset[ProductMonomial]:
    support: frozenset[ProductMonomial] = frozenset()
    for left, right in pairs:
        support ^= _commutator_support(left, right)
    return support


def active_orthogonality_data(
    candidate: S3L4Candidate | S3L4W16Candidate,
) -> dict[str, bool]:
    """Evaluate the active and latent aggregate commutators in the group ring."""
    ff = (candidate.f0, candidate.f1)
    gg = (_transpose_terms(ff[1]), _transpose_terms(ff[0]))
    psi_active = _sum_commutator_supports(((ff[0], gg[0]), (ff[1], gg[1])))
    psi_latent = _sum_commutator_supports(((ff[0], gg[1]), (ff[1], gg[0])))
    return {
        "active_psi_zero": not bool(psi_active),
        "latent_psi_nonzero": bool(psi_latent),
    }


def self_dual_active_orthogonality_data(
    candidate: S3L4SelfDualW16Candidate,
) -> dict[str, bool]:
    """Evaluate active and latent commutators for ``G_i = F_i.T``."""
    ff = (candidate.f0, candidate.f1)
    daggers = (_transpose_terms(ff[0]), _transpose_terms(ff[1]))
    psi_active = _sum_commutator_supports(
        ((ff[0], daggers[0]), (ff[1], daggers[1]))
    )
    psi_latent = _sum_commutator_supports(
        ((ff[0], daggers[1]), (ff[1], daggers[0]))
    )
    return {
        "active_psi_zero": not bool(psi_active),
        "latent_psi_nonzero": bool(psi_latent),
    }


def l8_active_orthogonality_data(
    candidate: S3L8Candidate | S3L8W16Candidate,
) -> dict[str, bool]:
    """Require active offsets 0, +/-1 to vanish and latent offset 2 to survive."""
    ff = candidate.entries
    gg_list: list[tuple[ProductMonomial, ...] | None] = [None] * 4
    for index, entry in enumerate(ff):
        gg_list[(index + 2) % 4] = _transpose_terms(entry)
    assert all(entry is not None for entry in gg_list)
    gg = tuple(gg_list)  # type: ignore[arg-type]
    psi: list[frozenset[ProductMonomial]] = []
    for offset in range(4):
        psi.append(
            _sum_commutator_supports(
                tuple(
                    (ff[index], gg[(offset - index) % 4])
                    for index in range(4)
                )
            )
        )
    return {
        "active_psi_0_zero": not bool(psi[0]),
        "active_psi_1_zero": not bool(psi[1]),
        "active_psi_3_zero": not bool(psi[3]),
        "latent_psi_2_nonzero": bool(psi[2]),
    }


def build_s3_l4_code(candidate: S3L4Candidate) -> codes.GALACode:
    ff, gg = candidate_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=1)


def build_s3_l4_w16_code(candidate: S3L4W16Candidate) -> codes.GALACode:
    ff, gg = candidate_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=1)


def build_s3_l4_self_dual_w16_code(
    candidate: S3L4SelfDualW16Candidate,
) -> codes.GALACode:
    ff, gg = self_dual_candidate_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=1)


def build_s3_l8_code(candidate: S3L8Candidate) -> codes.GALACode:
    ff, gg = l8_candidate_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=2)


def build_s3_l8_w16_code(candidate: S3L8W16Candidate) -> codes.GALACode:
    ff, gg = l8_candidate_generators(candidate)
    return codes.GALACode(ff, gg, num_active_rows=2)


def _bottom_subgroup_size(points: Sequence[tuple[int, int]]) -> int:
    """Size of the subgroup generated by all point differences."""
    if not points:
        return 1
    anchor = points[0]
    generators = {
        ((xx - anchor[0]) % BOTTOM_X_ORDER, (yy - anchor[1]) % BOTTOM_Y_ORDER)
        for xx, yy in points[1:]
    }
    reached = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            for sign in (-1, 1):
                value = (
                    (current[0] + sign * generator[0]) % BOTTOM_X_ORDER,
                    (current[1] + sign * generator[1]) % BOTTOM_Y_ORDER,
                )
                if value not in reached:
                    reached.add(value)
                    frontier.append(value)
    return len(reached)


def bottom_support_generates(
    candidate: S3L4Candidate | S3L4W16Candidate,
) -> bool:
    points = [term.bottom for term in itertools.chain(candidate.f0, candidate.f1)]
    return _bottom_subgroup_size(points) == BOTTOM_ORDER


def l8_bottom_support_generates(
    candidate: S3L8Candidate | S3L8W16Candidate,
) -> bool:
    points = [term.bottom for entry in candidate.entries for term in entry]
    return _bottom_subgroup_size(points) == BOTTOM_ORDER


def _gf2_rank(matrix: np.ndarray[Any, Any], field: type[np.ndarray[Any, Any]]) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=int).view(field)))


def has_odd_logical_support(code: codes.CSSCode) -> bool:
    """Whether ``ker(H_X)`` contains an odd-weight vector modulo Z checks.

    For the self-dual codes used here, every vector in ``ker(H_X)`` is a
    Z logical or stabilizer and all stabilizers have even weight.  An odd
    vector exists exactly when the all-ones row is absent from ``row(H_X)``.
    """
    matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
    ones = np.ones((1, code.num_qubits), dtype=np.uint8)
    return _gf2_rank(np.vstack([matrix_x, ones]), code.field) > code.code_x.rank


def _translation_permutations() -> tuple[np.ndarray, np.ndarray]:
    """Physical permutations for the two bottom generators on one lift block."""
    ring, top_members, xx, yy = _group_data()
    top_identity = top_members[0]
    matrices = [
        np.asarray(ring.group.lift(top_identity @ bottom_member), dtype=int)
        for bottom_member in (xx, yy)
    ]
    return tuple(np.argmax(matrix, axis=1).astype(int) for matrix in matrices)  # type: ignore[return-value]


def _lift_block_permutation(permutation: np.ndarray, num_blocks: int) -> np.ndarray:
    block_size = TOP_DEGREE * BOTTOM_ORDER
    return np.hstack(
        [block * block_size + permutation for block in range(num_blocks)]
    )


def _permute_vector(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(vector)
    transformed[permutation] = vector
    return transformed


def _permute_columns(matrix: np.ndarray[Any, Any], permutation: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(matrix)
    transformed[:, permutation] = matrix
    return transformed


def _zx_fold_permutation(num_blocks: int) -> np.ndarray:
    block_size = TOP_DEGREE * BOTTOM_ORDER
    if num_blocks < 2 or num_blocks % 2:
        raise ValueError("the ZX fold requires a positive even number of blocks")
    half = num_blocks // 2
    block_order = (*range(half - 1, -1, -1), *range(num_blocks - 1, half - 1, -1))
    return np.hstack(
        [block * block_size + np.arange(block_size) for block in block_order]
    )


@functools.lru_cache(maxsize=None)
def s3_internal_translation_orbits(num_blocks: int = 8) -> tuple[tuple[int, ...], ...]:
    """Partition physical qubits into bottom-translation orbits.

    For ``L=8`` there are ``8 * 3 = 24`` internal fibres, each containing the
    32 sites of ``C_8 x C_4``.  Computing the partition from the actual lift
    permutations avoids relying on a particular tensor-product index order.
    """
    perm_x, perm_y = _translation_permutations()
    full_x = _lift_block_permutation(perm_x, num_blocks)
    full_y = _lift_block_permutation(perm_y, num_blocks)
    num_qubits = num_blocks * TOP_DEGREE * BOTTOM_ORDER
    remaining = set(range(num_qubits))
    orbits: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        orbit = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for permutation in (full_x, full_y):
                translated = int(permutation[current])
                if translated not in orbit:
                    orbit.add(translated)
                    frontier.append(translated)
        remaining -= orbit
        orbits.append(tuple(sorted(orbit)))
    expected = num_blocks * TOP_DEGREE
    if len(orbits) != expected or any(len(orbit) != BOTTOM_ORDER for orbit in orbits):
        raise RuntimeError("bottom translations did not produce the expected fibres")
    return tuple(orbits)


def logical_translation_orbit(seed: np.ndarray, *, num_blocks: int = 8) -> np.ndarray:
    """Return the 32 rows obtained by translating a seed over ``C_8 x C_4``."""
    seed = np.asarray(seed, dtype=np.uint8)
    perm_x, perm_y = _translation_permutations()
    full_x = _lift_block_permutation(perm_x, num_blocks)
    full_y = _lift_block_permutation(perm_y, num_blocks)
    rows: list[np.ndarray] = []
    translated_x = seed.copy()
    for _xx in range(BOTTOM_X_ORDER):
        translated_y = translated_x.copy()
        for _yy in range(BOTTOM_Y_ORDER):
            rows.append(translated_y)
            translated_y = _permute_vector(translated_y, full_y)
        translated_x = _permute_vector(translated_x, full_x)
    return np.asarray(rows, dtype=np.uint8)


def _canonical_bottom_translation_support(
    support: Sequence[int], *, num_qubits: int, num_blocks: int
) -> tuple[int, ...]:
    seed = np.zeros(num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in logical_translation_orbit(seed, num_blocks=num_blocks)
    )


def analyze_translation_logical_seed(
    code: codes.CSSCode,
    support: Sequence[int],
    *,
    zx_fold_permutation: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Certify the disjoint translated orbit and its ZX-folded X partners."""
    block_size = TOP_DEGREE * BOTTOM_ORDER
    if code.num_qubits % block_size:
        raise ValueError("code length is not compatible with the S3 bottom lift")
    num_blocks = code.num_qubits // block_size
    seed = np.zeros(code.num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    internal_orbits = s3_internal_translation_orbits(num_blocks)
    occupancies = np.asarray(
        [np.count_nonzero(seed[list(orbit)]) for orbit in internal_orbits], dtype=int
    )
    used = np.flatnonzero(occupancies)
    z_orbit = logical_translation_orbit(seed, num_blocks=num_blocks)
    fold = np.asarray(
        _zx_fold_permutation(num_blocks)
        if zx_fold_permutation is None
        else zx_fold_permutation,
        dtype=int,
    )
    if sorted(fold.tolist()) != list(range(code.num_qubits)):
        raise ValueError("the ZX fold must be a permutation of the physical qubits")
    x_orbit = _permute_columns(z_orbit, fold)
    check_x = np.asarray(code.matrix_x, dtype=np.uint8)
    check_z = np.asarray(code.matrix_z, dtype=np.uint8)
    rank_z = code.code_z.rank
    orbit_rank_mod_stabilizers = (
        _gf2_rank(np.vstack([check_z, z_orbit]), code.field) - rank_z
    )
    pairing = (z_orbit @ x_orbit.T) % 2
    return {
        "weight": int(np.count_nonzero(seed)),
        "seed_support": np.flatnonzero(seed).astype(int).tolist(),
        "internal_orbits": used.astype(int).tolist(),
        "internal_occupancies": occupancies.astype(int).tolist(),
        "graph_supported": bool(np.all(occupancies <= 1)),
        "translation_orbit_size": int(z_orbit.shape[0]),
        "translated_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in z_orbit
        ],
        "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
        "z_orbit_in_kernel": bool(not np.any((check_x @ z_orbit.T) % 2)),
        "x_partner_orbit_in_kernel": bool(not np.any((check_z @ x_orbit.T) % 2)),
        "orbit_rank_mod_stabilizers": int(orbit_rank_mod_stabilizers),
        "zx_pairing_rank": _gf2_rank(pairing, code.field),
    }


def analyze_self_dual_translation_logical_seed(
    code: codes.CSSCode, support: Sequence[int]
) -> dict[str, Any]:
    """Certify one clean grid under identity transversal Hadamard."""
    result = analyze_translation_logical_seed(
        code,
        support,
        zx_fold_permutation=np.arange(code.num_qubits),
    )
    seed = np.zeros(code.num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    num_blocks = code.num_qubits // (TOP_DEGREE * BOTTOM_ORDER)
    orbit = logical_translation_orbit(seed, num_blocks=num_blocks)
    pairing = (orbit @ orbit.T) % 2
    grid = orbit.reshape(BOTTOM_X_ORDER, BOTTOM_Y_ORDER, code.num_qubits)
    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, num_blocks)
    translation_y = _lift_block_permutation(perm_y, num_blocks)
    shifted_x = _permute_columns(orbit, translation_x).reshape(grid.shape)
    shifted_y = _permute_columns(orbit, translation_y).reshape(grid.shape)
    result.update(
        {
            "identity_zx_fold": bool(
                np.array_equal(code.matrix_x, code.matrix_z)
            ),
            "pairing_is_identity": bool(
                np.array_equal(pairing, np.eye(BOTTOM_ORDER, dtype=np.uint8))
            ),
            "x_translation_is_C8_grid_shift": bool(
                np.array_equal(shifted_x, np.roll(grid, -1, axis=0))
            ),
            "y_translation_is_C4_grid_shift": bool(
                np.array_equal(shifted_y, np.roll(grid, -1, axis=1))
            ),
            "logical_support_union_weight": int(np.count_nonzero(np.any(orbit, axis=0))),
            "unused_physical_qubits": int(
                code.num_qubits - np.count_nonzero(np.any(orbit, axis=0))
            ),
        }
    )
    return result


def analyze_translation_logical_seeds(
    code: codes.CSSCode, supports: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Certify several disjoint 32-site sectors and their combined ZX pairing."""
    sector_results = [analyze_translation_logical_seed(code, support) for support in supports]
    seeds = []
    for support in supports:
        seed = np.zeros(code.num_qubits, dtype=np.uint8)
        seed[list(support)] = 1
        seeds.append(seed)
    num_blocks = code.num_qubits // (TOP_DEGREE * BOTTOM_ORDER)
    z_logicals = np.vstack(
        [logical_translation_orbit(seed, num_blocks=num_blocks) for seed in seeds]
    )
    x_logicals = _permute_columns(z_logicals, _zx_fold_permutation(num_blocks))
    stabilizer_z = np.asarray(code.matrix_z, dtype=np.uint8)
    pairing = (z_logicals @ x_logicals.T) % 2
    return {
        "num_sectors": len(supports),
        "num_logicals": int(z_logicals.shape[0]),
        "sector_results": sector_results,
        "pairwise_disjoint": bool(np.all(np.sum(z_logicals, axis=0) <= 1)),
        "combined_rank_mod_stabilizers": int(
            _gf2_rank(np.vstack([stabilizer_z, z_logicals]), code.field)
            - code.code_z.rank
        ),
        "combined_zx_pairing_rank": _gf2_rank(pairing, code.field),
    }


def _analyze_s3_l4_candidate(
    candidate: S3L4Candidate | S3L4W16Candidate,
    *,
    expected_check_weight: int,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Run the algebraic, ZX, LDPC, and translation acceptance checks."""
    started = time.perf_counter()
    cheap_checks = {
        **active_orthogonality_data(candidate),
        "bottom_support_generates_C8xC4": bottom_support_generates(candidate),
    }
    if not all(cheap_checks.values()):
        rejection_reasons = [name for name, passed in cheap_checks.items() if not passed]
        return {
            "schema_version": S3_ISING_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "polynomials": candidate.polynomial_text,
            "accepted": False,
            "rejection_reasons": rejection_reasons,
            "checks": cheap_checks,
            "total_seconds": round(time.perf_counter() - started, 6),
        }

    ff, gg = candidate_generators(candidate)
    code = codes.GALACode(ff, gg, num_active_rows=1)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)

    # For r_1 at L=4, the fold swaps the two block columns within each half.
    block_size = TOP_DEGREE * BOTTOM_ORDER
    zx_fold = np.hstack(
        [
            block_size + np.arange(block_size),
            np.arange(block_size),
            3 * block_size + np.arange(block_size),
            2 * block_size + np.arange(block_size),
        ]
    )
    folded_hx = _permute_columns(hx, zx_fold)

    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, candidate.num_blocks)
    translation_y = _lift_block_permutation(perm_y, candidate.num_blocks)
    rank_x = code.code_x.rank
    rank_z = code.code_z.rank

    checks = {
        **cheap_checks,
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "translation_ZX_fold": bool(np.array_equal(folded_hx, hz)),
        f"uniform_check_weight_{expected_check_weight}": bool(
            np.all(row_weights_x == expected_check_weight)
            and np.all(row_weights_z == expected_check_weight)
        ),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "rate_at_least_half": code.dimension * 2 >= code.num_qubits,
        "x_translation_stabilizer_automorphism": bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, translation_x)]), code.field)
            == rank_x
            and _gf2_rank(np.vstack([hz, _permute_columns(hz, translation_x)]), code.field)
            == rank_z
        ),
        "y_translation_stabilizer_automorphism": bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, translation_y)]), code.field)
            == rank_x
            and _gf2_rank(np.vstack([hz, _permute_columns(hz, translation_y)]), code.field)
            == rank_z
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
        "schema_version": S3_ISING_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomials": candidate.polynomial_text,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight_x": int(row_weights_x.max(initial=0)),
        "check_weight_z": int(row_weights_z.max(initial=0)),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def analyze_s3_l4_candidate(
    candidate: S3L4Candidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Run algebraic and lifted acceptance tests on a weight-12 candidate."""
    return _analyze_s3_l4_candidate(
        candidate,
        expected_check_weight=12,
        include_graph_metrics=include_graph_metrics,
    )


def analyze_s3_l4_w16_candidate(
    candidate: S3L4W16Candidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Run algebraic and lifted acceptance tests on a weight-16 candidate."""
    return _analyze_s3_l4_candidate(
        candidate,
        expected_check_weight=16,
        include_graph_metrics=include_graph_metrics,
    )


def analyze_s3_l4_self_dual_w16_candidate(
    candidate: S3L4SelfDualW16Candidate,
    *,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Run structural acceptance tests for the exact self-dual ansatz."""
    started = time.perf_counter()
    cheap_checks = {
        **self_dual_active_orthogonality_data(candidate),
        "bottom_support_generates_C8xC4": bottom_support_generates(candidate),
    }
    if not all(cheap_checks.values()):
        rejection_reasons = [name for name, passed in cheap_checks.items() if not passed]
        return {
            "schema_version": S3_ISING_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "polynomials": candidate.polynomial_text,
            "accepted": False,
            "rejection_reasons": rejection_reasons,
            "checks": cheap_checks,
            "total_seconds": round(time.perf_counter() - started, 6),
        }

    code = build_s3_l4_self_dual_w16_code(candidate)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)
    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, candidate.num_blocks)
    translation_y = _lift_block_permutation(perm_y, candidate.num_blocks)
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
        "uniform_check_weight_16": bool(
            np.all(row_weights_x == 16) and np.all(row_weights_z == 16)
        ),
        "doubly_even_stabilizer_generators": bool(
            np.all(row_weights_x % 4 == 0) and np.all(row_weights_z % 4 == 0)
        ),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "rate_at_least_half": code.dimension * 2 >= code.num_qubits,
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
        "schema_version": S3_ISING_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomials": candidate.polynomial_text,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight_x": int(row_weights_x.max(initial=0)),
        "check_weight_z": int(row_weights_z.max(initial=0)),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def _analyze_s3_l8_candidate(
    candidate: S3L8Candidate | S3L8W16Candidate,
    *,
    expected_check_weight: int,
    include_graph_metrics: bool = True,
) -> dict[str, Any]:
    """Run algebraic and lifted acceptance tests on an L=8 candidate."""
    started = time.perf_counter()
    cheap_checks = {
        **l8_active_orthogonality_data(candidate),
        "bottom_support_generates_C8xC4": l8_bottom_support_generates(candidate),
    }
    if not all(cheap_checks.values()):
        rejection_reasons = [name for name, passed in cheap_checks.items() if not passed]
        return {
            "schema_version": S3_ISING_SCHEMA_VERSION,
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "polynomials": candidate.polynomial_text,
            "accepted": False,
            "rejection_reasons": rejection_reasons,
            "checks": cheap_checks,
            "total_seconds": round(time.perf_counter() - started, 6),
        }

    ff, gg = l8_candidate_generators(candidate)
    code = codes.GALACode(ff, gg, num_active_rows=2)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)
    block_size = TOP_DEGREE * BOTTOM_ORDER

    # For r_1 at L=8, reverse the four sectors in each half and reverse the
    # two active check-block rows.
    block_order = (3, 2, 1, 0, 7, 6, 5, 4)
    zx_fold = np.hstack(
        [block * block_size + np.arange(block_size) for block in block_order]
    )
    folded_hx = _permute_columns(hx, zx_fold)
    row_order = np.hstack(
        [block_size + np.arange(block_size), np.arange(block_size)]
    )

    perm_x, perm_y = _translation_permutations()
    translation_x = _lift_block_permutation(perm_x, candidate.num_blocks)
    translation_y = _lift_block_permutation(perm_y, candidate.num_blocks)
    rank_x = code.code_x.rank
    rank_z = code.code_z.rank
    checks = {
        **cheap_checks,
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "translation_ZX_fold": bool(np.array_equal(folded_hx[row_order], hz)),
        f"uniform_check_weight_{expected_check_weight}": bool(
            np.all(row_weights_x == expected_check_weight)
            and np.all(row_weights_z == expected_check_weight)
        ),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "rate_at_least_half": code.dimension * 2 >= code.num_qubits,
        "x_translation_stabilizer_automorphism": bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, translation_x)]), code.field)
            == rank_x
            and _gf2_rank(np.vstack([hz, _permute_columns(hz, translation_x)]), code.field)
            == rank_z
        ),
        "y_translation_stabilizer_automorphism": bool(
            _gf2_rank(np.vstack([hx, _permute_columns(hx, translation_y)]), code.field)
            == rank_x
            and _gf2_rank(np.vstack([hz, _permute_columns(hz, translation_y)]), code.field)
            == rank_z
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
        "schema_version": S3_ISING_SCHEMA_VERSION,
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "polynomials": candidate.polynomial_text,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "check_weight_x": int(row_weights_x.max(initial=0)),
        "check_weight_z": int(row_weights_z.max(initial=0)),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "total_seconds": round(time.perf_counter() - started, 6),
    }


def analyze_s3_l8_candidate(
    candidate: S3L8Candidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Run algebraic and lifted acceptance tests on a weight-12 candidate."""
    return _analyze_s3_l8_candidate(
        candidate,
        expected_check_weight=12,
        include_graph_metrics=include_graph_metrics,
    )


def analyze_s3_l8_w16_candidate(
    candidate: S3L8W16Candidate, *, include_graph_metrics: bool = True
) -> dict[str, Any]:
    """Run algebraic and lifted acceptance tests on a weight-16 candidate."""
    return _analyze_s3_l8_candidate(
        candidate,
        expected_check_weight=16,
        include_graph_metrics=include_graph_metrics,
    )


def random_s3_l4_candidates(
    num_samples: int,
    *,
    seed: int = 0,
) -> Iterator[S3L4Candidate]:
    """Sample reproducible weight-12 candidates with one normalized term.

    ``F_0`` has two terms and ``F_1`` has four.  The identity at bottom
    coordinate ``(0,0)`` removes an irrelevant global normalization.  Duplicate
    candidates are suppressed.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    rng = random.Random(seed)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top in range(6)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    anchor = ProductMonomial(0, 0, 0)
    seen: set[S3L4Candidate] = set()
    attempts = 0
    max_attempts = max(100, 50 * num_samples)
    while len(seen) < num_samples and attempts < max_attempts:
        attempts += 1
        second = rng.choice([term for term in all_terms if term != anchor])
        f0 = tuple(sorted((anchor, second)))
        f1 = tuple(sorted(rng.sample(all_terms, 4)))
        candidate = S3L4Candidate(f0=f0, f1=f1)  # type: ignore[arg-type]
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def random_s3_l4_w16_candidates(
    num_samples: int, *, seed: int = 0
) -> Iterator[S3L4W16Candidate]:
    """Sample reproducible L=4 candidates with two weight-four entries."""
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    rng = random.Random(seed)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top in range(6)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    anchor = ProductMonomial(0, 0, 0)
    non_anchor_terms = [term for term in all_terms if term != anchor]
    seen: set[S3L4W16Candidate] = set()
    attempts = 0
    max_attempts = max(100, 50 * num_samples)
    while len(seen) < num_samples and attempts < max_attempts:
        attempts += 1
        f0 = tuple(sorted((anchor, *rng.sample(non_anchor_terms, 3))))
        f1 = tuple(sorted(rng.sample(all_terms, 4)))
        candidate = S3L4W16Candidate(f0=f0, f1=f1)  # type: ignore[arg-type]
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def random_s3_l4_self_dual_w16_candidates(
    num_samples: int, *, seed: int = 0
) -> Iterator[S3L4SelfDualW16Candidate]:
    """Sample the same polynomial supports with the self-dual G relation."""
    for candidate in random_s3_l4_w16_candidates(num_samples, seed=seed):
        yield S3L4SelfDualW16Candidate(f0=candidate.f0, f1=candidate.f1)


def random_s3_l8_candidates(
    num_samples: int, *, seed: int = 0
) -> Iterator[S3L8Candidate]:
    """Sample reproducible ``(1,1,1,3)`` L=8 candidates."""
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    rng = random.Random(seed)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top in range(6)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    anchor = ProductMonomial(0, 0, 0)
    seen: set[S3L8Candidate] = set()
    attempts = 0
    max_attempts = max(100, 50 * num_samples)
    while len(seen) < num_samples and attempts < max_attempts:
        attempts += 1
        singles = rng.sample([term for term in all_terms if term != anchor], 2)
        trinomial = tuple(sorted(rng.sample(all_terms, 3)))
        candidate = S3L8Candidate(
            f0=(anchor,),
            f1=(singles[0],),
            f2=(singles[1],),
            f3=trinomial,  # type: ignore[arg-type]
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def random_s3_l8_w16_candidates(
    num_samples: int, *, seed: int = 0
) -> Iterator[S3L8W16Candidate]:
    """Sample reproducible four-binomial weight-16 candidates."""
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    rng = random.Random(seed)
    all_terms = [
        ProductMonomial(top, xx, yy)
        for top in range(6)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    anchor = ProductMonomial(0, 0, 0)
    non_anchor_terms = [term for term in all_terms if term != anchor]
    seen: set[S3L8W16Candidate] = set()
    attempts = 0
    max_attempts = max(100, 50 * num_samples)
    while len(seen) < num_samples and attempts < max_attempts:
        attempts += 1
        entries = [
            tuple(sorted((anchor, rng.choice(non_anchor_terms)))),
            *[tuple(sorted(rng.sample(all_terms, 2))) for _ in range(3)],
        ]
        candidate = S3L8W16Candidate(*entries)  # type: ignore[arg-type]
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def find_disjoint_translation_logical_seed(
    code: codes.CSSCode,
    *,
    weight: int = 6,
    forbidden_internal_orbits: Sequence[int] = (),
    enforce_seed_nontrivial: bool = False,
    solver: str = "HIGHS",
) -> dict[str, Any] | None:
    """Find one logical seed whose 32 bottom translates are disjoint.

    The graph-support constraint permits at most one selected qubit in each of
    the 24 internal translation fibres.  Consequently all 32 translates of a
    solution are pairwise disjoint.  The returned orbit is independently
    checked modulo the Z-stabilizer row space.  Dense logical-basis constraints
    can optionally be imposed in the MILP; they are normally unnecessary for
    candidates whose stabilizer generators are substantially heavier than the
    requested seed.
    """
    import cvxpy as cp
    from qldpc.objects import Pauli

    block_size = TOP_DEGREE * BOTTOM_ORDER
    if code.num_qubits % block_size:
        raise ValueError("code length is not compatible with the S3 bottom lift")
    num_blocks = code.num_qubits // block_size
    internal_orbits = s3_internal_translation_orbits(num_blocks)
    forbidden = set(map(int, forbidden_internal_orbits))
    if not forbidden <= set(range(len(internal_orbits))):
        raise ValueError("invalid forbidden internal-orbit index")

    check = np.asarray(code.matrix_x, dtype=int)
    error = cp.Variable(code.num_qubits, boolean=True)
    syndrome_slack = cp.Variable(check.shape[0], integer=True)
    internal_used = cp.Variable(len(internal_orbits), boolean=True)
    constraints = [
        check @ error == 2 * syndrome_slack,
        syndrome_slack >= 0,
        syndrome_slack <= np.sum(check, axis=1) // 2,
        cp.sum(error) == weight,
        cp.sum(internal_used) == weight,
    ]
    if enforce_seed_nontrivial:
        conjugate_logicals = np.asarray(code.get_logical_ops(Pauli.X), dtype=int)
        logical_parities = cp.Variable(conjugate_logicals.shape[0], boolean=True)
        logical_slack = cp.Variable(conjugate_logicals.shape[0], integer=True)
        constraints.extend(
            [
                conjugate_logicals @ error
                == logical_parities + 2 * logical_slack,
                logical_slack >= 0,
                logical_slack <= np.sum(conjugate_logicals, axis=1) // 2,
                cp.sum(logical_parities) >= 1,
            ]
        )
    constraints.extend(
        cp.sum(error[list(orbit)]) == internal_used[index]
        for index, orbit in enumerate(internal_orbits)
    )
    constraints.extend(internal_used[index] == 0 for index in forbidden)

    problem = cp.Problem(cp.Minimize(0), constraints)
    started = time.perf_counter()
    problem.solve(solver=solver)
    elapsed = time.perf_counter() - started
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return None
    if problem.status != cp.OPTIMAL or error.value is None:
        raise RuntimeError(f"logical seed MILP did not terminate optimally: {problem.status}")

    seed = np.rint(error.value).astype(np.uint8)
    used = np.flatnonzero(np.rint(internal_used.value).astype(int)).astype(int)
    z_orbit = logical_translation_orbit(seed, num_blocks=num_blocks)
    zx_fold = _zx_fold_permutation(num_blocks)
    x_orbit = _permute_columns(z_orbit, zx_fold)
    stabilizer_z = np.asarray(code.matrix_z, dtype=np.uint8)
    rank_z = code.code_z.rank
    orbit_rank_mod_stabilizers = (
        _gf2_rank(np.vstack([stabilizer_z, z_orbit]), code.field) - rank_z
    )
    pairing = (z_orbit @ x_orbit.T) % 2
    result = {
        "weight": int(np.count_nonzero(seed)),
        "seed_support": np.flatnonzero(seed).astype(int).tolist(),
        "internal_orbits": used.tolist(),
        "translation_orbit_size": int(z_orbit.shape[0]),
        "translated_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in z_orbit
        ],
        "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
        "z_orbit_in_kernel": bool(not np.any((check @ z_orbit.T) % 2)),
        "x_partner_orbit_in_kernel": bool(
            not np.any((stabilizer_z @ x_orbit.T) % 2)
        ),
        "orbit_rank_mod_stabilizers": int(orbit_rank_mod_stabilizers),
        "zx_pairing_rank": _gf2_rank(pairing, code.field),
        "nontriviality_enforced_in_milp": enforce_seed_nontrivial,
        "solver": solver,
        "solver_status": problem.status,
        "seconds": round(elapsed, 6),
    }
    # These conditions are construction invariants, not optional scores.
    if not result["pairwise_disjoint"] or not result["z_orbit_in_kernel"]:
        raise RuntimeError("the returned seed failed its translation-orbit invariant")
    return result


def find_zero_syndrome_at_exact_weight(
    code: codes.CSSCode,
    *,
    weight: int,
    pauli: str = "Z",
    solver: str = "HIGHS",
) -> dict[str, Any] | None:
    """Find any zero-syndrome vector of one exact weight, or prove none exists.

    This deliberately omits dense logical-basis constraints.  An infeasible
    result therefore rules out both logicals and stabilizers at the requested
    weight and is a sufficient distance certificate when lower weights have
    already been screened.  A returned vector is classified against the
    relevant stabilizer row space before it is reported.
    """
    import cvxpy as cp

    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be 'X' or 'Z'")
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=int
    )
    stabilizer = np.asarray(
        code.matrix_x if pauli == "X" else code.matrix_z, dtype=int
    )
    error = cp.Variable(code.num_qubits, boolean=True)
    syndrome_slack = cp.Variable(check.shape[0], integer=True)
    constraints = [
        check @ error == 2 * syndrome_slack,
        syndrome_slack >= 0,
        syndrome_slack <= np.sum(check, axis=1) // 2,
        cp.sum(error) == weight,
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    started = time.perf_counter()
    problem.solve(solver=solver)
    elapsed = time.perf_counter() - started
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return None
    if problem.status != cp.OPTIMAL or error.value is None:
        raise RuntimeError(
            f"exact-weight syndrome MILP did not terminate optimally: {problem.status}"
        )
    solution = np.rint(error.value).astype(int)
    stabilizer_dual = np.asarray(stabilizer, dtype=int).view(code.field).null_space()
    return {
        "pauli": pauli,
        "weight": int(np.count_nonzero(solution)),
        "support": np.flatnonzero(solution).astype(int).tolist(),
        "is_nontrivial_logical": bool(
            np.any(stabilizer_dual @ solution.view(code.field))
        ),
        "solver": solver,
        "solver_status": problem.status,
        "seconds": round(elapsed, 6),
    }


def find_zero_syndrome_by_tanner_search(
    code: codes.CSSCode,
    *,
    weight: int = 6,
    pauli: str = "Z",
    max_nodes: int = 5_000_000,
    use_bottom_translation_normalization: bool = True,
    require_graph_support: bool = False,
    forbidden_internal_orbits: Sequence[int] = (),
    allowed_internal_orbits: Sequence[int] | None = None,
    excluded_translation_orbits: Sequence[Sequence[int]] = (),
) -> dict[str, Any] | None:
    """Search a low-weight kernel using the sparse Tanner incidence graph.

    At every nonzero partial syndrome, some future qubit must touch each odd
    check.  Branching on one such check gives a complete exact-weight search,
    while the degree-four columns keep the branch factor small.  For the L=8
    S3 family, bottom translations allow the first qubit to be fixed to one
    representative of each of the 24 internal fibres.
    """
    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be 'X' or 'Z'")
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=np.uint8
    )
    stabilizer = np.asarray(
        code.matrix_x if pauli == "X" else code.matrix_z, dtype=np.uint8
    )
    num_checks, num_qubits = check.shape
    lift_block_size = TOP_DEGREE * BOTTOM_ORDER
    inferred_num_blocks = (
        num_qubits // lift_block_size if num_qubits % lift_block_size == 0 else None
    )
    internal_orbits: tuple[tuple[int, ...], ...] | None = None
    qubit_to_internal: np.ndarray | None = None
    forbidden = set(map(int, forbidden_internal_orbits))
    allowed = (
        None
        if allowed_internal_orbits is None
        else set(map(int, allowed_internal_orbits))
    )
    if require_graph_support or forbidden or allowed is not None:
        if inferred_num_blocks is None:
            raise ValueError("internal-fibre constraints require an S3 bottom lift")
        internal_orbits = s3_internal_translation_orbits(inferred_num_blocks)
        if not forbidden <= set(range(len(internal_orbits))):
            raise ValueError("invalid forbidden internal-orbit index")
        if allowed is not None and not allowed <= set(range(len(internal_orbits))):
            raise ValueError("invalid allowed internal-orbit index")
        if allowed is not None and allowed & forbidden:
            raise ValueError("an internal orbit cannot be both allowed and forbidden")
        qubit_to_internal = np.empty(num_qubits, dtype=int)
        for internal, orbit in enumerate(internal_orbits):
            qubit_to_internal[list(orbit)] = internal
    column_keys = [
        int.from_bytes(np.packbits(check[:, index]).tobytes())
        for index in range(num_qubits)
    ]
    columns_by_key: dict[int, list[int]] = {}
    for index, key in enumerate(column_keys):
        columns_by_key.setdefault(key, []).append(index)
    # ``int.from_bytes`` uses big-endian byte interpretation.  Build the exact
    # single-bit keys from packed unit syndromes instead of assuming alignment.
    rows_by_packed_bit: dict[int, np.ndarray] = {}
    for row in range(num_checks):
        unit = np.zeros(num_checks, dtype=np.uint8)
        unit[row] = 1
        rows_by_packed_bit[int.from_bytes(np.packbits(unit).tobytes())] = np.flatnonzero(
            check[row]
        )

    if use_bottom_translation_normalization:
        if inferred_num_blocks is None:
            raise ValueError("bottom normalization requires an S3 bottom lift")
        normalization_orbits = s3_internal_translation_orbits(inferred_num_blocks)
        first_qubits = [
            orbit[0]
            for internal, orbit in enumerate(normalization_orbits)
            if internal not in forbidden
            and (allowed is None or internal in allowed)
        ]
    else:
        first_qubits = list(range(num_qubits))

    nodes = 0
    cutoff = False
    started = time.perf_counter()
    excluded = {
        _canonical_bottom_translation_support(
            support, num_qubits=num_qubits, num_blocks=int(inferred_num_blocks)
        )
        for support in excluded_translation_orbits
    }

    def admissible_solution(selected: frozenset[int]) -> tuple[int, ...] | None:
        support = tuple(sorted(selected))
        canonical = _canonical_bottom_translation_support(
            support, num_qubits=num_qubits, num_blocks=int(inferred_num_blocks)
        )
        return None if canonical in excluded else support

    def recurse(selected: frozenset[int], syndrome: int) -> tuple[int, ...] | None:
        nonlocal nodes, cutoff
        nodes += 1
        if nodes > max_nodes:
            cutoff = True
            return None
        remaining = weight - len(selected)
        if remaining == 0:
            return admissible_solution(selected) if syndrome == 0 else None
        if syndrome == 0:
            # A solution extending a smaller kernel component is not needed for
            # the distance-six candidates targeted here.
            return None
        if syndrome.bit_count() > 4 * remaining:
            return None
        if remaining == 1:
            for qubit in columns_by_key.get(syndrome, ()):
                if qubit in selected:
                    continue
                if qubit_to_internal is not None:
                    internal = int(qubit_to_internal[qubit])
                    if internal in forbidden:
                        continue
                    if allowed is not None and internal not in allowed:
                        continue
                    if require_graph_support and any(
                        int(qubit_to_internal[selected_qubit]) == internal
                        for selected_qubit in selected
                    ):
                        continue
                result = admissible_solution(selected | {qubit})
                if result is not None:
                    return result
            return None

        # Select the odd check with the fewest currently available qubits.
        odd_bits: list[int] = []
        value = syndrome
        while value:
            bit = value & -value
            odd_bits.append(bit)
            value ^= bit
        branch_columns: np.ndarray | None = None
        for bit in odd_bits:
            available = np.asarray(
                [q for q in rows_by_packed_bit[bit] if int(q) not in selected],
                dtype=int,
            )
            if branch_columns is None or len(available) < len(branch_columns):
                branch_columns = available
        assert branch_columns is not None
        for raw_qubit in branch_columns:
            qubit = int(raw_qubit)
            if qubit_to_internal is not None:
                internal = int(qubit_to_internal[qubit])
                if internal in forbidden:
                    continue
                if allowed is not None and internal not in allowed:
                    continue
                if require_graph_support and any(
                    int(qubit_to_internal[selected_qubit]) == internal
                    for selected_qubit in selected
                ):
                    continue
            result = recurse(selected | {qubit}, syndrome ^ column_keys[qubit])
            if result is not None:
                return result
            if cutoff:
                return None
        return None

    support: tuple[int, ...] | None = None
    for first in first_qubits:
        support = recurse(frozenset({first}), column_keys[first])
        if support is not None or cutoff:
            break
    if support is None:
        return {
            "pauli": pauli,
            "weight": weight,
            "support": None,
            "search_exhaustive": not cutoff,
            "nodes": nodes,
            "seconds": round(time.perf_counter() - started, 6),
        }

    solution = np.zeros(num_qubits, dtype=int)
    solution[list(support)] = 1
    stabilizer_dual = np.asarray(stabilizer, dtype=int).view(code.field).null_space()
    internal_occupancies: list[int] | None = None
    if inferred_num_blocks is not None:
        internal_occupancies = [
            int(np.count_nonzero(solution[list(orbit)]))
            for orbit in s3_internal_translation_orbits(inferred_num_blocks)
        ]
    return {
        "pauli": pauli,
        "weight": len(support),
        "support": list(support),
        "is_nontrivial_logical": bool(
            np.any(stabilizer_dual @ solution.view(code.field))
        ),
        "internal_occupancies": internal_occupancies,
        "graph_supported": bool(
            internal_occupancies is not None and max(internal_occupancies) <= 1
        ),
        "search_exhaustive": not cutoff,
        "nodes": nodes,
        "seconds": round(time.perf_counter() - started, 6),
    }


def certify_global_css_distance(
    code: codes.CSSCode,
    *,
    pauli: str = "Z",
    solver: str = "HIGHS",
) -> dict[str, Any]:
    """Minimize over every nontrivial logical class in one MILP.

    A zero-syndrome vector is nontrivial exactly when it has odd pairing with
    at least one member of a complete conjugate logical basis.  Binary variables
    encode all such parities and their sum is constrained to be nonzero.
    """
    import cvxpy as cp
    from qldpc.objects import Pauli

    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be 'X' or 'Z'")
    error_type = Pauli.X if pauli == "X" else Pauli.Z
    conjugate_type = Pauli.Z if pauli == "X" else Pauli.X
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=int
    )
    conjugate_logicals = np.asarray(code.get_logical_ops(conjugate_type), dtype=int)
    num_qubits = code.num_qubits

    error = cp.Variable(num_qubits, boolean=True)
    syndrome_slack = cp.Variable(check.shape[0], integer=True)
    logical_parities = cp.Variable(conjugate_logicals.shape[0], boolean=True)
    logical_slack = cp.Variable(conjugate_logicals.shape[0], integer=True)
    constraints = [
        check @ error == 2 * syndrome_slack,
        syndrome_slack >= 0,
        syndrome_slack <= np.sum(check, axis=1) // 2,
        conjugate_logicals @ error == logical_parities + 2 * logical_slack,
        logical_slack >= 0,
        logical_slack <= np.sum(conjugate_logicals, axis=1) // 2,
        cp.sum(logical_parities) >= 1,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(error)), constraints)
    started = time.perf_counter()
    value = problem.solve(solver=solver)
    elapsed = time.perf_counter() - started
    if problem.status != cp.OPTIMAL or error.value is None or not np.isfinite(value):
        raise RuntimeError(f"distance MILP did not terminate optimally: {problem.status}")
    solution = np.rint(error.value).astype(int)
    return {
        "pauli": pauli,
        "minimum_weight": int(np.count_nonzero(solution)),
        "support": np.flatnonzero(solution).astype(int).tolist(),
        "solver": solver,
        "solver_status": problem.status,
        "seconds": round(elapsed, 6),
        "num_logical_classes": conjugate_logicals.shape[0],
    }


def find_logical_up_to_weight_four(
    code: codes.CSSCode, *, pauli: str = "Z"
) -> dict[str, Any] | None:
    """Find a nontrivial zero-syndrome vector of weight at most four.

    Packed syndrome columns turn weights two through four into duplicate and
    meet-in-the-middle lookups.  Every returned witness is checked against the
    stabilizer row space, so a low-weight stabilizer is never misreported as a
    logical operator.
    """
    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be 'X' or 'Z'")
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=np.uint8
    )
    stabilizer = np.asarray(
        code.matrix_x if pauli == "X" else code.matrix_z, dtype=np.uint8
    )
    field = code.field
    stabilizer_dual = np.asarray(stabilizer, dtype=int).view(field).null_space()
    num_qubits = code.num_qubits
    syndrome_keys = [
        int.from_bytes(np.packbits(check[:, index]).tobytes())
        for index in range(num_qubits)
    ]

    def is_nontrivial(support: Sequence[int]) -> bool:
        vector = np.zeros(num_qubits, dtype=int)
        vector[list(support)] = 1
        return bool(np.any(stabilizer_dual @ vector.view(field)))

    started = time.perf_counter()
    zero_columns = [index for index, key in enumerate(syndrome_keys) if key == 0]
    for index in zero_columns:
        if is_nontrivial([index]):
            return {
                "pauli": pauli,
                "minimum_weight_upper_bound": 1,
                "support": [index],
                "method": "syndrome_mitm",
                "seconds": round(time.perf_counter() - started, 6),
            }

    by_syndrome: dict[int, list[int]] = {}
    for index, key in enumerate(syndrome_keys):
        for other in by_syndrome.get(key, []):
            support = [other, index]
            if is_nontrivial(support):
                return {
                    "pauli": pauli,
                    "minimum_weight_upper_bound": 2,
                    "support": support,
                    "method": "syndrome_mitm",
                    "seconds": round(time.perf_counter() - started, 6),
                }
        by_syndrome.setdefault(key, []).append(index)

    # A weight-three zero syndrome has c_i + c_j equal to a third column.
    for left in range(num_qubits):
        key_left = syndrome_keys[left]
        for right in range(left + 1, num_qubits):
            pair_key = key_left ^ syndrome_keys[right]
            for third in by_syndrome.get(pair_key, []):
                if third in (left, right):
                    continue
                support = sorted((left, right, third))
                if is_nontrivial(support):
                    return {
                        "pauli": pauli,
                        "minimum_weight_upper_bound": 3,
                        "support": support,
                        "method": "syndrome_mitm",
                        "seconds": round(time.perf_counter() - started, 6),
                    }

    # A weight-four zero syndrome is a collision between two disjoint pairs.
    pairs_by_syndrome: dict[int, list[tuple[int, int]]] = {}
    for left in range(num_qubits):
        key_left = syndrome_keys[left]
        for right in range(left + 1, num_qubits):
            pair_key = key_left ^ syndrome_keys[right]
            previous_pairs = pairs_by_syndrome.get(pair_key, [])
            for old_left, old_right in previous_pairs:
                if len({left, right, old_left, old_right}) != 4:
                    continue
                support = sorted((left, right, old_left, old_right))
                if is_nontrivial(support):
                    return {
                        "pauli": pauli,
                        "minimum_weight_upper_bound": 4,
                        "support": support,
                        "method": "syndrome_mitm",
                        "seconds": round(time.perf_counter() - started, 6),
                    }
            # Retaining a few overlapping pairs is enough to find a disjoint
            # representative without allowing pathological dictionaries to grow.
            if len(previous_pairs) < 8:
                pairs_by_syndrome.setdefault(pair_key, []).append((left, right))
    return None
