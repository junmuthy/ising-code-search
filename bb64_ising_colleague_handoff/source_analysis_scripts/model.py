"""Sparse polynomial GALA models over the logical group ``C4 x C2``.

The implemented strict ZX folds use represented binary transposes and a
reflection of the protograph blocks.  For the initial ``L=4`` family, fold
axis zero gives ``H_X = H_Z`` and axis one gives transversal Hadamard followed
by a fixed block permutation.
"""

from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal, Sequence

import networkx as nx
import numpy as np
from qldpc import abstract, codes

GRID_X_ORDER = 4
GRID_Y_ORDER = 2
GRID_ORDER = GRID_X_ORDER * GRID_Y_ORDER
SCHEMA_VERSION = 1

TopRepresentation = Literal["trivial", "s3-natural", "s3-linear"]


@dataclass(frozen=True, order=True)
class Monomial:
    """One term ``h x^a y^b`` in a product-group polynomial."""

    top: int
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.top < 0:
            raise ValueError("top index must be nonnegative")
        if not 0 <= self.x < GRID_X_ORDER:
            raise ValueError("x exponent must lie in range(4)")
        if not 0 <= self.y < GRID_Y_ORDER:
            raise ValueError("y exponent must lie in range(2)")


@dataclass(frozen=True)
class Candidate:
    """A sparse exact-ZX GALA candidate.

    ``entries`` stores ``F_0, ..., F_(L/2-1)``.  The G sequence is derived,
    rather than searched independently, from the requested strict ZX fold.
    """

    top_representation: TopRepresentation
    entries: tuple[tuple[Monomial, ...], ...]
    fold_axis: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("at least one F entry is required")
        if not 0 <= self.fold_axis < len(self.entries):
            raise ValueError("fold axis must lie between zero and L/2 - 1")
        top_count = 1 if self.top_representation == "trivial" else 6
        for entry in self.entries:
            if not entry:
                raise ValueError("polynomial entries must be nonempty")
            if tuple(sorted(entry)) != entry or len(set(entry)) != len(entry):
                raise ValueError("every polynomial support must be sorted and distinct")
            if any(term.top >= top_count for term in entry):
                raise ValueError("monomial top index is invalid for this representation")

    @property
    def half_blocks(self) -> int:
        return len(self.entries)

    @property
    def num_blocks(self) -> int:
        return 2 * self.half_blocks

    @property
    def top_lift_dimension(self) -> int:
        return {
            "trivial": 1,
            "s3-natural": 3,
            "s3-linear": 2,
        }[self.top_representation]

    @property
    def block_size(self) -> int:
        return self.top_lift_dimension * GRID_ORDER

    @property
    def num_qubits(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def nominal_check_weight(self) -> int:
        return 2 * sum(map(len, self.entries))

    @property
    def candidate_id(self) -> str:
        payload = (
            (self.top_representation, self.entries)
            if self.fold_axis == 0
            else (self.top_representation, self.entries, self.fold_axis)
        )
        digest = hashlib.sha256(repr(payload).encode()).hexdigest()
        fold_label = "" if self.fold_axis == 0 else f"-fold{self.fold_axis}"
        return (
            f"c4xc2-{self.top_representation}-l{self.num_blocks}-j1-"
            f"w{self.nominal_check_weight}{fold_label}-{digest[:16]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "top_representation": self.top_representation,
            "entries": [[asdict(term) for term in entry] for entry in self.entries],
            "fold_axis": self.fold_axis,
            "label": self.label,
            "num_blocks": self.num_blocks,
            "num_qubits": self.num_qubits,
            "nominal_check_weight": self.nominal_check_weight,
            "zx_relation": "lift(G_j) = lift(F_(j+fold_axis))^T",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            top_representation=data["top_representation"],
            entries=tuple(
                tuple(sorted(Monomial(**term) for term in entry))
                for entry in data["entries"]
            ),
            fold_axis=int(data.get("fold_axis", 0)),
            label=data.get("label", ""),
        )


@functools.lru_cache(maxsize=None)
def group_data(
    top_representation: TopRepresentation,
) -> tuple[
    abstract.GroupRing,
    tuple[abstract.GroupMember, ...],
    abstract.GroupMember,
    abstract.GroupMember,
]:
    """Build the represented product group and its two logical translations."""
    bottom = abstract.AbelianGroup(GRID_X_ORDER, GRID_Y_ORDER)
    xx, yy = bottom.generators
    if top_representation == "trivial":
        ring = abstract.GroupRing(bottom)
        identity = next(iter(sorted(abstract.TrivialGroup().generate())))
        return ring, (identity,), xx, yy
    if top_representation == "s3-natural":
        top = abstract.SymmetricGroup(3).with_natural_lift()
        product = abstract.Group.tensor_product(top, bottom)
        ring = abstract.GroupRing(product)
        return ring, tuple(sorted(top.generate())), xx, yy
    if top_representation == "s3-linear":
        top = abstract.SpecialLinearGroup(2, 2, linear_rep=True)
        product = abstract.Group.tensor_product(top, bottom)
        ring = abstract.GroupRing(product)
        return ring, tuple(sorted(top.generate())), xx, yy
    raise ValueError(f"unsupported top representation: {top_representation}")


@functools.lru_cache(maxsize=None)
def ring_monomial(top_representation: TopRepresentation, term: Monomial) -> abstract.RingMember:
    ring, top_members, xx, yy = group_data(top_representation)
    bottom_member = xx**term.x * yy**term.y
    member = bottom_member if top_representation == "trivial" else top_members[term.top] @ bottom_member
    return abstract.RingMember(ring, member)


@functools.lru_cache(maxsize=None)
def ring_polynomial(
    top_representation: TopRepresentation, terms: tuple[Monomial, ...]
) -> abstract.RingMember:
    ring, _top_members, _xx, _yy = group_data(top_representation)
    return functools.reduce(
        lambda left, right: left + right,
        (ring_monomial(top_representation, term) for term in terms),
        ring.zero,
    )


@functools.lru_cache(maxsize=None)
def _top_binary_transpose_map(
    top_representation: TopRepresentation,
) -> tuple[int, ...]:
    """Map each represented top element to the element lifting to its transpose."""
    if top_representation == "trivial":
        return (0,)
    lifts = [
        np.asarray(
            ring_monomial(top_representation, Monomial(top, 0, 0)).lift(),
            dtype=np.uint8,
        )
        for top in range(6)
    ]
    output = []
    for matrix in lifts:
        matches = [
            index
            for index, candidate in enumerate(lifts)
            if np.array_equal(candidate, matrix.T)
        ]
        if len(matches) != 1:
            raise RuntimeError("represented top group is not closed under binary transpose")
        output.append(matches[0])
    return tuple(output)


def binary_transpose_terms(
    top_representation: TopRepresentation,
    terms: Sequence[Monomial],
) -> tuple[Monomial, ...]:
    """Return the polynomial whose represented lift is the binary transpose."""
    top_transpose = _top_binary_transpose_map(top_representation)
    return tuple(
        sorted(
            Monomial(
                top_transpose[term.top],
                (-term.x) % GRID_X_ORDER,
                (-term.y) % GRID_Y_ORDER,
            )
            for term in terms
        )
    )


def candidate_generators(
    candidate: Candidate,
) -> tuple[tuple[abstract.RingMember, ...], tuple[abstract.RingMember, ...]]:
    """Return F and the exact-self-dual G sequence."""
    ff = tuple(
        ring_polynomial(candidate.top_representation, entry)
        for entry in candidate.entries
    )
    size = len(ff)
    gg = tuple(
        ring_polynomial(
            candidate.top_representation,
            binary_transpose_terms(
                candidate.top_representation,
                candidate.entries[(index + candidate.fold_axis) % size],
            ),
        )
        for index in range(size)
    )
    return ff, gg


def _actual_transpose_checks(
    ff: Sequence[abstract.RingMember],
    gg: Sequence[abstract.RingMember],
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a GALA row using actual binary transposes for a nonorthogonal top."""
    size = len(ff)
    block_size = np.asarray(ff[0].lift()).shape[0]

    def circulant(generators: Sequence[abstract.RingMember]) -> np.ndarray:
        return np.vstack(
            [
                np.hstack(
                    [
                        np.asarray(
                            generators[(column - row) % size].lift(),
                            dtype=np.uint8,
                        )
                        for column in range(size)
                    ]
                )
                for row in range(size)
            ]
        )

    matrix_f = circulant(ff)
    matrix_g = circulant(gg)
    hx = np.hstack([matrix_f[:block_size], matrix_g[:block_size]])
    hz = np.hstack([matrix_g.T[:block_size], matrix_f.T[:block_size]])
    return hx, hz


def _build_actual_transpose_code(
    ff: Sequence[abstract.RingMember],
    gg: Sequence[abstract.RingMember],
) -> codes.CSSCode:
    """Construct the validated CSS code from actual-transpose GALA checks."""
    return codes.CSSCode(*_actual_transpose_checks(ff, gg))


def build_code(candidate: Candidate, *, skip_validation: bool = False) -> codes.CSSCode:
    ff, gg = candidate_generators(candidate)
    if candidate.top_representation == "s3-linear":
        return _build_actual_transpose_code(ff, gg)
    return codes.GALACode(ff, gg, num_active_rows=1, skip_validation=skip_validation)


def gf2_rank(matrix: np.ndarray, field: type[np.ndarray]) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=np.uint8).view(field)))


def _permute_columns(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


@functools.lru_cache(maxsize=None)
def translation_permutations(
    top_representation: TopRepresentation,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical C4 and C2 permutations on one represented lift block."""
    ring, top_members, xx, yy = group_data(top_representation)
    matrices = []
    for bottom_member in (xx, yy):
        member = bottom_member if top_representation == "trivial" else top_members[0] @ bottom_member
        matrices.append(np.asarray(ring.group.lift(member), dtype=np.uint8))
    return tuple(np.argmax(matrix, axis=1).astype(int) for matrix in matrices)  # type: ignore[return-value]


def lift_block_permutation(permutation: np.ndarray, num_blocks: int) -> np.ndarray:
    block_size = len(permutation)
    return np.hstack(
        [block * block_size + permutation for block in range(num_blocks)]
    )


def zx_fold_permutation(candidate: Candidate) -> np.ndarray:
    """Physical block reflections implementing the strict GALA ZX fold.

    If ``G_j = F_(j+a)^T``, the F half is reflected as ``j -> a-j`` and the
    G half as ``j -> -a-j``.  The two reflections happen to coincide when
    there are two entries, which is why the distinction is invisible at
    ``L=4``.
    """
    if getattr(candidate, "zx_fold_kind", None) in {
        "half_swap",
        "twisted_half_swap",
    }:
        half = candidate.num_blocks // 2
        internal = np.asarray(
            getattr(
                candidate,
                "zx_internal_permutation",
                np.arange(candidate.block_size),
            ),
            dtype=int,
        )
        if internal.shape != (candidate.block_size,):
            raise ValueError("ZX internal permutation has the wrong size")
        full_order = tuple(range(half, candidate.num_blocks)) + tuple(range(half))
        return np.hstack(
            [
                block * candidate.block_size + internal
                for block in full_order
            ]
        ).astype(int)

    if getattr(candidate, "zx_fold_kind", None) == "twisted_protograph":
        half = candidate.num_blocks // 2
        internal = np.asarray(candidate.zx_internal_permutation, dtype=int)
        if internal.shape != (candidate.block_size,):
            raise ValueError("ZX internal permutation has the wrong size")
        # Source block i in either half maps to the opposite block -i.  Row
        # reflection then turns each reflected-transpose-invariant block into
        # the actual represented binary transpose required by H_Z.
        full_order = tuple(
            [half + (-block) % half for block in range(half)]
            + [(-block) % half for block in range(half)]
        )
        return np.hstack(
            [
                block * candidate.block_size + internal
                for block in full_order
            ]
        ).astype(int)

    half = candidate.half_blocks
    left_order = tuple(
        (candidate.fold_axis - block) % half for block in range(half)
    )
    right_order = tuple(
        (-candidate.fold_axis - block) % half for block in range(half)
    )
    full_order = (*left_order, *(half + block for block in right_order))
    return np.hstack(
        [
            block * candidate.block_size + np.arange(candidate.block_size)
            for block in full_order
        ]
    ).astype(int)


def bottom_support_generates(candidate: Candidate) -> bool:
    """Require polynomial differences to generate all of ``C4 x C2``."""
    points = [(term.x, term.y) for entry in candidate.entries for term in entry]
    anchor = points[0]
    generators = {
        ((xx - anchor[0]) % GRID_X_ORDER, (yy - anchor[1]) % GRID_Y_ORDER)
        for xx, yy in points[1:]
    }
    reached = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        point = frontier.pop()
        for dx, dy in generators:
            image = ((point[0] + dx) % GRID_X_ORDER, (point[1] + dy) % GRID_Y_ORDER)
            if image not in reached:
                reached.add(image)
                frontier.append(image)
    return len(reached) == GRID_ORDER


def _tanner_connected(matrix: np.ndarray) -> bool:
    graph = nx.Graph()
    rows, cols = matrix.shape
    graph.add_nodes_from(range(rows), bipartite=0)
    graph.add_nodes_from(range(rows, rows + cols), bipartite=1)
    for row, col in np.argwhere(matrix):
        graph.add_edge(int(row), rows + int(col))
    return nx.is_connected(graph)


def analyze_structure(candidate: Candidate, *, check_weight_ceiling: int = 16) -> dict[str, Any]:
    """Run the cheap exact structural gates before logical or distance work."""
    checks: dict[str, bool] = {
        "n_at_most_200": candidate.num_qubits <= 200,
        "nominal_check_weight_at_most_ceiling": candidate.nominal_check_weight
        <= check_weight_ceiling,
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
        code = build_code(candidate)
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
    px, py = translation_permutations(candidate.top_representation)
    full_px = lift_block_permutation(px, candidate.num_blocks)
    full_py = lift_block_permutation(py, candidate.num_blocks)
    zx_fold = zx_fold_permutation(candidate)

    def translation_is_automorphism(permutation: np.ndarray) -> bool:
        return bool(
            gf2_rank(np.vstack([hx, _permute_columns(hx, permutation)]), code.field)
            == code.code_x.rank
            and gf2_rank(np.vstack([hz, _permute_columns(hz, permutation)]), code.field)
            == code.code_z.rank
        )

    checks.update(
        {
            "active_css_orthogonality": not bool(np.any((hx @ hz.T) % 2)),
            "strict_ZX_fold": bool(
                np.array_equal(_permute_columns(hx, zx_fold), hz)
                and np.array_equal(_permute_columns(hz, zx_fold), hx)
            ),
            "k_at_least_8": code.dimension >= 8,
            "actual_check_weight_at_most_ceiling": bool(
                row_weights_x.max(initial=0) <= check_weight_ceiling
                and row_weights_z.max(initial=0) <= check_weight_ceiling
            ),
            "no_zero_checks": bool(np.all(row_weights_x > 0) and np.all(row_weights_z > 0)),
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
        "zx_fold_is_identity": bool(
            np.array_equal(zx_fold, np.arange(code.num_qubits))
            and np.array_equal(hx, hz)
        ),
        "zx_fold_permutation": zx_fold.astype(int).tolist(),
    }


def quotient_s3_seed() -> Candidate:
    """Quotient the saved ``[[384,200,7]]`` seed to ``S3 x C4 x C2``."""
    mm = Monomial
    return Candidate(
        top_representation="s3-natural",
        entries=(
            tuple(sorted((mm(0, 0, 0), mm(0, 2, 0), mm(2, 0, 1), mm(5, 0, 1)))),
            tuple(sorted((mm(0, 2, 0), mm(0, 0, 1), mm(3, 0, 0), mm(3, 1, 0)))),
        ),
        label="quotient of s3-l4-j1-self-dual-w16-36d018790fa7a28a",
    )


def random_candidates(
    *,
    top_representation: TopRepresentation,
    entry_weights: Sequence[int],
    count: int,
    seed: int,
    fold_axis: int = 0,
) -> Iterator[Candidate]:
    """Sample normalized sparse candidates reproducibly."""
    if count < 0 or any(weight < 1 for weight in entry_weights):
        raise ValueError("candidate counts and entry weights must be positive")
    top_count = 1 if top_representation == "trivial" else 6
    terms = [
        Monomial(top, xx, yy)
        for top in range(top_count)
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    ]
    anchor = Monomial(0, 0, 0)
    rng = random.Random(seed)
    seen: set[tuple[tuple[Monomial, ...], ...]] = set()
    attempts = 0
    max_attempts = max(100, count * 100)
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        entries: list[tuple[Monomial, ...]] = []
        first_weight = int(entry_weights[0])
        first = tuple(sorted((anchor, *rng.sample([term for term in terms if term != anchor], first_weight - 1))))
        entries.append(first)
        entries.extend(
            tuple(sorted(rng.sample(terms, int(weight))))
            for weight in entry_weights[1:]
        )
        key = tuple(entries)
        if key in seen:
            continue
        seen.add(key)
        yield Candidate(
            top_representation=top_representation,
            entries=key,
            fold_axis=fold_axis,
        )
