"""Two-dimensional ``SL(2,2) ~= S3`` lifts for rapid GALA rescreening."""

from __future__ import annotations

import functools
import itertools
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from qldpc import abstract, codes
from qldpc.objects import Pauli

from .s3_ising import (
    ProductMonomial,
    _bottom_subgroup_size,
    _gf2_rank,
    _polynomial_product_support,
    _transpose_terms,
    find_logical_up_to_weight_four,
)

BOTTOM_X_ORDER = 8
BOTTOM_Y_ORDER = 4
BOTTOM_ORDER = BOTTOM_X_ORDER * BOTTOM_Y_ORDER
TOP_LIFT_DIM = 2
BLOCK_SIZE = TOP_LIFT_DIM * BOTTOM_ORDER

Family = Literal[
    "l4-fold",
    "l4-self-dual",
    "l8-fold",
    "l12-edge-fold",
    "l12-vertex-fold",
    "l12-self-dual",
]


@functools.lru_cache(maxsize=1)
def linear_group_data() -> tuple[
    abstract.GroupRing,
    tuple[abstract.GroupMember, ...],
    abstract.GroupMember,
    abstract.GroupMember,
]:
    """Return ``SL(2,2) x C8 x C4`` with a two-dimensional top lift."""
    top = abstract.SpecialLinearGroup(2, 2, linear_rep=True)
    bottom = abstract.AbelianGroup(BOTTOM_X_ORDER, BOTTOM_Y_ORDER)
    product = abstract.Group.tensor_product(top, bottom)
    ring = abstract.GroupRing(product)
    top_members = tuple(sorted(top.generate()))
    xx, yy = bottom.generators
    return ring, top_members, xx, yy


@functools.lru_cache(maxsize=None)
def linear_ring_monomial(term: ProductMonomial) -> abstract.RingMember:
    ring, top_members, xx, yy = linear_group_data()
    member = top_members[term.top] @ (xx**term.x * yy**term.y)
    return abstract.RingMember(ring, member)


@functools.lru_cache(maxsize=None)
def linear_ring_polynomial(
    terms: tuple[ProductMonomial, ...],
) -> abstract.RingMember:
    ring, _top_members, _xx, _yy = linear_group_data()
    return functools.reduce(
        lambda left, right: left + right,
        (linear_ring_monomial(term) for term in terms),
        ring.zero,
    )


def candidate_entries(record: dict[str, Any]) -> tuple[tuple[ProductMonomial, ...], ...]:
    """Read polynomial entries from any saved S3 search record."""
    data = record.get("candidate", record)
    if "entries" in data:
        raw_entries = data["entries"]
    else:
        raw_entries = [
            data[name]
            for name in ("f0", "f1", "f2", "f3")
            if name in data
        ]
    return tuple(
        tuple(sorted(ProductMonomial(**term) for term in entry))
        for entry in raw_entries
    )


def infer_family(record: dict[str, Any]) -> Family:
    """Map a saved family label to its two-dimensional construction rule."""
    label = record.get("candidate", record).get("family", "")
    mappings: dict[str, Family] = {
        "s3_l4_j1_weight12": "l4-fold",
        "s3_l4_j1_weight16": "l4-fold",
        "s3_l4_j1_self_dual_weight16": "l4-self-dual",
        "s3_l8_j2_weight12": "l8-fold",
        "s3_l8_j2_weight16": "l8-fold",
        "s3_l12_j3_fold_weight12": "l12-edge-fold",
        "s3_l12_j3_vertex_fold_weight12": "l12-vertex-fold",
        "s3_l12_j3_self_dual_weight12": "l12-self-dual",
    }
    try:
        return mappings[label]
    except KeyError as error:
        raise ValueError(f"unsupported saved S3 family: {label}") from error


def linear_generators(
    entries: Sequence[tuple[ProductMonomial, ...]], family: Family
) -> tuple[tuple[abstract.RingMember, ...], tuple[abstract.RingMember, ...], int]:
    """Construct F/G generators with the saved family's ring-level relation."""
    ff = tuple(linear_ring_polynomial(tuple(entry)) for entry in entries)
    size = len(ff)
    gg: list[abstract.RingMember | None] = [None] * size
    if family == "l4-fold":
        shift, active_rows = 1, 1
        for index, member in enumerate(ff):
            gg[(index + shift) % size] = member.T
    elif family == "l4-self-dual":
        active_rows = 1
        for index, member in enumerate(ff):
            gg[index] = member.T
    elif family == "l8-fold":
        shift, active_rows = 2, 2
        for index, member in enumerate(ff):
            gg[(index + shift) % size] = member.T
    elif family == "l12-edge-fold":
        shift, active_rows = 3, 3
        for index, member in enumerate(ff):
            gg[(index + shift) % size] = member.T
    elif family == "l12-vertex-fold":
        active_rows = 3
        for index, member in enumerate(ff):
            gg[index] = member.T
    elif family == "l12-self-dual":
        active_rows = 3
        for index in range(size):
            gg[index] = ff[(-index) % size].T
    else:  # pragma: no cover
        raise ValueError(f"unsupported family: {family}")
    assert all(member is not None for member in gg)
    return ff, tuple(gg), active_rows  # type: ignore[arg-type,return-value]


def build_linear_code(
    entries: Sequence[tuple[ProductMonomial, ...]], family: Family
) -> codes.CSSCode:
    """Build GALA checks using the actual binary transpose.

    qLDPC's ``GALACode`` currently lifts the group-ring inverse when forming
    its transposed blocks.  That equals the binary transpose for orthogonal
    permutation lifts, but not for the two-dimensional ``SL(2,2)`` lift.
    Constructing the parent block matrices explicitly preserves the intended
    CSS formula ``H_Z = [G^T | F^T]``.
    """
    ff, gg, active_rows = linear_generators(entries, family)
    return _build_linear_css_code(ff, gg, active_rows)


def _build_linear_css_code(
    ff: Sequence[abstract.RingMember],
    gg: Sequence[abstract.RingMember],
    active_rows: int,
) -> codes.CSSCode:
    """Lift arbitrary block-circulant F/G rows using actual transposes."""
    if len(ff) != len(gg):
        raise ValueError("F and G must have the same number of entries")
    size = len(ff)
    if not 1 <= active_rows <= size:
        raise ValueError("active row count must lie between one and half L")

    def block_circulant(generators: Sequence[abstract.RingMember]) -> np.ndarray:
        rows = []
        for row in range(size):
            rows.append(
                np.hstack(
                    [
                        np.asarray(
                            generators[(column - row) % size].lift(),
                            dtype=np.uint8,
                        )
                        for column in range(size)
                    ]
                )
            )
        return np.vstack(rows)

    matrix_f = block_circulant(ff)
    matrix_g = block_circulant(gg)
    active_size = active_rows * BLOCK_SIZE
    matrix_x = np.hstack(
        [matrix_f[:active_size], matrix_g[:active_size]]
    )
    matrix_z = np.hstack(
        [matrix_g.T[:active_size], matrix_f.T[:active_size]]
    )
    return codes.CSSCode(matrix_x, matrix_z)


def build_linear_fold_code(
    entries: Sequence[tuple[ProductMonomial, ...]],
    *,
    active_rows: int,
    shift: int,
) -> codes.CSSCode:
    """Build an arbitrary even-L folded family with ``G_(i+s)=F_i.T``."""
    ff = tuple(linear_ring_polynomial(tuple(entry)) for entry in entries)
    size = len(ff)
    gg: list[abstract.RingMember | None] = [None] * size
    for index, member in enumerate(ff):
        gg[(index + shift) % size] = member.T
    assert all(member is not None for member in gg)
    return _build_linear_css_code(
        ff, tuple(gg), active_rows  # type: ignore[arg-type]
    )


def linear_fold_permutation(
    *, half_blocks: int, active_rows: int, shift: int
) -> np.ndarray:
    """Return the contragredient ZX fold for an arbitrary folded family."""
    axis = (active_rows - 1 - shift) % half_blocks
    return fold_permutation(2 * half_blocks, axis, swap_top=True)


def linear_fold_active_orthogonality_data(
    entries: Sequence[tuple[ProductMonomial, ...]],
    *,
    active_rows: int,
    shift: int,
) -> dict[str, Any]:
    """Evaluate active and latent ring correlations for a folded family."""
    size = len(entries)
    if not 1 <= active_rows <= size:
        raise ValueError("active row count must lie between one and half L")
    gg: list[tuple[ProductMonomial, ...] | None] = [None] * size
    for index, entry in enumerate(entries):
        gg[(index + shift) % size] = _transpose_terms(entry)
    assert all(entry is not None for entry in gg)

    def correlation(offset: int) -> frozenset[ProductMonomial]:
        support: frozenset[ProductMonomial] = frozenset()
        for index, left in enumerate(entries):
            right = gg[(offset - index) % size]
            assert right is not None
            support ^= _polynomial_product_support(left, right)
            support ^= _polynomial_product_support(right, left)
        return support

    active_offsets = sorted(
        set(range(active_rows))
        | set(range(size - active_rows + 1, size))
    )
    latent_offsets = [
        offset for offset in range(size) if offset not in active_offsets
    ]
    active_sizes = {
        str(offset): len(correlation(offset)) for offset in active_offsets
    }
    latent_sizes = {
        str(offset): len(correlation(offset)) for offset in latent_offsets
    }
    return {
        "active_offsets": active_offsets,
        "latent_offsets": latent_offsets,
        "active_support_sizes": active_sizes,
        "latent_support_sizes": latent_sizes,
        "active_offsets_zero": not any(active_sizes.values()),
        "some_latent_offset_nonzero": any(latent_sizes.values()),
    }


def linear_bottom_support_generates(
    entries: Sequence[tuple[ProductMonomial, ...]],
) -> bool:
    """Whether the bottom exponents generate all of ``C8 x C4``."""
    points = [term.bottom for entry in entries for term in entry]
    return bool(points) and _bottom_subgroup_size(points) == BOTTOM_ORDER


def _internal_swap() -> np.ndarray:
    """The fixed alternating-form swap on the two top coordinates."""
    return np.hstack(
        [
            BOTTOM_ORDER + np.arange(BOTTOM_ORDER),
            np.arange(BOTTOM_ORDER),
        ]
    )


def fold_permutation(num_blocks: int, axis: int, *, swap_top: bool) -> np.ndarray:
    """Reflect both protograph halves and optionally swap top coordinates."""
    if num_blocks % 2:
        raise ValueError("the number of blocks must be even")
    half = num_blocks // 2
    internal = _internal_swap() if swap_top else np.arange(BLOCK_SIZE)
    block_images = tuple((axis - block) % half for block in range(half))
    full_images = (*block_images, *(half + block for block in block_images))
    return block_permutation(full_images, swap_top=swap_top)


def block_permutation(
    block_images: Sequence[int], *, swap_top: bool
) -> np.ndarray:
    """Lift a protograph-block permutation and optional top-coordinate swap."""
    internal = _internal_swap() if swap_top else np.arange(BLOCK_SIZE)
    return np.hstack(
        [image * BLOCK_SIZE + internal for image in block_images]
    ).astype(int)


def expected_zx_fold(num_blocks: int, family: Family) -> np.ndarray:
    """The contragredient fold implied by each generator relation."""
    half = num_blocks // 2
    if family in {"l4-self-dual", "l12-self-dual"}:
        return block_permutation(range(num_blocks), swap_top=True)
    axes: dict[Family, int] = {
        "l4-fold": 1,
        "l8-fold": 3,
        "l12-edge-fold": 5,
        "l12-vertex-fold": 2,
    }
    return fold_permutation(num_blocks, axes[family], swap_top=True)


def _permute_columns(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(matrix)
    transformed[:, permutation] = matrix
    return transformed


def is_zx_fold(code: codes.CSSCode, permutation: np.ndarray) -> bool:
    """Whether a physical qubit permutation exchanges X/Z check spaces."""
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    return bool(
        _gf2_rank(np.vstack([hz, _permute_columns(hx, permutation)]), code.field)
        == code.code_z.rank
        and _gf2_rank(np.vstack([hx, _permute_columns(hz, permutation)]), code.field)
        == code.code_x.rank
    )


def find_simple_zx_folds(code: codes.CSSCode) -> list[dict[str, Any]]:
    """Enumerate all half-reflections with identity or alternating-form swap."""
    half = code.num_qubits // BLOCK_SIZE // 2
    folds: list[dict[str, Any]] = []
    for axis in range(half):
        for swap_top in (False, True):
            permutation = fold_permutation(
                2 * half, axis, swap_top=swap_top
            )
            if is_zx_fold(code, permutation):
                folds.append(
                    {
                        "axis": axis,
                        "swap_top": swap_top,
                        "permutation": permutation,
                    }
                )
    return folds


def analyze_linear_record(
    record: dict[str, Any], *, screen_weight_four: bool = True
) -> dict[str, Any]:
    """Relift one saved record and apply cheap structural/low-weight filters."""
    family = infer_family(record)
    entries = candidate_entries(record)
    try:
        code = build_linear_code(entries, family)
    except Exception as error:
        return {
            "source_candidate_id": record.get("candidate_id"),
            "family": family,
            "accepted": False,
            "rejection_reasons": ["code_construction_failed"],
            "error": f"{type(error).__name__}: {error}",
        }
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    expected_fold = expected_zx_fold(code.num_qubits // BLOCK_SIZE, family)
    expected_fold_works = is_zx_fold(code, expected_fold)
    row_weights_x = np.count_nonzero(hx, axis=1)
    row_weights_z = np.count_nonzero(hz, axis=1)
    column_weights_x = np.count_nonzero(hx, axis=0)
    column_weights_z = np.count_nonzero(hz, axis=0)
    checks = {
        "css_orthogonal": bool(not np.any((hx @ hz.T) % 2)),
        "expected_ZX_fold": expected_fold_works,
        "bottom_translation_capacity_one_grid": code.num_qubits >= 32 * 6,
    }
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    low_weight = None
    if not rejection_reasons and screen_weight_four:
        low_weight = find_logical_up_to_weight_four(code)
        if low_weight is not None:
            rejection_reasons.append("logical_up_to_weight_four")
    return {
        "source_candidate_id": record.get("candidate_id"),
        "source_family": record.get("candidate", record).get("family"),
        "family": family,
        "entries": [
            [
                {"top": term.top, "x": term.x, "y": term.y}
                for term in entry
            ]
            for entry in entries
        ],
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "checks": checks,
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": code.code_x.rank,
        "rank_z": code.code_z.rank,
        "row_weights_x": sorted(set(row_weights_x.tolist())),
        "row_weights_z": sorted(set(row_weights_z.tolist())),
        "maximum_check_weight": int(max(row_weights_x.max(), row_weights_z.max())),
        "column_degrees_x": sorted(set(column_weights_x.tolist())),
        "column_degrees_z": sorted(set(column_weights_z.tolist())),
        "folds": [{"expected": True, "swap_top": True}]
        if expected_fold_works
        else [],
        "logical_up_to_weight_four": low_weight,
    }


@functools.lru_cache(maxsize=1)
def _linear_translation_permutations() -> tuple[np.ndarray, np.ndarray]:
    ring, top_members, xx, yy = linear_group_data()
    identity = top_members[0]
    matrices = [
        np.asarray(ring.group.lift(identity @ member), dtype=np.uint8)
        for member in (xx, yy)
    ]
    return tuple(np.argmax(matrix, axis=1).astype(int) for matrix in matrices)  # type: ignore[return-value]


def _lift_internal_permutation(permutation: np.ndarray, num_blocks: int) -> np.ndarray:
    return np.hstack(
        [block * BLOCK_SIZE + permutation for block in range(num_blocks)]
    )


@functools.lru_cache(maxsize=None)
def linear_internal_translation_orbits(
    num_blocks: int,
) -> tuple[tuple[int, ...], ...]:
    """The `2L` physical fibres under the bottom `C8 x C4` action."""
    perm_x, perm_y = _linear_translation_permutations()
    full_x = _lift_internal_permutation(perm_x, num_blocks)
    full_y = _lift_internal_permutation(perm_y, num_blocks)
    remaining = set(range(num_blocks * BLOCK_SIZE))
    orbits: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        orbit = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for permutation in (full_x, full_y):
                image = int(permutation[current])
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        remaining -= orbit
        orbits.append(tuple(sorted(orbit)))
    if len(orbits) != 2 * num_blocks or any(
        len(orbit) != BOTTOM_ORDER for orbit in orbits
    ):
        raise RuntimeError("unexpected two-dimensional translation fibres")
    return tuple(orbits)


def linear_logical_translation_orbit(
    seed: np.ndarray, *, num_blocks: int
) -> np.ndarray:
    """Translate one support through its full `C8 x C4` orbit."""
    perm_x, perm_y = _linear_translation_permutations()
    full_x = _lift_internal_permutation(perm_x, num_blocks)
    full_y = _lift_internal_permutation(perm_y, num_blocks)

    def apply(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        output = np.zeros_like(vector)
        output[permutation] = vector
        return output

    rows: list[np.ndarray] = []
    translated_x = np.asarray(seed, dtype=np.uint8).copy()
    for _xx in range(BOTTOM_X_ORDER):
        translated_y = translated_x.copy()
        for _yy in range(BOTTOM_Y_ORDER):
            rows.append(translated_y)
            translated_y = apply(translated_y, full_y)
        translated_x = apply(translated_x, full_x)
    return np.asarray(rows, dtype=np.uint8)


def _canonical_linear_translation_support(
    support: Sequence[int], *, num_qubits: int, num_blocks: int
) -> tuple[int, ...]:
    seed = np.zeros(num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in linear_logical_translation_orbit(seed, num_blocks=num_blocks)
    )


def find_linear_zero_syndrome_by_tanner_search(
    code: codes.CSSCode,
    *,
    weight: int,
    max_nodes: int = 1_000_000,
    require_graph_support: bool = False,
    excluded_translation_orbits: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """Sparse exact-weight Z-kernel search normalized by bottom translation."""
    check = np.asarray(code.matrix_x, dtype=np.uint8)
    stabilizer = np.asarray(code.matrix_z, dtype=np.uint8)
    num_checks, num_qubits = check.shape
    if num_qubits % BLOCK_SIZE:
        raise ValueError("code is not compatible with the two-dimensional lift")
    num_blocks = num_qubits // BLOCK_SIZE
    internal_orbits = linear_internal_translation_orbits(num_blocks)
    qubit_to_internal = np.empty(num_qubits, dtype=int)
    for internal, orbit in enumerate(internal_orbits):
        qubit_to_internal[list(orbit)] = internal
    first_qubits = [orbit[0] for orbit in internal_orbits]
    column_keys = [
        int.from_bytes(np.packbits(check[:, index]).tobytes())
        for index in range(num_qubits)
    ]
    columns_by_key: dict[int, list[int]] = {}
    for index, key in enumerate(column_keys):
        columns_by_key.setdefault(key, []).append(index)
    rows_by_packed_bit: dict[int, np.ndarray] = {}
    for row in range(num_checks):
        unit = np.zeros(num_checks, dtype=np.uint8)
        unit[row] = 1
        rows_by_packed_bit[int.from_bytes(np.packbits(unit).tobytes())] = (
            np.flatnonzero(check[row])
        )
    excluded = {
        _canonical_linear_translation_support(
            support, num_qubits=num_qubits, num_blocks=num_blocks
        )
        for support in excluded_translation_orbits
    }
    maximum_column_weight = int(np.count_nonzero(check, axis=0).max(initial=0))
    nodes = 0
    cutoff = False
    started = time.perf_counter()

    def admissible(selected: frozenset[int]) -> tuple[int, ...] | None:
        support = tuple(sorted(selected))
        canonical = _canonical_linear_translation_support(
            support, num_qubits=num_qubits, num_blocks=num_blocks
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
            return admissible(selected) if syndrome == 0 else None
        if syndrome == 0:
            return None
        if syndrome.bit_count() > maximum_column_weight * remaining:
            return None
        if remaining == 1:
            for qubit in columns_by_key.get(syndrome, ()):
                if qubit in selected:
                    continue
                internal = int(qubit_to_internal[qubit])
                if require_graph_support and any(
                    int(qubit_to_internal[chosen]) == internal
                    for chosen in selected
                ):
                    continue
                result = admissible(selected | {qubit})
                if result is not None:
                    return result
            return None
        odd_bits: list[int] = []
        value = syndrome
        while value:
            bit = value & -value
            odd_bits.append(bit)
            value ^= bit
        branch: np.ndarray | None = None
        for bit in odd_bits:
            available = np.asarray(
                [qubit for qubit in rows_by_packed_bit[bit] if int(qubit) not in selected],
                dtype=int,
            )
            if branch is None or len(available) < len(branch):
                branch = available
        assert branch is not None
        for raw_qubit in branch:
            qubit = int(raw_qubit)
            internal = int(qubit_to_internal[qubit])
            if require_graph_support and any(
                int(qubit_to_internal[chosen]) == internal
                for chosen in selected
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
    result: dict[str, Any] = {
        "weight": weight,
        "support": list(support) if support is not None else None,
        "search_exhaustive": not cutoff,
        "nodes": nodes,
        "seconds": round(time.perf_counter() - started, 6),
    }
    if support is None:
        return result
    solution = np.zeros(num_qubits, dtype=int)
    solution[list(support)] = 1
    stabilizer_dual = np.asarray(stabilizer, dtype=int).view(code.field).null_space()
    occupancies = [
        int(np.count_nonzero(solution[list(orbit)]))
        for orbit in internal_orbits
    ]
    result.update(
        {
            "is_nontrivial_logical": bool(
                np.any(stabilizer_dual @ solution.view(code.field))
            ),
            "graph_supported": max(occupancies) <= 1,
            "internal_occupancies": occupancies,
        }
    )
    return result


def analyze_linear_translation_seed(
    code: codes.CSSCode,
    support: Sequence[int],
    *,
    fold: Sequence[int],
) -> dict[str, Any]:
    """Certify one translated grid and its folded X orbit."""
    num_blocks = code.num_qubits // BLOCK_SIZE
    seed = np.zeros(code.num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    internal = linear_internal_translation_orbits(num_blocks)
    occupancies = [int(np.count_nonzero(seed[list(orbit)])) for orbit in internal]
    z_orbit = linear_logical_translation_orbit(seed, num_blocks=num_blocks)
    x_orbit = _permute_columns(z_orbit, np.asarray(fold, dtype=int))
    pairing = (z_orbit @ x_orbit.T) % 2
    return {
        "seed_support": list(map(int, support)),
        "weight": len(support),
        "internal_orbits": np.flatnonzero(occupancies).astype(int).tolist(),
        "graph_supported": max(occupancies) <= 1,
        "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
        "z_orbit_in_kernel": bool(
            not np.any((np.asarray(code.matrix_x, dtype=np.uint8) @ z_orbit.T) % 2)
        ),
        "x_orbit_in_kernel": bool(
            not np.any((np.asarray(code.matrix_z, dtype=np.uint8) @ x_orbit.T) % 2)
        ),
        "orbit_rank_mod_stabilizers": int(
            _gf2_rank(
                np.vstack([np.asarray(code.matrix_z, dtype=np.uint8), z_orbit]),
                code.field,
            )
            - code.code_z.rank
        ),
        "zx_pairing_rank": _gf2_rank(pairing, code.field),
    }


def analyze_linear_grid_combinations(
    code: codes.CSSCode,
    grids: Sequence[dict[str, Any]],
    *,
    fold: Sequence[int],
    maximum_grids: int,
) -> dict[str, Any]:
    """Test all support-disjoint grid sets for independence and joint ZX rank."""
    num_blocks = code.num_qubits // BLOCK_SIZE
    arrays: list[np.ndarray] = []
    for grid in grids:
        seed = np.zeros(code.num_qubits, dtype=np.uint8)
        seed[grid["seed_support"]] = 1
        arrays.append(
            linear_logical_translation_orbit(seed, num_blocks=num_blocks)
        )
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    coordinates = [(orbit @ logical_x.T) % 2 for orbit in arrays]
    folded = [
        _permute_columns(orbit, np.asarray(fold, dtype=int))
        for orbit in arrays
    ]
    cross = {
        (left, right): (arrays[left] @ folded[right].T) % 2
        for left in range(len(arrays))
        for right in range(len(arrays))
    }
    internal = [set(map(int, grid["internal_orbits"])) for grid in grids]
    summaries: dict[str, Any] = {}
    hits: list[dict[str, Any]] = []
    for size in range(1, maximum_grids + 1):
        tested = disjoint = independent = full = 0
        best_logical_rank = best_pairing_rank = 0
        for indices in itertools.combinations(range(len(grids)), size):
            tested += 1
            used: set[int] = set()
            overlap = False
            for index in indices:
                if used & internal[index]:
                    overlap = True
                    break
                used.update(internal[index])
            if overlap:
                continue
            disjoint += 1
            logical_rank = _gf2_rank(
                np.vstack([coordinates[index] for index in indices]),
                code.field,
            )
            pairing = np.block(
                [
                    [cross[(left, right)] for right in indices]
                    for left in indices
                ]
            ).astype(np.uint8)
            pairing_rank = _gf2_rank(pairing, code.field)
            best_logical_rank = max(best_logical_rank, logical_rank)
            best_pairing_rank = max(best_pairing_rank, pairing_rank)
            if logical_rank != BOTTOM_ORDER * size:
                continue
            independent += 1
            if pairing_rank != BOTTOM_ORDER * size:
                continue
            full += 1
            row_weights = np.count_nonzero(pairing, axis=1)
            column_weights = np.count_nonzero(pairing, axis=0)
            hits.append(
                {
                    "num_grids": size,
                    "indices": list(indices),
                    "seed_supports": [grids[index]["seed_support"] for index in indices],
                    "internal_orbits": [grids[index]["internal_orbits"] for index in indices],
                    "combined_rank_mod_stabilizers": logical_rank,
                    "combined_zx_pairing_rank": pairing_rank,
                    "pairing_is_permutation": bool(
                        np.all(row_weights == 1) and np.all(column_weights == 1)
                    ),
                }
            )
        summaries[str(size)] = {
            "combinations_tested": tested,
            "disjoint_combinations": disjoint,
            "independent_combinations": independent,
            "full_rank_combinations": full,
            "best_logical_rank": best_logical_rank,
            "best_pairing_rank": best_pairing_rank,
        }
    return {"by_num_grids": summaries, "full_rank_hits": hits}
