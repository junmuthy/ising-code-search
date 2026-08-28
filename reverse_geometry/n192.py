"""Reverse-order ``n=192`` search over natural ``S3 x C8 x C4``.

The minimal two-block layout contains six physical translation sheets:
two data halves times the three coordinates of the natural permutation
representation of ``S3``.  Logical supports are designed first.  Only after
their two-batch disjointness and permutation ZX pairing are established do we
construct the complete linear space of fold-tied polynomial checks that
annihilate them.
"""

from __future__ import annotations

import functools
import itertools
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gala_search.single_row import gf2_nullspace, gf2_rank, gf2_rref

X_ORDER = 8
Y_ORDER = 4
LATTICE_ORDER = X_ORDER * Y_ORDER
NUM_HALVES = 2
TOP_DEGREE = 3
NUM_SHEETS = NUM_HALVES * TOP_DEGREE
BLOCK_SIZE = TOP_DEGREE * LATTICE_ORDER
NUM_QUBITS = NUM_HALVES * BLOCK_SIZE
TARGET_LOGICALS = LATTICE_ORDER
TARGET_DISTANCE = 7
MAXIMUM_CHECK_WEIGHT = 12
SCHEMA_VERSION = 1


def lattice_index(xx: int, yy: int) -> int:
    return (xx % X_ORDER) * Y_ORDER + yy % Y_ORDER


def lattice_coordinate(index: int) -> tuple[int, int]:
    return divmod(index, Y_ORDER)


def parity(index: int) -> int:
    xx, yy = lattice_coordinate(index)
    return (xx + yy) % 2


def _permutation_matrix(images: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((len(images), len(images)), dtype=np.uint8)
    matrix[np.asarray(images, dtype=int), np.arange(len(images))] = 1
    return matrix


@dataclass(frozen=True)
class NaturalS3Fold:
    """Translation-commuting fold on the two halves and three top sheets."""

    top_images: tuple[int, int, int]
    block_images: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (0, 1),
        (0, 1),
        (0, 1),
    )

    def __post_init__(self) -> None:
        if sorted(self.top_images) != list(range(TOP_DEGREE)):
            raise ValueError("top_images must be a permutation of three coordinates")
        if len(self.block_images) != TOP_DEGREE or any(
            sorted(images) != list(range(NUM_HALVES))
            for images in self.block_images
        ):
            raise ValueError("every top coordinate needs a two-block permutation")

    @property
    def sheet_images(self) -> tuple[int, ...]:
        return tuple(
            self.block_images[top][block] * TOP_DEGREE + self.top_images[top]
            for block in range(NUM_HALVES)
            for top in range(TOP_DEGREE)
        )

    @property
    def fixed_sheets(self) -> tuple[int, ...]:
        return tuple(
            sheet
            for sheet, image in enumerate(self.sheet_images)
            if sheet == image
        )

    @property
    def order(self) -> int:
        images = self.sheet_images
        visited: set[int] = set()
        lengths = []
        for start in range(NUM_SHEETS):
            if start in visited:
                continue
            current = start
            length = 0
            while current not in visited:
                visited.add(current)
                length += 1
                current = images[current]
            lengths.append(length)
        return int(np.lcm.reduce(lengths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_images": list(self.top_images),
            "block_images": [list(images) for images in self.block_images],
            "sheet_images": list(self.sheet_images),
            "fixed_sheets": list(self.fixed_sheets),
            "order": self.order,
        }


def standard_transposition_fold() -> NaturalS3Fold:
    """Fix top coordinate zero and exchange coordinates one and two."""
    return NaturalS3Fold((0, 2, 1))


def seed_array() -> np.ndarray:
    return np.zeros((NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER), dtype=np.uint8)


def translate_seed(seed: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(seed, dx % X_ORDER, axis=2), dy % Y_ORDER, axis=3)


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            translate_seed(seed, xx, yy).reshape(NUM_QUBITS)
            for xx in range(X_ORDER)
            for yy in range(Y_ORDER)
        ],
        dtype=np.uint8,
    )


def fold_seed(seed: np.ndarray, fold: NaturalS3Fold) -> np.ndarray:
    output = np.zeros_like(seed)
    for block in range(NUM_HALVES):
        for top in range(TOP_DEGREE):
            new_block = fold.block_images[top][block]
            new_top = fold.top_images[top]
            output[new_block, new_top] = seed[block, top]
    return output


def pairing_matrix(seed: np.ndarray, fold: NaturalS3Fold) -> np.ndarray:
    zz = translation_orbit(seed)
    xx = translation_orbit(fold_seed(seed, fold))
    return (zz @ xx.T) % 2


def batch_disjointness(seed: np.ndarray) -> dict[str, Any]:
    orbit = translation_orbit(seed)
    sites = [
        (xx, yy) for xx in range(X_ORDER) for yy in range(Y_ORDER)
    ]
    rows = {
        color: [
            index
            for index, (xx, yy) in enumerate(sites)
            if (xx + yy) % 2 == color
        ]
        for color in (0, 1)
    }
    maxima = {
        str(color): int(np.sum(orbit[indices], axis=0).max(initial=0))
        for color, indices in rows.items()
    }
    return {
        "batch_sizes": {str(color): len(indices) for color, indices in rows.items()},
        "maximum_multiplicity": maxima,
        "disjoint_within_each_batch": all(value <= 1 for value in maxima.values()),
        "all_at_once_disjoint": bool(np.sum(orbit, axis=0).max(initial=0) <= 1),
        "used_physical_qubits_by_batch": {
            str(color): int(np.count_nonzero(np.sum(orbit[indices], axis=0)))
            for color, indices in rows.items()
        },
    }


def analyze_seed(seed: np.ndarray, fold: NaturalS3Fold) -> dict[str, Any]:
    seed = np.asarray(seed, dtype=np.uint8)
    expected_shape = (NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER)
    if seed.shape != expected_shape:
        raise ValueError(f"expected seed shape {expected_shape}")
    zz = translation_orbit(seed)
    pairing = pairing_matrix(seed, fold)
    row_weights = np.count_nonzero(pairing, axis=1)
    column_weights = np.count_nonzero(pairing, axis=0)
    occupancies = np.count_nonzero(seed, axis=(2, 3))
    return {
        "weight": int(np.count_nonzero(seed)),
        "support": np.flatnonzero(seed.reshape(NUM_QUBITS)).astype(int).tolist(),
        "sheet_occupancies": occupancies.astype(int).reshape(-1).tolist(),
        "fold_invariant_seed": bool(np.array_equal(seed, fold_seed(seed, fold))),
        "translation_orbit_rank": gf2_rank(zz),
        "batching": batch_disjointness(seed),
        "zx_pairing_rank": gf2_rank(pairing),
        "zx_pairing_weight": int(np.count_nonzero(pairing)),
        "pairing_is_permutation": bool(
            np.all(row_weights == 1) and np.all(column_weights == 1)
        ),
        "pairing_is_identity": bool(
            np.array_equal(pairing, np.eye(TARGET_LOGICALS, dtype=np.uint8))
        ),
        "pairing_kernel": pairing[0].astype(int).tolist(),
        "fold": fold.to_dict(),
    }


def cancellation_witness(
    *,
    fixed_site: int = 0,
    first_pair_sites: tuple[int, int] = (0, 1),
    second_pair_site: int = 0,
    fixed_block: int = 0,
) -> np.ndarray:
    """Construct a weight-seven identity-pairing support.

    One singleton on a fold-fixed sheet supplies one diagonal intersection.
    Identical supports on each exchanged sheet pair contribute twice and
    cancel over ``GF(2)``.
    """
    if parity(first_pair_sites[0]) == parity(first_pair_sites[1]):
        raise ValueError("the two-point sheet support must use opposite parities")
    seed = seed_array()
    fx, fy = lattice_coordinate(fixed_site)
    seed[fixed_block, 0, fx, fy] = 1
    for site in first_pair_sites:
        xx, yy = lattice_coordinate(site)
        seed[0, 1, xx, yy] = 1
        seed[0, 2, xx, yy] = 1
    sx, sy = lattice_coordinate(second_pair_site)
    seed[1, 1, sx, sy] = 1
    seed[1, 2, sx, sy] = 1
    return seed


def top_parity_projections(seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project each physical half onto the natural module's trivial quotient."""
    return tuple(
        np.bitwise_xor.reduce(seed[half], axis=0)
        for half in range(NUM_HALVES)
    )  # type: ignore[return-value]


def random_batched_seed(rng: random.Random, *, weight: int) -> np.ndarray:
    if not 1 <= weight <= 2 * NUM_SHEETS:
        raise ValueError("batched support weight must lie between one and twelve")
    slots = list(itertools.product(range(NUM_SHEETS), range(2)))
    selected = rng.sample(slots, weight)
    sites = {
        color: [site for site in range(LATTICE_ORDER) if parity(site) == color]
        for color in (0, 1)
    }
    seed = seed_array()
    for sheet, color in selected:
        block, top = divmod(sheet, TOP_DEGREE)
        xx, yy = lattice_coordinate(rng.choice(sites[color]))
        seed[block, top, xx, yy] = 1
    return seed


def _support_sites(seed: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            lattice_index(int(xx), int(yy))
            for xx, yy in np.argwhere(seed[block, top])
        )
        for block in range(NUM_HALVES)
        for top in range(TOP_DEGREE)
    )


def pairing_kernel_mask(seed: np.ndarray, fold: NaturalS3Fold) -> int:
    """Return the 32-bit translation-correlation kernel of one seed."""
    left = _support_sites(seed)
    right = _support_sites(fold_seed(seed, fold))
    kernel = 0
    for sheet in range(NUM_SHEETS):
        for left_site in left[sheet]:
            lx, ly = lattice_coordinate(left_site)
            for right_site in right[sheet]:
                rx, ry = lattice_coordinate(right_site)
                delta = lattice_index(lx - rx, ly - ry)
                kernel ^= 1 << delta
    return kernel


def search_projected_geometry_witnesses(
    *,
    random_seed: int,
    trials: int,
    target_witnesses: int,
    weights: Sequence[int] = tuple(range(7, 13)),
) -> dict[str, Any]:
    """Search general batched seeds with nonzero projections in both halves."""
    if trials < 1 or target_witnesses < 1:
        raise ValueError("trial and target counts must be positive")
    if not weights or any(not 7 <= weight <= 12 for weight in weights):
        raise ValueError("geometry weights must lie between seven and twelve")
    rng = random.Random(random_seed)
    fold = standard_transposition_fold()
    rank_kernel_weights: Counter[int] = Counter()
    projection_survivors = 0
    witnesses: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    started = time.perf_counter()
    for trial in range(trials):
        weight = weights[trial % len(weights)]
        candidate = random_batched_seed(rng, weight=weight)
        projections = top_parity_projections(candidate)
        projection_weights = [int(np.count_nonzero(item)) for item in projections]
        if not all(projection_weights):
            continue
        projection_survivors += 1
        kernel = pairing_kernel_mask(candidate, fold)
        kernel_weight = kernel.bit_count()
        rank_kernel_weights[kernel_weight] += 1
        if kernel_weight != 1:
            continue
        support = tuple(np.flatnonzero(candidate.reshape(NUM_QUBITS)).tolist())
        if support in seen:
            continue
        seen.add(support)
        analysis = analyze_seed(candidate, fold)
        if not analysis["pairing_is_permutation"]:
            raise AssertionError("one-term correlation kernel was not a permutation")
        analysis["top_projection_weights"] = projection_weights
        analysis["pairing_translation"] = int(kernel.bit_length() - 1)
        witnesses.append(analysis)
        if len(witnesses) >= target_witnesses:
            break
    return {
        "random_seed": random_seed,
        "trials_requested": trials,
        "trials_completed": trial + 1 if trials else 0,
        "weights": list(weights),
        "projection_survivors": projection_survivors,
        "kernel_weight_counts": {
            str(key): value for key, value in sorted(rank_kernel_weights.items())
        },
        "target_witnesses": target_witnesses,
        "witnesses_found": len(witnesses),
        "witnesses": witnesses,
        "seconds": round(time.perf_counter() - started, 6),
    }


def sample_cancellation_witnesses(
    *, seed: int, count: int
) -> tuple[dict[str, Any], ...]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    by_parity = {
        color: [site for site in range(LATTICE_ORDER) if parity(site) == color]
        for color in (0, 1)
    }
    fold = standard_transposition_fold()
    witnesses: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    while len(witnesses) < count:
        pair = (rng.choice(by_parity[0]), rng.choice(by_parity[1]))
        candidate = cancellation_witness(
            fixed_site=rng.randrange(LATTICE_ORDER),
            first_pair_sites=pair,
            second_pair_site=rng.randrange(LATTICE_ORDER),
            fixed_block=rng.randrange(NUM_HALVES),
        )
        support = tuple(np.flatnonzero(candidate.reshape(NUM_QUBITS)).tolist())
        if support in seen:
            continue
        seen.add(support)
        analysis = analyze_seed(candidate, fold)
        if not (
            analysis["batching"]["disjoint_within_each_batch"]
            and analysis["translation_orbit_rank"] == TARGET_LOGICALS
            and analysis["pairing_is_identity"]
        ):
            raise AssertionError("constructed cancellation witness failed")
        witnesses.append(analysis)
    return tuple(witnesses)


@functools.lru_cache(maxsize=1)
def top_permutations() -> tuple[tuple[int, int, int], ...]:
    return tuple(itertools.permutations(range(TOP_DEGREE)))


@functools.lru_cache(maxsize=None)
def bottom_shift(dx: int, dy: int) -> np.ndarray:
    matrix = np.zeros((LATTICE_ORDER, LATTICE_ORDER), dtype=np.uint8)
    for xx in range(X_ORDER):
        for yy in range(Y_ORDER):
            old = lattice_index(xx, yy)
            new = lattice_index(xx + dx, yy + dy)
            matrix[new, old] = 1
    return matrix


@functools.lru_cache(maxsize=None)
def represented_monomial(top: int, dx: int, dy: int) -> np.ndarray:
    top_matrix = _permutation_matrix(top_permutations()[top])
    return np.kron(top_matrix, bottom_shift(dx, dy)).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def represented_terms() -> np.ndarray:
    return np.asarray(
        [
            represented_monomial(top, xx, yy)
            for top in range(len(top_permutations()))
            for xx in range(X_ORDER)
            for yy in range(Y_ORDER)
        ],
        dtype=np.uint8,
    )


def fold_internal_matrix(fold: NaturalS3Fold) -> np.ndarray:
    if any(images != (0, 1) for images in fold.block_images):
        raise ValueError("fold-tied polynomial probe currently requires fixed halves")
    return np.kron(
        _permutation_matrix(fold.top_images),
        np.eye(LATTICE_ORDER, dtype=np.uint8),
    )


def fold_tied_checks_from_f(
    f_matrix: np.ndarray,
    fold: NaturalS3Fold,
    *,
    check_top_images: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the forward-fold partner ``G=Q F^T R^T``."""
    qq = fold_internal_matrix(fold)
    if not np.array_equal(qq, qq.T):
        raise ValueError("the initial fold-tied probe requires an involution")
    check_images = check_top_images or fold.top_images
    rr = np.kron(
        _permutation_matrix(check_images),
        np.eye(LATTICE_ORDER, dtype=np.uint8),
    )
    gg = (qq @ f_matrix.T @ rr.T) % 2
    matrix_x = np.hstack([f_matrix, gg]).astype(np.uint8)
    matrix_z = np.hstack([gg.T, f_matrix.T]).astype(np.uint8)
    return matrix_x, matrix_z


@functools.lru_cache(maxsize=None)
def fold_tied_basis_checks(
    check_top_images: tuple[int, int, int] = (0, 2, 1),
) -> tuple[np.ndarray, np.ndarray]:
    fold = standard_transposition_fold()
    checks_x = []
    checks_z = []
    for term in represented_terms():
        matrix_x, matrix_z = fold_tied_checks_from_f(
            term, fold, check_top_images=check_top_images
        )
        checks_x.append(matrix_x)
        checks_z.append(matrix_z)
    return np.asarray(checks_x), np.asarray(checks_z)


def fold_tied_annihilator(seed: np.ndarray) -> dict[str, Any]:
    """Compute the full linear generator space preserving one logical grid."""
    started = time.perf_counter()
    fold = standard_transposition_fold()
    analysis = analyze_seed(seed, fold)
    if not analysis["pairing_is_permutation"]:
        raise ValueError("annihilator input must have permutation ZX pairing")
    zz = translation_orbit(seed)
    checks_x, checks_z = fold_tied_basis_checks()
    constraint_columns = np.asarray(
        [(matrix @ zz.T % 2).reshape(-1) for matrix in checks_x],
        dtype=np.uint8,
    )
    constraints = constraint_columns.T
    constraint_rank = gf2_rank(constraints)
    nullspace = gf2_nullspace(constraints)
    lifted_x = []
    lifted_z = []
    for vector in nullspace:
        selected = np.flatnonzero(vector)
        lifted_x.append(
            np.bitwise_xor.reduce(checks_x[selected], axis=0)
            if len(selected)
            else np.zeros((BLOCK_SIZE, NUM_QUBITS), dtype=np.uint8)
        )
        lifted_z.append(
            np.bitwise_xor.reduce(checks_z[selected], axis=0)
            if len(selected)
            else np.zeros((BLOCK_SIZE, NUM_QUBITS), dtype=np.uint8)
        )
    capacity_x_raw = gf2_rank(np.vstack(lifted_x)) if lifted_x else 0
    capacity_z_raw = gf2_rank(np.vstack(lifted_z)) if lifted_z else 0
    capacity_x = min(BLOCK_SIZE, capacity_x_raw)
    capacity_z = min(BLOCK_SIZE, capacity_z_raw)
    required_rank = (NUM_QUBITS - TARGET_LOGICALS) // 2

    qq = fold_internal_matrix(fold)
    data_fold = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    data_fold[:BLOCK_SIZE, :BLOCK_SIZE] = qq
    data_fold[BLOCK_SIZE:, BLOCK_SIZE:] = qq
    relation_holds = all(
        np.array_equal(matrix_z, (qq @ matrix_x @ data_fold.T) % 2)
        for matrix_x, matrix_z in zip(checks_x[:8], checks_z[:8], strict=True)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "natural-s3-minimal-two-block-fold-tied",
        "n": NUM_QUBITS,
        "target_logicals": TARGET_LOGICALS,
        "target_rank_per_css_type_for_k32": required_rank,
        "coefficient_variables": int(checks_x.shape[0]),
        "constraint_equations": int(constraints.shape[0]),
        "constraint_rank": constraint_rank,
        "constraint_nullity": int(len(nullspace)),
        "rank_capacity_x_upper_bound": capacity_x,
        "rank_capacity_z_upper_bound": capacity_z,
        "rank_capacity_survives_k32_target": min(capacity_x, capacity_z) >= required_rank,
        "exact_displayed_zx_relation": relation_holds,
        "css_not_yet_imposed": True,
        "maximum_check_weight_target": MAXIMUM_CHECK_WEIGHT,
        "seconds": round(time.perf_counter() - started, 6),
    }


def _fold_tied_constraint_data(
    seed: np.ndarray,
    *,
    check_top_images: tuple[int, int, int] = (0, 2, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zz = translation_orbit(seed)
    checks_x, checks_z = fold_tied_basis_checks(check_top_images)
    fold = standard_transposition_fold()
    qq = fold_internal_matrix(fold)
    rr = np.kron(
        _permutation_matrix(check_top_images),
        np.eye(LATTICE_ORDER, dtype=np.uint8),
    )
    data_fold = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    data_fold[:BLOCK_SIZE, :BLOCK_SIZE] = qq
    data_fold[BLOCK_SIZE:, BLOCK_SIZE:] = qq
    constraint_columns = np.asarray(
        [
            np.concatenate(
                (
                    (matrix_x @ zz.T % 2).reshape(-1),
                    (matrix_z ^ (rr @ matrix_x @ data_fold.T) % 2).reshape(-1),
                )
            )
            for matrix_x, matrix_z in zip(checks_x, checks_z, strict=True)
        ],
        dtype=np.uint8,
    )
    reduced, _pivots = gf2_rref(constraint_columns.T)
    return reduced, checks_x, checks_z, zz


def _serialize_term(index: int) -> dict[str, int]:
    bottom = index % LATTICE_ORDER
    top = index // LATTICE_ORDER
    xx, yy = lattice_coordinate(bottom)
    return {"top": top, "x": xx, "y": yy}


def coefficient_index(term: dict[str, int]) -> int:
    return int(term["top"]) * LATTICE_ORDER + lattice_index(
        int(term["x"]), int(term["y"])
    )


def matrices_from_terms(
    terms: Sequence[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([coefficient_index(term) for term in terms], dtype=int)
    if not len(indices) or len(set(indices.tolist())) != len(indices):
        raise ValueError("generator terms must be nonempty and distinct")
    checks_x, checks_z = fold_tied_basis_checks()
    return (
        np.bitwise_xor.reduce(checks_x[indices], axis=0),
        np.bitwise_xor.reduce(checks_z[indices], axis=0),
    )


def tanner_connected(matrix_x: np.ndarray, matrix_z: np.ndarray) -> bool:
    """Whether the combined nonzero-check Tanner graph is connected."""
    import networkx as nx

    num_qubits = matrix_x.shape[1]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    check_node = num_qubits
    for matrix in (matrix_x, matrix_z):
        for row in matrix:
            support = np.flatnonzero(row)
            if not len(support):
                continue
            graph.add_node(check_node)
            graph.add_edges_from((check_node, int(qubit)) for qubit in support)
            check_node += 1
    return nx.is_connected(graph)


def _generator_record(
    coefficients: np.ndarray,
    checks_x: np.ndarray,
    checks_z: np.ndarray,
    zz: np.ndarray,
) -> dict[str, Any]:
    selected = np.flatnonzero(coefficients)
    matrix_x = np.bitwise_xor.reduce(checks_x[selected], axis=0)
    matrix_z = np.bitwise_xor.reduce(checks_z[selected], axis=0)
    css = bool(not np.any((matrix_x @ matrix_z.T) % 2))
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    column_weights_x = np.count_nonzero(matrix_x, axis=0)
    column_weights_z = np.count_nonzero(matrix_z, axis=0)
    maximum_check_weight = int(
        max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
    )
    no_zero_columns = bool(
        np.all(column_weights_x > 0) and np.all(column_weights_z > 0)
    )
    target_rank = (
        gf2_rank(np.vstack([matrix_z, zz])) - rank_z if css else None
    )
    return {
        "coefficient_weight": int(len(selected)),
        "terms": [_serialize_term(int(index)) for index in selected],
        "binary_f_row_weights": sorted(
            set(
                np.count_nonzero(
                    np.bitwise_xor.reduce(represented_terms()[selected], axis=0),
                    axis=1,
                ).astype(int).tolist()
            )
        ),
        "css_orthogonal": css,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "k_if_css": NUM_QUBITS - rank_x - rank_z if css else None,
        "target_grid_rank_mod_z_stabilizers": target_rank,
        "row_weights_x": sorted(set(row_weights_x.astype(int).tolist())),
        "row_weights_z": sorted(set(row_weights_z.astype(int).tolist())),
        "maximum_check_weight": maximum_check_weight,
        "column_degrees_x": sorted(set(column_weights_x.astype(int).tolist())),
        "column_degrees_z": sorted(set(column_weights_z.astype(int).tolist())),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "zero_columns_x": int(np.count_nonzero(column_weights_x == 0)),
        "zero_columns_z": int(np.count_nonzero(column_weights_z == 0)),
        "no_zero_columns": no_zero_columns,
        "logical_orbit_in_x_kernel": bool(not np.any((matrix_x @ zz.T) % 2)),
        "structural_pilot_hit": bool(
            css
            and target_rank == TARGET_LOGICALS
            and maximum_check_weight <= MAXIMUM_CHECK_WEIGHT
            and no_zero_columns
        ),
    }


def search_sparse_fold_tied_generators(
    seed: np.ndarray,
    *,
    minimum_terms: int = 1,
    maximum_terms: int = 6,
    solutions: int = 100,
    seconds_per_solution: float = 5,
    random_seed: int = 0,
    require_no_zero_columns: bool = False,
    require_nonzero_top_augmentation: bool = False,
) -> dict[str, Any]:
    """Enumerate sparse annihilator vectors with repeated exact MILPs.

    A coefficient weight of at most six bounds every displayed check row by
    twelve because ``H_X=[F|QF^TQ]``.  CSS commutation is evaluated exactly
    after each linear annihilator solution.  This is a bounded pilot rather
    than an exhaustive solution of the remaining quadratic CSS constraint.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, eye, hstack, vstack

    if not 1 <= minimum_terms <= maximum_terms <= 6:
        raise ValueError("term bounds must satisfy 1 <= minimum <= maximum <= 6")
    if solutions < 1 or seconds_per_solution <= 0:
        raise ValueError("solution count and time limit must be positive")
    started = time.perf_counter()
    rows, checks_x, checks_z, zz = _fold_tied_constraint_data(seed)
    num_rows, num_coefficients = rows.shape
    probe_data: np.ndarray | None = None
    num_output_bits = 0
    if require_no_zero_columns:
        probe_columns = [
            half * BLOCK_SIZE + top * LATTICE_ORDER
            for half in range(NUM_HALVES)
            for top in range(TOP_DEGREE)
        ]
        probe_data = np.asarray(
            [
                np.concatenate(
                    [
                        matrix_x[:, probe_columns].T.reshape(-1),
                        matrix_z[:, probe_columns].T.reshape(-1),
                    ]
                )
                for matrix_x, matrix_z in zip(checks_x, checks_z, strict=True)
            ],
            dtype=np.uint8,
        ).T
        num_output_bits = probe_data.shape[0]
    num_augmentation_bits = LATTICE_ORDER if require_nonzero_top_augmentation else 0
    num_variables = (
        num_coefficients
        + num_rows
        + 2 * num_output_bits
        + 2 * num_augmentation_bits
    )
    parity_matrix = hstack(
        [
            csr_matrix(rows.astype(float)),
            -2 * eye(num_rows, format="csr"),
            csr_matrix(
                (num_rows, 2 * num_output_bits + 2 * num_augmentation_bits)
            ),
        ],
        format="csr",
    )
    constraint_blocks = [parity_matrix]
    lower_blocks = [np.zeros(num_rows)]
    upper_blocks = [np.zeros(num_rows)]

    weight_row = np.zeros((1, num_variables), dtype=float)
    weight_row[0, :num_coefficients] = 1
    constraint_blocks.append(csr_matrix(weight_row))
    lower_blocks.append(np.asarray([float(minimum_terms)]))
    upper_blocks.append(np.asarray([float(maximum_terms)]))

    # The six natural S3 permutation matrices have one representation-kernel
    # relation at every bottom monomial.  Exclude those zero-lift weight-six
    # solutions explicitly.
    kernel_rows = np.zeros(
        (LATTICE_ORDER, num_variables), dtype=float
    )
    for bottom in range(LATTICE_ORDER):
        for top in range(6):
            kernel_rows[bottom, top * LATTICE_ORDER + bottom] = 1
    constraint_blocks.append(csr_matrix(kernel_rows))
    lower_blocks.append(np.full(LATTICE_ORDER, -np.inf))
    upper_blocks.append(np.full(LATTICE_ORDER, 5.0))

    if require_no_zero_columns:
        assert probe_data is not None
        output_start = num_coefficients + num_rows
        parity_start = output_start + num_output_bits
        output_equations = hstack(
            [
                csr_matrix(probe_data.astype(float)),
                csr_matrix((num_output_bits, num_rows)),
                -eye(num_output_bits, format="csr"),
                -2 * eye(num_output_bits, format="csr"),
            ],
            format="csr",
        )
        constraint_blocks.append(output_equations)
        lower_blocks.append(np.zeros(num_output_bits))
        upper_blocks.append(np.zeros(num_output_bits))
        nonzero_rows = np.zeros((2 * NUM_SHEETS, num_variables), dtype=float)
        bits_per_probe = BLOCK_SIZE
        for probe in range(2 * NUM_SHEETS):
            start = output_start + probe * bits_per_probe
            nonzero_rows[probe, start : start + bits_per_probe] = 1
        constraint_blocks.append(csr_matrix(nonzero_rows))
        lower_blocks.append(np.ones(2 * NUM_SHEETS))
        upper_blocks.append(np.full(2 * NUM_SHEETS, np.inf))

    if require_nonzero_top_augmentation:
        augmentation_start = num_coefficients + num_rows + 2 * num_output_bits
        augmentation_slack_start = augmentation_start + LATTICE_ORDER
        augmentation_equations = np.zeros(
            (LATTICE_ORDER, num_variables), dtype=float
        )
        for bottom in range(LATTICE_ORDER):
            for top in range(6):
                augmentation_equations[
                    bottom, top * LATTICE_ORDER + bottom
                ] = 1
            augmentation_equations[bottom, augmentation_start + bottom] = -1
            augmentation_equations[
                bottom, augmentation_slack_start + bottom
            ] = -2
        constraint_blocks.append(csr_matrix(augmentation_equations))
        lower_blocks.append(np.zeros(LATTICE_ORDER))
        upper_blocks.append(np.zeros(LATTICE_ORDER))
        nonzero_augmentation = np.zeros((1, num_variables), dtype=float)
        nonzero_augmentation[
            0, augmentation_start:augmentation_slack_start
        ] = 1
        constraint_blocks.append(csr_matrix(nonzero_augmentation))
        lower_blocks.append(np.ones(1))
        upper_blocks.append(np.full(1, np.inf))

    lower = np.zeros(num_variables)
    upper = np.ones(num_variables)
    upper[num_coefficients : num_coefficients + num_rows] = np.maximum(
        1, np.count_nonzero(rows, axis=1) // 2
    )
    if require_no_zero_columns:
        output_start = num_coefficients + num_rows
        parity_start = output_start + num_output_bits
        upper[output_start:parity_start] = 1
        upper[parity_start:] = np.maximum(
            1,
            np.pad(
                np.count_nonzero(probe_data, axis=1) // 2,
                (0, 2 * num_augmentation_bits),
                constant_values=1,
            ),
        )
    if require_nonzero_top_augmentation:
        augmentation_start = num_coefficients + num_rows + 2 * num_output_bits
        augmentation_slack_start = augmentation_start + LATTICE_ORDER
        upper[augmentation_start:augmentation_slack_start] = 1
        upper[augmentation_slack_start:] = 3
    integrality = np.ones(num_variables, dtype=int)
    rng = random.Random(random_seed)
    records: list[dict[str, Any]] = []
    statuses: Counter[int] = Counter()
    css_hits = []

    for _attempt in range(solutions):
        matrix = vstack(constraint_blocks, format="csr")
        lb = np.concatenate(lower_blocks)
        ub = np.concatenate(upper_blocks)
        objective = np.zeros(num_variables)
        objective[:num_coefficients] = 10_000 + np.asarray(
            [rng.randrange(1_000) for _ in range(num_coefficients)]
        )
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, lb, ub),
            options={"time_limit": seconds_per_solution, "presolve": True},
        )
        statuses[int(result.status)] += 1
        if result.x is None:
            break
        coefficients = (result.x[:num_coefficients] > 0.5).astype(np.uint8)
        record = _generator_record(coefficients, checks_x, checks_z, zz)
        records.append(record)
        if record["css_orthogonal"]:
            css_hits.append(record)

        selected = np.flatnonzero(coefficients)
        excluded = np.zeros((1, num_variables), dtype=float)
        excluded[0, :num_coefficients] = 1
        excluded[0, selected] = -1
        # sum(not-selected x) + sum(selected (1-x)) >= 1
        constraint_blocks.append(csr_matrix(excluded))
        lower_blocks.append(np.asarray([1.0 - len(selected)]))
        upper_blocks.append(np.asarray([np.inf]))

    return {
        "method": "repeated-sparse-annihilator-milp",
        "minimum_f_terms": minimum_terms,
        "maximum_f_terms": maximum_terms,
        "maximum_check_weight_bound": 2 * maximum_terms,
        "solutions_requested": solutions,
        "solutions_found": len(records),
        "solver_status_counts": {str(key): value for key, value in sorted(statuses.items())},
        "coefficient_weight_counts": dict(
            sorted(Counter(record["coefficient_weight"] for record in records).items())
        ),
        "css_hits": css_hits,
        "num_css_hits": len(css_hits),
        "structural_pilot_hits": [
            record for record in records if record["structural_pilot_hit"]
        ],
        "num_structural_pilot_hits": sum(
            record["structural_pilot_hit"] for record in records
        ),
        "records": records,
        "bounded_pilot": True,
        "required_no_zero_columns_in_milp": require_no_zero_columns,
        "required_nonzero_top_augmentation_in_milp": require_nonzero_top_augmentation,
        "seconds": round(time.perf_counter() - started, 6),
    }


def _packed_constraint_columns(rows: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sum(int(rows[row, column]) << row for row in range(rows.shape[0]))
        for column in range(rows.shape[1])
    )


def _has_nonzero_top_augmentation(coefficients: np.ndarray) -> bool:
    return bool(np.any(np.sum(coefficients.reshape(6, LATTICE_ORDER), axis=0) % 2))


def search_mitm_six_term_generators(
    seed: np.ndarray,
    *,
    restarts: int = 10,
    maximum_records: int = 100,
    random_seed: int = 0,
    require_nonzero_top_augmentation: bool = True,
    require_no_zero_columns: bool = True,
    triples_per_syndrome: int = 4,
    term_weights: Sequence[int] = (6,),
    check_top_images: tuple[int, int, int] = (0, 2, 1),
) -> dict[str, Any]:
    """Find sparse annihilators by randomized meet-in-the-middle matching."""
    if min(restarts, maximum_records, triples_per_syndrome) < 1:
        raise ValueError("MITM limits must be positive")
    if not term_weights or any(not 2 <= weight <= 6 for weight in term_weights):
        raise ValueError("MITM term weights must lie between two and six")
    started = time.perf_counter()
    rows, checks_x, checks_z, zz = _fold_tied_constraint_data(
        seed, check_top_images=check_top_images
    )
    syndromes = _packed_constraint_columns(rows)
    rng = random.Random(random_seed)
    records: list[dict[str, Any]] = []
    structural_hits = []
    seen: set[tuple[int, ...]] = set()
    counters: Counter[str] = Counter()
    completed = 0

    stop = False
    for term_weight in term_weights:
        left_size = term_weight // 2
        right_size = term_weight - left_size
        for _restart in range(restarts):
            order = list(range(len(syndromes)))
            rng.shuffle(order)
            left = order[: len(order) // 2]
            right = order[len(order) // 2 :]
            left_by_syndrome: dict[int, list[tuple[int, ...]]] = {}
            for combination in itertools.combinations(left, left_size):
                key = 0
                for index in combination:
                    key ^= syndromes[index]
                bucket = left_by_syndrome.setdefault(key, [])
                if len(bucket) < triples_per_syndrome:
                    bucket.append(combination)
            counters[f"weight_{term_weight}_left_combinations"] += sum(
                len(bucket) for bucket in left_by_syndrome.values()
            )

            for combination in itertools.combinations(right, right_size):
                key = 0
                for index in combination:
                    key ^= syndromes[index]
                matches = left_by_syndrome.get(key)
                if not matches:
                    continue
                counters[f"weight_{term_weight}_syndrome_matches"] += len(matches)
                for match in matches:
                    support = tuple(sorted((*match, *combination)))
                    if support in seen:
                        counters["duplicate_supports"] += 1
                        continue
                    seen.add(support)
                    coefficients = np.zeros(len(syndromes), dtype=np.uint8)
                    coefficients[list(support)] = 1
                    if require_nonzero_top_augmentation and not _has_nonzero_top_augmentation(coefficients):
                        counters["zero_top_augmentation"] += 1
                        counters[f"weight_{term_weight}_zero_top_augmentation"] += 1
                        continue
                    record = _generator_record(coefficients, checks_x, checks_z, zz)
                    if require_no_zero_columns and not record["no_zero_columns"]:
                        counters["zero_columns"] += 1
                        counters[f"weight_{term_weight}_zero_columns"] += 1
                        continue
                    records.append(record)
                    counters["retained_records"] += 1
                    counters[f"weight_{term_weight}_retained_records"] += 1
                    if record["structural_pilot_hit"]:
                        structural_hits.append(record)
                        counters["structural_hits"] += 1
                    if len(records) >= maximum_records:
                        stop = True
                        break
                if stop:
                    break
            completed += 1
            if stop:
                break
        if stop:
            break

    return {
        "method": "randomized-three-plus-three-meet-in-the-middle",
        "restarts_requested": restarts,
        "restarts_completed": completed,
        "maximum_records": maximum_records,
        "term_weights": list(term_weights),
        "data_top_fold": list(standard_transposition_fold().top_images),
        "check_top_fold": list(check_top_images),
        "require_nonzero_top_augmentation": require_nonzero_top_augmentation,
        "require_no_zero_columns": require_no_zero_columns,
        "counters": dict(sorted(counters.items())),
        "records": records,
        "structural_pilot_hits": structural_hits,
        "num_structural_pilot_hits": len(structural_hits),
        "seconds": round(time.perf_counter() - started, 6),
    }


def geometry_run(*, random_seed: int, witnesses: int, annihilator_witnesses: int) -> dict[str, Any]:
    started = time.perf_counter()
    records = sample_cancellation_witnesses(seed=random_seed, count=witnesses)
    annihilators = []
    for record in records[:annihilator_witnesses]:
        seed = seed_array().reshape(NUM_QUBITS)
        seed[record["support"]] = 1
        annihilators.append(fold_tied_annihilator(seed.reshape(NUM_HALVES, TOP_DEGREE, X_ORDER, Y_ORDER)))
    return {
        "schema_version": SCHEMA_VERSION,
        "search": "n192-natural-s3-reverse-geometry",
        "n": NUM_QUBITS,
        "target": "one C8 x C4 logical grid in two parity batches",
        "witnesses_requested": witnesses,
        "witnesses_found": len(records),
        "all_weight_seven": all(record["weight"] == TARGET_DISTANCE for record in records),
        "all_batch_disjoint": all(record["batching"]["disjoint_within_each_batch"] for record in records),
        "all_pairing_identity": all(record["pairing_is_identity"] for record in records),
        "geometry_survives": bool(records),
        "witnesses": list(records),
        "annihilator_records": annihilators,
        "annihilator_stage_survives": any(
            record["rank_capacity_survives_k32_target"] for record in annihilators
        ),
        "generator_stage_ready": any(
            record["rank_capacity_survives_k32_target"] for record in annihilators
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }
