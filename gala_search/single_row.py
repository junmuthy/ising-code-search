"""Seed-first ``[[32,4,6]]`` search over ``GL(2,2) x C4``."""

from __future__ import annotations

import functools
import itertools
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qldpc import codes

SINGLE_ROW_SCHEMA_VERSION = 1
CYCLE_ORDER = 4
TOP_DIMENSION = 2
BLOCK_SIZE = TOP_DIMENSION * CYCLE_ORDER
NUM_DATA_BLOCKS = 4
NUM_CHECK_BLOCKS = 2
NUM_QUBITS = NUM_DATA_BLOCKS * BLOCK_SIZE
NUM_CHECKS = NUM_CHECK_BLOCKS * BLOCK_SIZE
NUM_FIBRES = NUM_DATA_BLOCKS * TOP_DIMENSION
NUM_LOGICALS = CYCLE_ORDER
TARGET_DIMENSION = 4
TARGET_CHECK_RANK = (NUM_QUBITS - TARGET_DIMENSION) // 2

TOP_MATRIX_BASIS = (
    np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
    np.asarray([[1, 1], [0, 1]], dtype=np.uint8),
    np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
    np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
)
ENTRY_NAMES = ("F0", "F1", "G0", "G1")
COEFFICIENTS_PER_ENTRY = len(TOP_MATRIX_BASIS) * CYCLE_ORDER
NUM_COEFFICIENTS = len(ENTRY_NAMES) * COEFFICIENTS_PER_ENTRY


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return binary reduced row echelon form and pivot columns."""
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    row = 0
    pivots: list[int] = []
    for column in range(reduced.shape[1]):
        choices = np.flatnonzero(reduced[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for other in np.flatnonzero(reduced[:, column]):
            if other != row:
                reduced[other] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced[:row], tuple(pivots)


def gf2_rank(matrix: np.ndarray) -> int:
    return len(gf2_rref(matrix)[1])


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    reduced, pivots = gf2_rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in set(pivots)]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for basis_row, free_column in enumerate(free):
        basis[basis_row, free_column] = 1
        for row, pivot in enumerate(pivots):
            basis[basis_row, pivot] = reduced[row, free_column]
    return basis


def _pack(vector: Sequence[int]) -> int:
    return sum(int(bit) << index for index, bit in enumerate(vector))


def _packed_row_basis(matrix: np.ndarray) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = _pack(row)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def _packed_in_span(value: int, basis: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def _cycle_lengths(permutation: Sequence[int]) -> list[int]:
    seen = np.zeros(len(permutation), dtype=bool)
    output: list[int] = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        current = start
        length = 0
        while not seen[current]:
            seen[current] = True
            length += 1
            current = int(permutation[current])
        output.append(length)
    return sorted(output)


@dataclass(frozen=True)
class TranslationFold:
    """A C4-commuting permutation of physical or check fibres."""

    fibre_images: tuple[int, ...]
    fibre_shifts: tuple[int, ...]

    def __post_init__(self) -> None:
        if sorted(self.fibre_images) != list(range(len(self.fibre_images))):
            raise ValueError("fibre images must be a permutation")
        if len(self.fibre_shifts) != len(self.fibre_images):
            raise ValueError("every fibre needs a cyclic shift")

    @property
    def permutation(self) -> np.ndarray:
        output = np.empty(len(self.fibre_images) * CYCLE_ORDER, dtype=int)
        for fibre, image in enumerate(self.fibre_images):
            for position in range(CYCLE_ORDER):
                source = fibre * CYCLE_ORDER + position
                output[source] = image * CYCLE_ORDER + (
                    position + self.fibre_shifts[fibre]
                ) % CYCLE_ORDER
        return output

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(_cycle_lengths(self.permutation))
        return {
            "fibre_images": list(self.fibre_images),
            "fibre_shifts": [value % CYCLE_ORDER for value in self.fibre_shifts],
            "cycle_length_counts": {
                str(length): count for length, count in sorted(counts.items())
            },
            "order": int(np.lcm.reduce(tuple(counts))),
        }


def structured_gl_fold(
    forward_blocks: Sequence[int], backward_blocks: Sequence[int]
) -> TranslationFold:
    """Swap GL coordinates while permitting different protograph maps."""
    size = len(forward_blocks)
    images = [0] * (2 * size)
    for block in range(size):
        images[2 * block] = 2 * int(forward_blocks[block]) + 1
        images[2 * block + 1] = 2 * int(backward_blocks[block])
    return TranslationFold(tuple(images), (0,) * (2 * size))


def iter_structured_gl_folds(num_blocks: int) -> Iterable[TranslationFold]:
    maps = tuple(itertools.permutations(range(num_blocks)))
    for forward in maps:
        for backward in maps:
            yield structured_gl_fold(forward, backward)


def _permute_columns(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, np.asarray(permutation, dtype=int)] = matrix
    return output


def _permute_rows(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[np.asarray(permutation, dtype=int)] = matrix
    return output


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    rows = []
    current = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(CYCLE_ORDER):
        rows.append(current)
        shifted = np.zeros_like(current)
        for fibre in range(len(seed) // CYCLE_ORDER):
            section = current[fibre * 4 : (fibre + 1) * 4]
            shifted[fibre * 4 : (fibre + 1) * 4] = np.roll(section, 1)
        current = shifted
    return np.asarray(rows, dtype=np.uint8)


def canonical_seed_support(support: Sequence[int]) -> tuple[int, ...]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in translation_orbit(seed)
    )


def random_graph_seed(rng: random.Random, *, weight: int = 6) -> tuple[int, ...]:
    fibres = rng.sample(range(NUM_FIBRES), weight)
    support = [fibre * CYCLE_ORDER + rng.randrange(CYCLE_ORDER) for fibre in fibres]
    return canonical_seed_support(support)


def analyze_seed_fold(support: Sequence[int], fold: TranslationFold) -> dict[str, Any]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    z_orbit = translation_orbit(seed)
    x_orbit = _permute_columns(z_orbit, fold.permutation)
    pairing = (z_orbit @ x_orbit.T) % 2
    occupancies = [
        int(np.count_nonzero(seed[fibre * 4 : (fibre + 1) * 4]))
        for fibre in range(NUM_FIBRES)
    ]
    return {
        "seed_support": list(map(int, support)),
        "weight": len(support),
        "occupied_fibres": np.flatnonzero(occupancies).astype(int).tolist(),
        "graph_supported": max(occupancies, default=0) <= 1,
        "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
        "physical_orbit_rank": gf2_rank(z_orbit),
        "zx_pairing_rank": gf2_rank(pairing),
        "zx_pairing_weight": int(np.count_nonzero(pairing)),
        "pairing_is_permutation": bool(
            np.all(np.count_nonzero(pairing, axis=0) == 1)
            and np.all(np.count_nonzero(pairing, axis=1) == 1)
        ),
    }


def search_seed_fold_witnesses(
    *, seed: int, trials_per_fold: int = 20, target_witnesses: int = 100
) -> dict[str, Any]:
    rng = random.Random(seed)
    ranks: Counter[int] = Counter()
    witnesses: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    tested = 0
    started = time.perf_counter()
    for fold in iter_structured_gl_folds(NUM_DATA_BLOCKS):
        for _ in range(trials_per_fold):
            support = random_graph_seed(rng)
            tested += 1
            analysis = analyze_seed_fold(support, fold)
            ranks[analysis["zx_pairing_rank"]] += 1
            if analysis["zx_pairing_rank"] != NUM_LOGICALS:
                continue
            key = (tuple(support), tuple(fold.permutation))
            if key in seen:
                continue
            seen.add(key)
            witnesses.append({"fold": fold.to_dict(), **analysis})
            if len(witnesses) >= target_witnesses:
                return {
                    "seed": seed,
                    "tested": tested,
                    "complete_fold_sweep": False,
                    "rank_counts": dict(sorted(ranks.items())),
                    "witnesses": witnesses,
                    "seconds": round(time.perf_counter() - started, 6),
                }
    return {
        "seed": seed,
        "tested": tested,
        "complete_fold_sweep": True,
        "rank_counts": dict(sorted(ranks.items())),
        "witnesses": witnesses,
        "seconds": round(time.perf_counter() - started, 6),
    }


@functools.lru_cache(maxsize=None)
def cyclic_shift_matrix(shift: int) -> np.ndarray:
    matrix = np.zeros((CYCLE_ORDER, CYCLE_ORDER), dtype=np.uint8)
    for source in range(CYCLE_ORDER):
        matrix[(source + shift) % CYCLE_ORDER, source] = 1
    return matrix


@functools.lru_cache(maxsize=None)
def coefficient_lift(top: int, shift: int) -> np.ndarray:
    return np.kron(TOP_MATRIX_BASIS[top], cyclic_shift_matrix(shift)).astype(
        np.uint8
    )


def _basis_checks(entry: int, top: int, shift: int) -> tuple[np.ndarray, np.ndarray]:
    lift = coefficient_lift(top, shift)
    matrix_x = np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
    matrix_z = np.zeros_like(matrix_x)
    offset = entry % 2
    side = entry // 2
    for row in range(NUM_CHECK_BLOCKS):
        column = side * 2 + (offset + row) % 2
        matrix_x[
            row * BLOCK_SIZE : (row + 1) * BLOCK_SIZE,
            column * BLOCK_SIZE : (column + 1) * BLOCK_SIZE,
        ] = lift
        dual_column = (1 - side) * 2 + (offset + row) % 2
        matrix_z[
            row * BLOCK_SIZE : (row + 1) * BLOCK_SIZE,
            dual_column * BLOCK_SIZE : (dual_column + 1) * BLOCK_SIZE,
        ] = lift.T
    return matrix_x, matrix_z


@functools.lru_cache(maxsize=1)
def coefficient_checks() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple(
        _basis_checks(entry, top, shift)
        for entry in range(4)
        for top in range(4)
        for shift in range(4)
    )


def checks_from_coefficients(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(coefficients)
    if not len(selected):
        zero = np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
        return zero, zero.copy()
    matrix_x = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][0] for index in selected]), axis=0
    )
    matrix_z = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][1] for index in selected]), axis=0
    )
    return matrix_x, matrix_z


def polynomial_constraint_matrix(
    support: Sequence[int],
    *,
    data_fold: TranslationFold,
    check_fold: TranslationFold,
) -> np.ndarray:
    """Linear seed-kernel and exact forward-fold equations."""
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    columns = []
    for matrix_x, matrix_z in coefficient_checks():
        folded_x = _permute_rows(
            _permute_columns(matrix_x, data_fold.permutation),
            check_fold.permutation,
        )
        columns.append(
            np.concatenate([(matrix_x @ seed) % 2, (matrix_z ^ folded_x).ravel()])
        )
    return np.asarray(columns, dtype=np.uint8).T


def seed_constraint_matrix(support: Sequence[int]) -> np.ndarray:
    """Return only the linear equations forcing the C4 seed into ker(HX)."""
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return np.asarray(
        [(matrix_x @ seed) % 2 for matrix_x, _matrix_z in coefficient_checks()],
        dtype=np.uint8,
    ).T


def add_automatic_css_relation(
    constraints: np.ndarray, *, q_bits: int, a: int, b: int
) -> np.ndarray:
    """Impose ``G=q*(aF0+bF1,bF0+aF1)`` over F2[C4]."""
    if not 1 <= q_bits < 1 << CYCLE_ORDER:
        raise ValueError("q must be a nonzero C4 polynomial")
    if (a, b) not in {(1, 0), (0, 1), (1, 1)}:
        raise ValueError("unsupported automatic-CSS protograph relation")
    equations = np.zeros(
        (2 * COEFFICIENTS_PER_ENTRY, NUM_COEFFICIENTS), dtype=np.uint8
    )
    q_support = [shift for shift in range(4) if q_bits >> shift & 1]
    for target_entry in range(2):
        g_entry = 2 + target_entry
        source_weights = (a, b) if target_entry == 0 else (b, a)
        for top in range(4):
            for target_shift in range(4):
                equation = target_entry * COEFFICIENTS_PER_ENTRY + top * 4 + target_shift
                equations[
                    equation,
                    g_entry * COEFFICIENTS_PER_ENTRY + top * 4 + target_shift,
                ] = 1
                for source_entry, enabled in enumerate(source_weights):
                    if not enabled:
                        continue
                    for q_shift in q_support:
                        source_shift = (target_shift - q_shift) % 4
                        source_index = (
                            source_entry * COEFFICIENTS_PER_ENTRY
                            + top * 4
                            + source_shift
                        )
                        equations[equation, source_index] ^= 1
    return np.vstack([constraints, equations])


def _rowspaces_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_rank = gf2_rank(left)
    right_rank = gf2_rank(right)
    return bool(
        left_rank == right_rank
        and gf2_rank(np.vstack([left, right])) == left_rank
    )


def is_zx_fold(
    matrix_x: np.ndarray, matrix_z: np.ndarray, permutation: Sequence[int]
) -> bool:
    return bool(
        _rowspaces_equal(_permute_columns(matrix_x, permutation), matrix_z)
        and _rowspaces_equal(_permute_columns(matrix_z, permutation), matrix_x)
    )


def analyze_logical_grid(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    support: Sequence[int],
    fold: TranslationFold,
) -> dict[str, Any]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    z_orbit = translation_orbit(seed)
    x_orbit = _permute_columns(z_orbit, fold.permutation)
    pairing = (z_orbit @ x_orbit.T) % 2
    return {
        "seed_support": list(map(int, support)),
        "weight": len(support),
        "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
        "z_orbit_in_kernel": bool(not np.any((matrix_x @ z_orbit.T) % 2)),
        "x_orbit_in_kernel": bool(not np.any((matrix_z @ x_orbit.T) % 2)),
        "orbit_rank_mod_stabilizers": (
            gf2_rank(np.vstack([matrix_z, z_orbit])) - gf2_rank(matrix_z)
        ),
        "zx_pairing_rank": gf2_rank(pairing),
        "pairing": pairing.astype(int).tolist(),
    }


def _tanner_connected(matrix_x: np.ndarray, matrix_z: np.ndarray) -> bool:
    checks = np.vstack([matrix_x, matrix_z])
    num_checks, num_qubits = checks.shape
    adjacency = [set() for _ in range(num_checks + num_qubits)]
    rows, columns = np.nonzero(checks)
    for check, qubit in zip(rows, columns):
        adjacency[int(check)].add(num_checks + int(qubit))
        adjacency[num_checks + int(qubit)].add(int(check))
    active = [index for index, neighbors in enumerate(adjacency) if neighbors]
    if not active:
        return False
    reached = {active[0]}
    stack = [active[0]]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    return len(reached) == len(active) and all(
        adjacency[num_checks + qubit] for qubit in range(num_qubits)
    )


def find_css_logical_below_six(
    check: np.ndarray, stabilizer: np.ndarray
) -> dict[str, Any] | None:
    column_syndromes = [_pack(check[:, qubit]) for qubit in range(NUM_QUBITS)]
    stabilizer_basis = _packed_row_basis(stabilizer)
    for weight in range(1, 6):
        for support in itertools.combinations(range(NUM_QUBITS), weight):
            syndrome = 0
            vector = 0
            for qubit in support:
                syndrome ^= column_syndromes[qubit]
                vector |= 1 << qubit
            if syndrome == 0 and not _packed_in_span(vector, stabilizer_basis):
                return {"weight": weight, "support": list(support)}
    return None


def certify_distance_six(matrix_x: np.ndarray, matrix_z: np.ndarray) -> dict[str, Any]:
    import math

    started = time.perf_counter()
    z_logical = find_css_logical_below_six(matrix_x, matrix_z)
    x_logical = find_css_logical_below_six(matrix_z, matrix_x)
    return {
        "certified_distance_at_least_six": z_logical is None and x_logical is None,
        "z_logical_below_six": z_logical,
        "x_logical_below_six": x_logical,
        "supports_checked_per_sector": sum(
            math.comb(NUM_QUBITS, weight) for weight in range(1, 6)
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


def serialize_coefficients(coefficients: np.ndarray) -> dict[str, Any]:
    entries: dict[str, list[dict[str, int]]] = {name: [] for name in ENTRY_NAMES}
    for index in np.flatnonzero(coefficients):
        entry, local = divmod(int(index), COEFFICIENTS_PER_ENTRY)
        top, shift = divmod(local, CYCLE_ORDER)
        entries[ENTRY_NAMES[entry]].append({"top_basis": top, "x": shift})
    return entries


def analyze_candidate(
    coefficients: np.ndarray,
    *,
    support: Sequence[int],
    data_fold: TranslationFold,
    check_fold: TranslationFold,
    certify_distance: bool = True,
    require_exact_forward: bool = True,
) -> dict[str, Any]:
    matrix_x, matrix_z = checks_from_coefficients(coefficients)
    folded_x = _permute_rows(
        _permute_columns(matrix_x, data_fold.permutation), check_fold.permutation
    )
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    grid = analyze_logical_grid(matrix_x, matrix_z, support, data_fold)
    checks = {
        "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
        "rank_14_14": rank_x == TARGET_CHECK_RANK and rank_z == TARGET_CHECK_RANK,
        "reverse_zx_fold": is_zx_fold(matrix_x, matrix_z, data_fold.permutation),
        "logical_grid": bool(
            grid["pairwise_disjoint"]
            and grid["z_orbit_in_kernel"]
            and grid["x_orbit_in_kernel"]
            and grid["orbit_rank_mod_stabilizers"] == NUM_LOGICALS
            and grid["zx_pairing_rank"] == NUM_LOGICALS
        ),
        "tanner_connected": _tanner_connected(matrix_x, matrix_z),
    }
    if require_exact_forward:
        checks["exact_forward_fold"] = bool(np.array_equal(matrix_z, folded_x))
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    result: dict[str, Any] = {
        "n": NUM_QUBITS,
        "k": NUM_QUBITS - rank_x - rank_z if checks["css_orthogonal"] else None,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "logical_grid": grid,
        "row_weights_x": sorted(set(map(int, row_weights_x))),
        "row_weights_z": sorted(set(map(int, row_weights_z))),
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "coefficient_weight": int(np.count_nonzero(coefficients)),
        "entries": serialize_coefficients(coefficients),
    }
    if result["accepted_structurally"] and certify_distance:
        result["distance"] = certify_distance_six(matrix_x, matrix_z)
        result["accepted"] = result["distance"]["certified_distance_at_least_six"]
    else:
        result["distance"] = None
        result["accepted"] = False
    return result


def solve_constraint_space(
    constraints: np.ndarray,
    *,
    support: Sequence[int],
    data_fold: TranslationFold,
    check_fold: TranslationFold,
    maximum_check_weight: int = 12,
    enumeration_limit: int = 20,
    random_samples: int = 20_000,
    random_seed: int = 0,
    maximum_hits: int = 3,
    require_exact_forward: bool = True,
    alternative_data_folds: Sequence[TranslationFold] = (),
) -> dict[str, Any]:
    """Search a complete linear generator space for valid rank-14 checks."""
    started = time.perf_counter()
    reduced, pivots = gf2_rref(constraints)
    basis = gf2_nullspace(reduced)
    nullity = len(basis)
    lifted_x = [checks_from_coefficients(vector)[0] for vector in basis]
    lifted_z = [checks_from_coefficients(vector)[1] for vector in basis]
    capacity_x = gf2_rank(np.vstack(lifted_x)) if lifted_x else 0
    capacity_z = gf2_rank(np.vstack(lifted_z)) if lifted_z else 0
    common: dict[str, Any] = {
        "constraint_rank": len(pivots),
        "constraint_nullity": nullity,
        "rank_capacity_x": capacity_x,
        "rank_capacity_z": capacity_z,
        "required_rank": TARGET_CHECK_RANK,
        "maximum_check_weight": maximum_check_weight,
    }
    if min(capacity_x, capacity_z) < TARGET_CHECK_RANK:
        return {
            **common,
            "status": "exact_rank_capacity_no_go",
            "tested": 0,
            "hits": [],
            "seconds": round(time.perf_counter() - started, 6),
        }

    rng = random.Random(random_seed)
    if nullity <= enumeration_limit:
        generator_vectors: Iterable[np.ndarray] = (
            np.fromiter(
                ((integer >> bit) & 1 for bit in range(nullity)),
                dtype=np.uint8,
                count=nullity,
            )
            for integer in range(1, 1 << nullity)
        )
        method = "exhaustive"
        planned = (1 << nullity) - 1
    else:
        generator_vectors = (
            np.fromiter(
                (rng.getrandbits(1) for _ in range(nullity)),
                dtype=np.uint8,
                count=nullity,
            )
            for _ in range(random_samples)
        )
        method = "random_probe"
        planned = random_samples

    hits: list[dict[str, Any]] = []
    tested = 0
    rank_counts: Counter[int] = Counter()
    filter_counts: Counter[str] = Counter()
    for generator_bits in generator_vectors:
        tested += 1
        selected = np.flatnonzero(generator_bits)
        if not len(selected):
            continue
        coefficients = np.bitwise_xor.reduce(basis[selected], axis=0)
        if any(
            not np.any(
                coefficients[
                    entry * COEFFICIENTS_PER_ENTRY : (entry + 1)
                    * COEFFICIENTS_PER_ENTRY
                ]
            )
            for entry in range(4)
        ):
            continue
        matrix_x, matrix_z = checks_from_coefficients(coefficients)
        rank_x = gf2_rank(matrix_x)
        rank_counts[rank_x] += 1
        if rank_x != TARGET_CHECK_RANK or gf2_rank(matrix_z) != TARGET_CHECK_RANK:
            continue
        filter_counts["rank_14_14"] += 1
        if max(
            np.count_nonzero(matrix_x, axis=1).max(initial=0),
            np.count_nonzero(matrix_z, axis=1).max(initial=0),
        ) > maximum_check_weight:
            continue
        filter_counts["check_weight"] += 1
        if np.any((matrix_x @ matrix_z.T) % 2):
            continue
        filter_counts["css_orthogonal"] += 1
        folds_to_test = (data_fold, *alternative_data_folds)
        seen_folds: set[tuple[int, ...]] = set()
        for candidate_fold in folds_to_test:
            fold_key = tuple(map(int, candidate_fold.permutation))
            if fold_key in seen_folds:
                continue
            seen_folds.add(fold_key)
            if not is_zx_fold(matrix_x, matrix_z, candidate_fold.permutation):
                continue
            filter_counts["reverse_zx_fold"] += 1
            analysis = analyze_candidate(
                coefficients,
                support=support,
                data_fold=candidate_fold,
                check_fold=check_fold,
                certify_distance=True,
                require_exact_forward=require_exact_forward,
            )
            for name, passed in analysis["checks"].items():
                if passed:
                    filter_counts[name] += 1
            if analysis["accepted_structurally"]:
                hits.append(
                    {
                        "coefficients": np.flatnonzero(coefficients).astype(int).tolist(),
                        "data_fold": candidate_fold.to_dict(),
                        "analysis": analysis,
                    }
                )
                break
        if len(hits) >= maximum_hits:
            break
    return {
        **common,
        "status": "hit" if hits else "searched_without_structural_hit",
        "method": method,
        "planned": planned,
        "tested": tested,
        "rank_counts": dict(sorted(rank_counts.items())),
        "filter_counts": dict(sorted(filter_counts.items())),
        "hits": hits,
        "seconds": round(time.perf_counter() - started, 6),
    }


def build_code(coefficients: Sequence[int] | np.ndarray) -> codes.CSSCode:
    """Construct a qLDPC CSSCode from a bit vector or selected indices."""
    array = np.asarray(coefficients)
    if array.shape != (NUM_COEFFICIENTS,):
        vector = np.zeros(NUM_COEFFICIENTS, dtype=np.uint8)
        vector[list(map(int, coefficients))] = 1
    else:
        vector = np.asarray(array, dtype=np.uint8)
    matrix_x, matrix_z = checks_from_coefficients(vector)
    return codes.CSSCode(matrix_x, matrix_z)


def gf2_affine_solve(
    matrix: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one solution and a nullspace basis for ``matrix*x=target``."""
    augmented = np.hstack(
        [np.asarray(matrix, dtype=np.uint8), np.asarray(target, dtype=np.uint8)[:, None]]
    )
    reduced, pivots = gf2_rref(augmented)
    num_variables = matrix.shape[1]
    if num_variables in pivots:
        return None
    particular = np.zeros(num_variables, dtype=np.uint8)
    for row, pivot in enumerate(pivots):
        if pivot < num_variables:
            particular[pivot] = reduced[row, num_variables]
    return particular, gf2_nullspace(matrix)


@functools.lru_cache(maxsize=1)
def f_block_bases() -> tuple[np.ndarray, ...]:
    return tuple(coefficient_checks()[index][0][:, :16] for index in range(32))


@functools.lru_cache(maxsize=1)
def g_block_bases() -> tuple[np.ndarray, ...]:
    return tuple(coefficient_checks()[32 + index][0][:, 16:] for index in range(32))


def fixed_f_affine_system(
    f_support: Sequence[int], seed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the binary affine CSS and seed-kernel equations for ``G``."""
    f_matrix_x = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][0] for index in f_support]),
        axis=0,
    )
    f_matrix_z = np.bitwise_xor.reduce(
        np.asarray([coefficient_checks()[index][1] for index in f_support]),
        axis=0,
    )
    g_check_bases = coefficient_checks()[32:]
    css_columns = [
        (
            ((f_matrix_x @ matrix_z.T) % 2)
            ^ ((matrix_x @ f_matrix_z.T) % 2)
        ).ravel()
        for matrix_x, matrix_z in g_check_bases
    ]
    seed_columns = [
        (matrix_x @ seed) % 2 for matrix_x, _matrix_z in g_check_bases
    ]
    affine_matrix = np.vstack(
        [
            np.asarray(css_columns, dtype=np.uint8).T,
            np.asarray(seed_columns, dtype=np.uint8).T,
        ]
    )
    affine_target = np.concatenate(
        [np.zeros(16 * 16, dtype=np.uint8), (f_matrix_x @ seed) % 2]
    )
    return f_matrix_x, f_matrix_z, affine_matrix, affine_target


def search_fixed_f_solved_g(
    *,
    support: Sequence[int],
    data_folds: Sequence[TranslationFold],
    f_trials: int = 10_000,
    g_samples: int = 32,
    f_weight_min: int = 4,
    f_weight_max: int = 10,
    maximum_check_weight: int = 12,
    random_seed: int = 0,
    maximum_hits: int = 3,
) -> dict[str, Any]:
    """Sample sparse F and solve the affine CSS/kernel system for G."""
    rng = random.Random(random_seed)
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    check_fold = next(iter_structured_gl_folds(2))
    counters: Counter[str] = Counter()
    nullities: Counter[int] = Counter()
    hits: list[dict[str, Any]] = []
    seen_f: set[tuple[int, ...]] = set()
    started = time.perf_counter()

    for _trial in range(f_trials):
        weight = rng.randint(f_weight_min, f_weight_max)
        f_support = tuple(sorted(rng.sample(range(32), weight)))
        if f_support in seen_f:
            continue
        seen_f.add(f_support)
        counters["f_tested"] += 1
        f_matrix_x, _f_matrix_z, affine_matrix, affine_target = (
            fixed_f_affine_system(f_support, seed)
        )
        solution = gf2_affine_solve(affine_matrix, affine_target)
        if solution is None:
            counters["affine_inconsistent"] += 1
            continue
        counters["affine_consistent"] += 1
        particular, basis = solution
        nullities[len(basis)] += 1

        if len(basis) <= 8:
            generator_vectors: Iterable[np.ndarray] = (
                np.fromiter(
                    ((integer >> bit) & 1 for bit in range(len(basis))),
                    dtype=np.uint8,
                    count=len(basis),
                )
                for integer in range(1 << len(basis))
            )
        else:
            generator_vectors = (
                np.fromiter(
                    (rng.getrandbits(1) for _ in range(len(basis))),
                    dtype=np.uint8,
                    count=len(basis),
                )
                for _ in range(g_samples)
            )
        for generators in generator_vectors:
            counters["g_tested"] += 1
            chosen = np.flatnonzero(generators)
            g_vector = particular.copy()
            if len(chosen):
                g_vector ^= np.bitwise_xor.reduce(basis[chosen], axis=0)
            if not np.any(g_vector[:16]) or not np.any(g_vector[16:]):
                continue
            coefficients = np.concatenate(
                [
                    np.fromiter(
                        (1 if index in f_support else 0 for index in range(32)),
                        dtype=np.uint8,
                        count=32,
                    ),
                    g_vector,
                ]
            )
            matrix_x, matrix_z = checks_from_coefficients(coefficients)
            if gf2_rank(matrix_x) != 14 or gf2_rank(matrix_z) != 14:
                continue
            counters["rank_14_14"] += 1
            if max(
                np.count_nonzero(matrix_x, axis=1).max(initial=0),
                np.count_nonzero(matrix_z, axis=1).max(initial=0),
            ) > maximum_check_weight:
                continue
            counters["check_weight"] += 1
            if np.any((matrix_x @ matrix_z.T) % 2):
                raise AssertionError("affine CSS solver returned noncommuting checks")
            for data_fold in data_folds:
                if not is_zx_fold(matrix_x, matrix_z, data_fold.permutation):
                    continue
                counters["zx_fold"] += 1
                analysis = analyze_candidate(
                    coefficients,
                    support=support,
                    data_fold=data_fold,
                    check_fold=check_fold,
                    certify_distance=True,
                    require_exact_forward=False,
                )
                if analysis["accepted_structurally"]:
                    hits.append(
                        {
                            "coefficients": np.flatnonzero(coefficients)
                            .astype(int)
                            .tolist(),
                            "data_fold": data_fold.to_dict(),
                            "analysis": analysis,
                        }
                    )
                    break
            if len(hits) >= maximum_hits:
                break
        if len(hits) >= maximum_hits:
            break
    return {
        "status": "hit" if hits else "sampled_without_hit",
        "arguments": {
            "f_trials": f_trials,
            "g_samples": g_samples,
            "f_weight_min": f_weight_min,
            "f_weight_max": f_weight_max,
            "maximum_check_weight": maximum_check_weight,
            "random_seed": random_seed,
        },
        "counters": dict(sorted(counters.items())),
        "affine_nullities": dict(sorted(nullities.items())),
        "compatible_folds": len(data_folds),
        "hits": hits,
        "seconds": round(time.perf_counter() - started, 6),
    }
