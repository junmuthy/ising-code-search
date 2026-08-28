"""Seed-first half-checkerboard search over the faithful two-dimensional S3 lift.

The target is one logical ``C8 x C4`` grid in a ``[[256, 32, >=6]]`` CSS
code.  Four GALA data blocks of size 64 give eight physical translation
orbits, so a graph-supported seed of weight six or seven has enough room for
32 pairwise-disjoint translates.

This module deliberately separates two questions:

1. Does a proposed physical ZX fold give a nondegenerate pairing on the
   translated grid?
2. Do sparse group-ring generators exist which annihilate the seed and make
   that permutation an actual ZX fold of the code?

The first question is cheap and is answered before any code is lifted.  The
second is expressed as a binary linear constraint system and passed to a
small mixed-integer search for sparse polynomial support.
"""

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

from .s3_ising import ProductMonomial
from .s3_linear import (
    BLOCK_SIZE,
    BOTTOM_ORDER,
    BOTTOM_X_ORDER,
    BOTTOM_Y_ORDER,
    _build_linear_css_code,
    _permute_columns,
    analyze_linear_translation_seed,
    is_zx_fold,
    linear_internal_translation_orbits,
    linear_logical_translation_orbit,
    linear_ring_monomial,
    linear_ring_polynomial,
)

HALF_GRID_SCHEMA_VERSION = 1
NUM_PHYSICAL_BLOCKS = 4
NUM_CHECK_BLOCKS = 2
NUM_QUBITS = NUM_PHYSICAL_BLOCKS * BLOCK_SIZE
NUM_CHECKS = NUM_CHECK_BLOCKS * BLOCK_SIZE
NUM_TRANSLATION_ORBITS = 2 * NUM_PHYSICAL_BLOCKS
NUM_LOGICALS = BOTTOM_ORDER
TARGET_DIMENSION = NUM_LOGICALS
# These four group elements lift to a basis of ``M_2(F_2)``.  Using all six
# elements as independent coefficient variables introduces a two-dimensional
# representation kernel at every bottom monomial, which a sparse solver can
# exploit to synthesize a formally nonzero polynomial with a zero binary lift.
TOP_MATRIX_BASIS = (0, 1, 2, 3)


def gf2_rank(matrix: np.ndarray) -> int:
    """Return the rank of a binary matrix without requiring a field object."""
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    rank = 0
    for column in range(reduced.shape[1]):
        pivots = np.flatnonzero(reduced[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in np.flatnonzero(reduced[:, column]):
            if row != rank:
                reduced[row] ^= reduced[rank]
        rank += 1
        if rank == reduced.shape[0]:
            break
    return rank


def _cycle_lengths(permutation: Sequence[int]) -> list[int]:
    visited = np.zeros(len(permutation), dtype=bool)
    lengths: list[int] = []
    for start in range(len(permutation)):
        if visited[start]:
            continue
        current = start
        length = 0
        while not visited[current]:
            visited[current] = True
            length += 1
            current = int(permutation[current])
        lengths.append(length)
    return sorted(lengths)


@dataclass(frozen=True)
class StructuredGLFold:
    """A translation-commuting fold on protograph and GL coordinates.

    ``forward_blocks`` maps top coordinate zero to top coordinate one, while
    ``backward_blocks`` maps top coordinate one to top coordinate zero.  If
    the two maps agree, this reduces to the usual top-independent GL
    alternating-form swap.  Allowing them to differ gives longer even cycles
    and removes the automatic rank ceiling seen for the involutory folds.
    """

    forward_blocks: tuple[int, ...]
    backward_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.forward_blocks) != len(self.backward_blocks):
            raise ValueError("forward and backward maps must have equal size")
        expected = list(range(len(self.forward_blocks)))
        if sorted(self.forward_blocks) != expected:
            raise ValueError("forward block map must be a permutation")
        if sorted(self.backward_blocks) != expected:
            raise ValueError("backward block map must be a permutation")

    @property
    def num_blocks(self) -> int:
        return len(self.forward_blocks)

    @property
    def is_top_independent(self) -> bool:
        return self.forward_blocks == self.backward_blocks

    @property
    def permutation(self) -> np.ndarray:
        output = np.empty(self.num_blocks * BLOCK_SIZE, dtype=int)
        for block in range(self.num_blocks):
            base = block * BLOCK_SIZE
            forward = self.forward_blocks[block] * BLOCK_SIZE
            backward = self.backward_blocks[block] * BLOCK_SIZE
            bottom = np.arange(BOTTOM_ORDER)
            output[base + bottom] = forward + BOTTOM_ORDER + bottom
            output[base + BOTTOM_ORDER + bottom] = backward + bottom
        return output

    def to_dict(self) -> dict[str, Any]:
        permutation = self.permutation
        cycle_counts = Counter(_cycle_lengths(permutation))
        return {
            "forward_blocks": list(self.forward_blocks),
            "backward_blocks": list(self.backward_blocks),
            "top_independent": self.is_top_independent,
            "cycle_length_counts": {
                str(length): count for length, count in sorted(cycle_counts.items())
            },
            "order": int(np.lcm.reduce(tuple(cycle_counts))),
        }


def iter_structured_gl_folds(num_blocks: int = NUM_PHYSICAL_BLOCKS) -> Iterable[StructuredGLFold]:
    """Enumerate every translation-commuting, top-swapping sheet fold."""
    permutations = tuple(itertools.permutations(range(num_blocks)))
    for forward in permutations:
        for backward in permutations:
            yield StructuredGLFold(forward, backward)


def canonical_translation_support(support: Sequence[int]) -> tuple[int, ...]:
    """Choose the lexicographically least translate of a half-grid seed."""
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in linear_logical_translation_orbit(
            seed, num_blocks=NUM_PHYSICAL_BLOCKS
        )
    )


def analyze_fold_seed(
    support: Sequence[int], fold: StructuredGLFold | Sequence[int]
) -> dict[str, Any]:
    """Analyze disjointness and ZX pairing before constructing stabilizers."""
    permutation = (
        fold.permutation
        if isinstance(fold, StructuredGLFold)
        else np.asarray(fold, dtype=int)
    )
    if len(permutation) != NUM_QUBITS:
        raise ValueError(f"expected a {NUM_QUBITS}-qubit permutation")
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    internal_orbits = linear_internal_translation_orbits(NUM_PHYSICAL_BLOCKS)
    occupancies = [
        int(np.count_nonzero(seed[list(orbit)])) for orbit in internal_orbits
    ]
    orbit = linear_logical_translation_orbit(
        seed, num_blocks=NUM_PHYSICAL_BLOCKS
    )
    folded = _permute_columns(orbit, permutation)
    pairing = (orbit @ folded.T) % 2
    row_weights = np.count_nonzero(pairing, axis=1)
    column_weights = np.count_nonzero(pairing, axis=0)
    return {
        "seed_support": list(canonical_translation_support(support)),
        "weight": int(np.count_nonzero(seed)),
        "internal_orbits": np.flatnonzero(occupancies).astype(int).tolist(),
        "internal_occupancies": occupancies,
        "graph_supported": max(occupancies, default=0) <= 1,
        "pairwise_disjoint": bool(np.all(np.sum(orbit, axis=0) <= 1)),
        "physical_orbit_rank": gf2_rank(orbit),
        "zx_pairing_rank": gf2_rank(pairing),
        "zx_pairing_weight": int(np.count_nonzero(pairing)),
        "pairing_is_permutation": bool(
            np.all(row_weights == 1) and np.all(column_weights == 1)
        ),
        "pairing_kernel": pairing[0].astype(int).tolist(),
    }


def random_graph_seed(
    rng: random.Random, *, weight: int = 6
) -> tuple[int, ...]:
    """Sample one qubit from each of ``weight`` distinct translation fibres."""
    if not 1 <= weight <= NUM_TRANSLATION_ORBITS:
        raise ValueError("graph seed weight must lie between one and eight")
    orbits = linear_internal_translation_orbits(NUM_PHYSICAL_BLOCKS)
    chosen = rng.sample(range(NUM_TRANSLATION_ORBITS), weight)
    return canonical_translation_support([rng.choice(orbits[index]) for index in chosen])


def standard_fold_control(
    *, seed: int, weight: int = 6, trials_per_fold: int = 2_000
) -> dict[str, Any]:
    """Measure the pairing-rank ceiling of the two ordinary ``L=4`` folds."""
    if trials_per_fold < 1:
        raise ValueError("trial count must be positive")
    rng = random.Random(seed)
    folds = {
        "fixed_blocks": StructuredGLFold(
            tuple(range(NUM_PHYSICAL_BLOCKS)),
            tuple(range(NUM_PHYSICAL_BLOCKS)),
        ),
        "half_reflection": StructuredGLFold(
            (1, 0, 3, 2),
            (1, 0, 3, 2),
        ),
    }
    results: dict[str, Any] = {}
    for name, fold in folds.items():
        counts: Counter[int] = Counter()
        for _trial in range(trials_per_fold):
            support = random_graph_seed(rng, weight=weight)
            counts[analyze_fold_seed(support, fold)["zx_pairing_rank"]] += 1
        results[name] = {
            "fold": fold.to_dict(),
            "rank_counts": dict(sorted(counts.items())),
            "maximum_sampled_rank": max(counts, default=0),
        }
    return {
        "seed": seed,
        "weight": weight,
        "trials_per_fold": trials_per_fold,
        "folds": results,
    }


def search_fold_seed_witnesses(
    *,
    seed: int,
    weight: int = 6,
    trials_per_fold: int = 20,
    target_witnesses: int = 100,
) -> dict[str, Any]:
    """Search all structured folds for graph seeds of full ZX rank."""
    if min(trials_per_fold, target_witnesses) < 1:
        raise ValueError("trial and target counts must be positive")
    rng = random.Random(seed)
    ranks: Counter[int] = Counter()
    witnesses: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    tested = 0
    started = time.perf_counter()
    for fold in iter_structured_gl_folds():
        for _trial in range(trials_per_fold):
            support = random_graph_seed(rng, weight=weight)
            tested += 1
            analysis = analyze_fold_seed(support, fold)
            ranks[analysis["zx_pairing_rank"]] += 1
            if analysis["zx_pairing_rank"] != NUM_LOGICALS:
                continue
            key = (fold.forward_blocks, fold.backward_blocks, tuple(support))
            if key in seen:
                continue
            seen.add(key)
            witnesses.append({"fold": fold.to_dict(), **analysis})
            if len(witnesses) >= target_witnesses:
                return {
                    "seed": seed,
                    "weight": weight,
                    "trials_per_fold": trials_per_fold,
                    "tested": tested,
                    "complete_fold_sweep": False,
                    "rank_counts": dict(sorted(ranks.items())),
                    "witnesses": sorted(
                        witnesses,
                        key=lambda item: (
                            not item["pairing_is_permutation"],
                            item["zx_pairing_weight"],
                            item["seed_support"],
                        ),
                    ),
                    "seconds": round(time.perf_counter() - started, 6),
                }
    return {
        "seed": seed,
        "weight": weight,
        "trials_per_fold": trials_per_fold,
        "tested": tested,
        "complete_fold_sweep": True,
        "rank_counts": dict(sorted(ranks.items())),
        "witnesses": sorted(
            witnesses,
            key=lambda item: (
                not item["pairing_is_permutation"],
                item["zx_pairing_weight"],
                item["seed_support"],
            ),
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


@functools.lru_cache(maxsize=1)
def polynomial_terms() -> tuple[ProductMonomial, ...]:
    return tuple(
        ProductMonomial(top, xx, yy)
        for top in TOP_MATRIX_BASIS
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    )


@functools.lru_cache(maxsize=1)
def polynomial_term_lifts() -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(linear_ring_monomial(term).lift(), dtype=np.uint8)
        for term in polynomial_terms()
    )


def _basis_checks(entry: int, lift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Checks contributed by one monomial in ``F0,F1,G0,G1``."""
    if entry not in range(4):
        raise ValueError("entry index must select F0, F1, G0, or G1")
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
def packed_coefficient_check_rows() -> tuple[tuple[int, ...], ...]:
    """Return packed ``H_X`` rows for every independent coefficient variable."""
    packed: list[tuple[int, ...]] = []
    for entry in range(4):
        for lift in polynomial_term_lifts():
            matrix_x, _matrix_z = _basis_checks(entry, lift)
            packed.append(
                tuple(
                    int.from_bytes(np.packbits(row).tobytes()) for row in matrix_x
                )
            )
    return tuple(packed)


def gf2_rank_packed_rows(rows: Sequence[int]) -> int:
    """Rank binary row vectors represented as Python integers."""
    basis: dict[int, int] = {}
    for raw_value in rows:
        value = int(raw_value)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def _permute_rows(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[np.asarray(permutation, dtype=int)] = matrix
    return output


def polynomial_constraint_matrix(
    seed_support: Sequence[int],
    *,
    data_fold: StructuredGLFold,
    check_fold: StructuredGLFold,
) -> np.ndarray:
    """Return constraints for a kernel seed and an exact one-way ZX fold.

    There are 768 coefficient variables, ordered by the four ring entries and
    then by ``polynomial_terms()``.  A coefficient vector lies in the kernel
    precisely when the resulting ``H_X`` annihilates the seed and

    ``H_Z = check_fold(H_X data_fold)``.

    CSS orthogonality and the reverse fold direction remain nonlinear/rank
    conditions and are checked after sparse synthesis.
    """
    if data_fold.num_blocks != NUM_PHYSICAL_BLOCKS:
        raise ValueError("data fold must act on four physical blocks")
    if check_fold.num_blocks != NUM_CHECK_BLOCKS:
        raise ValueError("check fold must act on two check blocks")
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(seed_support)] = 1
    columns: list[np.ndarray] = []
    for entry in range(4):
        for lift in polynomial_term_lifts():
            matrix_x, matrix_z = _basis_checks(entry, lift)
            folded_x = _permute_rows(
                _permute_columns(matrix_x, data_fold.permutation),
                check_fold.permutation,
            )
            columns.append(
                np.concatenate([(matrix_x @ seed) % 2, (matrix_z ^ folded_x).ravel()])
            )
    return np.asarray(columns, dtype=np.uint8).T


def add_automatic_css_relation(
    constraint_matrix: np.ndarray,
    relation: str,
    *,
    x_shift: int = 0,
    y_shift: int = 0,
) -> np.ndarray:
    """Restrict ``G`` to an automatic-CSS linear image of ``F``.

    For the two-entry block-circulant protograph, either ``G=qF`` or
    ``(G0,G1)=q(F1,F0)`` cancels both aggregate commutators in characteristic
    two when the common factor ``q=x^u y^v`` is central.  The individual
    entries may still contain noncommuting GL matrices.
    """
    if relation not in {"independent", "identity", "swap"}:
        raise ValueError(f"unknown CSS relation: {relation}")
    matrix = np.asarray(constraint_matrix, dtype=np.uint8)
    if relation == "independent":
        if x_shift % BOTTOM_X_ORDER or y_shift % BOTTOM_Y_ORDER:
            raise ValueError("CSS shifts require an identity or swap relation")
        return matrix
    entry_size = len(polynomial_terms())
    equations = np.zeros((2 * entry_size, 4 * entry_size), dtype=np.uint8)
    source_entries = (0, 1) if relation == "identity" else (1, 0)
    term_indices = {term: index for index, term in enumerate(polynomial_terms())}
    for target_offset, source_entry in enumerate(source_entries):
        target_entry = 2 + target_offset
        for source_index, term in enumerate(polynomial_terms()):
            target_term = ProductMonomial(
                term.top,
                (term.x + x_shift) % BOTTOM_X_ORDER,
                (term.y + y_shift) % BOTTOM_Y_ORDER,
            )
            target_index = term_indices[target_term]
            equation = target_offset * entry_size + source_index
            equations[equation, source_entry * entry_size + source_index] = 1
            equations[equation, target_entry * entry_size + target_index] = 1
    return np.vstack([matrix, equations])


def independent_constraint_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-space basis using Python integers for fast elimination."""
    binary = np.asarray(matrix, dtype=np.uint8)
    num_columns = binary.shape[1]
    byte_count = (num_columns + 7) // 8
    basis: dict[int, int] = {}
    for packed in np.packbits(binary, axis=1):
        value = int.from_bytes(packed.tobytes())
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    rows = []
    for value in basis.values():
        packed = np.frombuffer(value.to_bytes(byte_count), dtype=np.uint8)
        rows.append(np.unpackbits(packed)[:num_columns])
    return np.asarray(rows, dtype=np.uint8)


def solve_sparse_polynomial_constraints(
    constraint_matrix: np.ndarray,
    *,
    maximum_terms: int = 16,
    time_limit: float = 30,
    minimize_terms: bool = True,
    enumeration_dimension_limit: int = 24,
    rank_probe_trials: int = 0,
    rank_probe_restarts: int = 0,
    random_seed: int = 0,
    probe_only: bool = False,
) -> dict[str, Any]:
    """Use HiGHS to find a sparse nonzero polynomial in every GALA entry."""
    import galois

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, eye, hstack, lil_matrix, vstack

    if maximum_terms < 4:
        raise ValueError("at least one term is required in each of four entries")
    rows = independent_constraint_rows(constraint_matrix)
    num_constraints, num_coefficients = rows.shape
    nullity = num_coefficients - num_constraints
    field = galois.GF(2)
    basis = np.asarray(
        field(np.asarray(rows, dtype=int)).null_space(), dtype=np.uint8
    )
    terms = polynomial_terms()

    coefficient_checks = np.asarray(packed_coefficient_check_rows(), dtype=object)
    zero_check = np.zeros(NUM_CHECKS, dtype=object)
    lifted_basis = []
    for vector in basis:
        selected = np.flatnonzero(vector)
        lifted_basis.append(
            np.bitwise_xor.reduce(coefficient_checks[selected], axis=0)
            if len(selected)
            else zero_check.copy()
        )
    lifted_basis_array = np.asarray(lifted_basis, dtype=object)
    rank_capacity = (
        gf2_rank_packed_rows(lifted_basis_array.ravel()) if len(lifted_basis) else 0
    )
    required_rank = (NUM_QUBITS - TARGET_DIMENSION) // 2
    common: dict[str, Any] = {
        "constraint_rows": int(constraint_matrix.shape[0]),
        "constraint_variables": int(constraint_matrix.shape[1]),
        "constraint_rank": int(num_constraints),
        "constraint_nullity": int(nullity),
        "maximum_possible_rank_x_upper_bound": int(rank_capacity),
        "required_rank_x_for_k32": required_rank,
        "minimum_possible_k_lower_bound": NUM_QUBITS - 2 * rank_capacity,
        "maximum_terms": maximum_terms,
        "time_limit": time_limit,
        "minimize_terms": minimize_terms,
        "enumeration_dimension_limit": enumeration_dimension_limit,
        "rank_probe_trials": rank_probe_trials,
        "rank_probe_restarts": rank_probe_restarts,
        "random_seed": random_seed,
        "entries": None,
    }
    if rank_capacity < required_rank:
        return {
            **common,
            "search_method": "exact_linear_rank_capacity",
            "solver_status": 3,
            "solver_message": (
                "the entire constrained polynomial space has insufficient "
                "binary check-row capacity for k=32"
            ),
            "seconds": 0.0,
        }

    probe: dict[str, Any] | None = None
    if rank_probe_trials or rank_probe_restarts:
        rng = random.Random(random_seed)
        rank_counts: Counter[int] = Counter()
        best_rank = -1
        best_weight: int | None = None
        best_vector: np.ndarray | None = None
        target_weight: int | None = None
        target_vector: np.ndarray | None = None

        def score(generator_bits: np.ndarray) -> tuple[int, int, np.ndarray]:
            selected = np.flatnonzero(generator_bits)
            coefficients = (
                np.bitwise_xor.reduce(basis[selected], axis=0)
                if len(selected)
                else np.zeros(num_coefficients, dtype=np.uint8)
            )
            check_rows = (
                np.bitwise_xor.reduce(lifted_basis_array[selected], axis=0)
                if len(selected)
                else zero_check
            )
            return (
                gf2_rank_packed_rows(check_rows),
                int(np.count_nonzero(coefficients)),
                coefficients,
            )

        def retain(rank: int, weight: int, coefficients: np.ndarray) -> None:
            nonlocal best_rank, best_weight, best_vector, target_weight, target_vector
            rank_counts[rank] += 1
            if rank > best_rank or (
                rank == best_rank and (best_weight is None or weight < best_weight)
            ):
                best_rank = rank
                best_weight = weight
                best_vector = coefficients.copy()
            if rank == required_rank and (
                target_weight is None or weight < target_weight
            ):
                target_weight = weight
                target_vector = coefficients.copy()

        for _trial in range(rank_probe_trials):
            generators = np.fromiter(
                (rng.getrandbits(1) for _ in range(nullity)),
                dtype=np.uint8,
                count=nullity,
            )
            retain(*score(generators))

        for _restart in range(rank_probe_restarts):
            generators = np.fromiter(
                (rng.getrandbits(1) for _ in range(nullity)),
                dtype=np.uint8,
                count=nullity,
            )
            current_rank, current_weight, current_vector = score(generators)
            retain(current_rank, current_weight, current_vector)
            while True:
                neighbor_scores = []
                for index in range(nullity):
                    generators[index] ^= 1
                    neighbor_scores.append((*score(generators)[:2], index))
                    generators[index] ^= 1
                maximum_rank = max(item[0] for item in neighbor_scores)
                if maximum_rank <= current_rank:
                    break
                choices = [item for item in neighbor_scores if item[0] == maximum_rank]
                _rank, _weight, chosen = min(choices, key=lambda item: item[1])
                generators[chosen] ^= 1
                current_rank, current_weight, current_vector = score(generators)
                retain(current_rank, current_weight, current_vector)

        probe = {
            "samples_scored": int(sum(rank_counts.values())),
            "rank_counts": dict(sorted(rank_counts.items())),
            "maximum_observed_rank_x": best_rank,
            "best_rank_coefficient_weight": best_weight,
            "reached_required_rank": best_rank >= required_rank,
            "target_rank_observed": target_vector is not None,
            "best_target_rank_coefficient_weight": target_weight,
            "target_rank_coefficients": (
                np.flatnonzero(target_vector).astype(int).tolist()
                if target_vector is not None
                else None
            ),
            "best_coefficients": (
                np.flatnonzero(best_vector).astype(int).tolist()
                if best_vector is not None
                else None
            ),
        }
        common["rank_probe"] = probe
        if target_vector is not None:
            chosen = np.flatnonzero(target_vector).astype(int)
            entries: list[list[dict[str, int]]] = []
            for entry in range(4):
                local = chosen[(chosen // len(terms)) == entry] % len(terms)
                entries.append(
                    [
                        {
                            "top": terms[index].top,
                            "x": terms[index].x,
                            "y": terms[index].y,
                        }
                        for index in local
                    ]
                )
            common.update(
                {
                    "objective_terms": int(len(chosen)),
                    "chosen_coefficients": chosen.tolist(),
                    "entries": entries,
                }
            )
    if probe_only:
        return {
            **common,
            "search_method": "rank_probe",
            "solver_status": 4,
            "solver_message": "rank probe completed without sparse optimization",
            "seconds": 0.0,
        }
    if nullity <= enumeration_dimension_limit:
        started = time.perf_counter()
        current = np.zeros(num_coefficients, dtype=np.uint8)
        previous_gray = 0
        best_weight = num_coefficients + 1
        best: np.ndarray | None = None
        entry_size = len(polynomial_terms())
        for integer in range(1, 1 << nullity):
            gray = integer ^ (integer >> 1)
            changed = gray ^ previous_gray
            current ^= basis[changed.bit_length() - 1]
            previous_gray = gray
            weight = int(np.count_nonzero(current))
            if weight >= best_weight:
                continue
            if any(
                not np.any(current[entry * entry_size : (entry + 1) * entry_size])
                for entry in range(4)
            ):
                continue
            best_weight = weight
            best = current.copy()
        output = {
            **common,
            "search_method": "exhaustive_nullspace",
            "nullspace_vectors_tested": (1 << nullity) - 1,
            "exact_minimum_terms_with_all_entries": (
                best_weight if best is not None else None
            ),
            "solver_status": 0 if best is not None and best_weight <= maximum_terms else 2,
            "solver_message": (
                "exact sparse solution found"
                if best is not None and best_weight <= maximum_terms
                else "exhaustive nullspace has no solution under the term ceiling"
            ),
            "seconds": round(time.perf_counter() - started, 6),
        }
        if best is None or best_weight > maximum_terms:
            return output
        chosen = np.flatnonzero(best).astype(int)
        entries: list[list[dict[str, int]]] = []
        for entry in range(4):
            local = chosen[(chosen // len(terms)) == entry] % len(terms)
            entries.append(
                [
                    {
                        "top": terms[index].top,
                        "x": terms[index].x,
                        "y": terms[index].y,
                    }
                    for index in local
                ]
            )
        output.update(
            {
                "objective_terms": int(len(chosen)),
                "chosen_coefficients": chosen.tolist(),
                "entries": entries,
            }
        )
        return output

    num_variables = num_coefficients + num_constraints
    parity = hstack(
        [csr_matrix(rows, dtype=float), -2 * eye(num_constraints, format="csr")],
        format="csr",
    )
    selectors = lil_matrix((5, num_variables), dtype=float)
    for entry in range(4):
        start = entry * len(polynomial_terms())
        stop = (entry + 1) * len(polynomial_terms())
        selectors[entry, start:stop] = 1
    selectors[4, :num_coefficients] = 1
    matrix = vstack([parity, selectors.tocsr()], format="csr")
    lower = np.concatenate([np.zeros(num_constraints), np.ones(4), [4]])
    upper = np.concatenate(
        [
            np.zeros(num_constraints),
            np.full(4, maximum_terms),
            [maximum_terms],
        ]
    )
    variable_lower = np.zeros(num_variables)
    variable_upper = np.ones(num_variables)
    row_weights = np.sum(rows, axis=1)
    variable_upper[num_coefficients:] = np.maximum(1, row_weights // 2)
    objective = np.zeros(num_variables)
    if minimize_terms:
        objective[:num_coefficients] = 1
    started = time.perf_counter()
    result = milp(
        objective,
        integrality=np.ones(num_variables),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0},
    )
    output: dict[str, Any] = {
        **common,
        "search_method": "milp",
        "solver_status": int(result.status),
        "solver_message": result.message,
        "seconds": round(time.perf_counter() - started, 6),
    }
    if result.x is None:
        return output
    chosen = np.flatnonzero(result.x[:num_coefficients] > 0.5).astype(int)
    terms = polynomial_terms()
    entries: list[list[dict[str, int]]] = []
    for entry in range(4):
        local = chosen[(chosen // len(terms)) == entry] % len(terms)
        entries.append(
            [
                {"top": terms[index].top, "x": terms[index].x, "y": terms[index].y}
                for index in local
            ]
        )
    output.update(
        {
            "objective_terms": int(len(chosen)),
            "chosen_coefficients": chosen.tolist(),
            "entries": entries,
        }
    )
    return output


def deserialize_entries(
    entries: Sequence[Sequence[dict[str, int]]],
) -> tuple[tuple[ProductMonomial, ...], ...]:
    if len(entries) != 4:
        raise ValueError("expected F0, F1, G0, and G1")
    return tuple(
        tuple(sorted(ProductMonomial(**term) for term in entry))
        for entry in entries
    )


def build_half_grid_code(
    entries: Sequence[Sequence[ProductMonomial]],
) -> codes.CSSCode:
    """Build the full ``L=4,J=2`` code with actual binary transposes."""
    if len(entries) != 4:
        raise ValueError("expected F0, F1, G0, and G1")
    lifted = tuple(linear_ring_polynomial(tuple(entry)) for entry in entries)
    return _build_linear_css_code(lifted[:2], lifted[2:], NUM_CHECK_BLOCKS)


def analyze_polynomial_solution(
    entries: Sequence[Sequence[ProductMonomial]],
    *,
    seed_support: Sequence[int],
    data_fold: StructuredGLFold,
    check_fold: StructuredGLFold,
) -> dict[str, Any]:
    """Apply all cheap exact checks to a synthesized polynomial candidate."""
    code = build_half_grid_code(entries)
    matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
    matrix_z = np.asarray(code.matrix_z, dtype=np.uint8)
    folded_x = _permute_rows(
        _permute_columns(matrix_x, data_fold.permutation),
        check_fold.permutation,
    )
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    checks = {
        "exact_forward_fold": bool(np.array_equal(matrix_z, folded_x)),
        "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
        "reverse_fold": is_zx_fold(code, data_fold.permutation),
        "exact_dimension_32": code.dimension == TARGET_DIMENSION,
    }
    grid = analyze_linear_translation_seed(
        code, seed_support, fold=data_fold.permutation
    )
    checks.update(
        {
            "seed_is_Z_logical_grid": bool(
                grid["z_orbit_in_kernel"]
                and grid["graph_supported"]
                and grid["pairwise_disjoint"]
                and grid["orbit_rank_mod_stabilizers"] == NUM_LOGICALS
            ),
            "folded_X_grid_in_kernel": grid["x_orbit_in_kernel"],
            "ZX_pairing_rank_32": grid["zx_pairing_rank"] == NUM_LOGICALS,
        }
    )
    return {
        "n": code.num_qubits,
        "k": code.dimension,
        "rank_x": code.code_x.rank,
        "rank_z": code.code_z.rank,
        "checks": checks,
        "accepted_structurally": all(checks.values()),
        "row_weights_x": sorted(set(map(int, row_weights_x))),
        "row_weights_z": sorted(set(map(int, row_weights_z))),
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "logical_grid": grid,
    }
