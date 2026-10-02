#!/usr/bin/env python3
"""Preflight a one-grid GL(2,2) x C4 x C4 GALA search at n=128.

This script deliberately stops before sparse generator enumeration.  It tests
the two failure gates exposed by the earlier n=32 single-row search:

1. a weight-six graph seed must have 16 disjoint translates and a nonsingular
   logical ZX pairing under a translation-commuting physical fold;
2. after imposing the seed-kernel and exact forward-fold equations, the full
   linear generator space must have enough binary row-rank capacity for
   rank(H_X) = rank(H_Z) = 56 (and hence k=16).

For spaces that pass gate 2, it also tests the simple automatic-CSS families
G=qF and G=q swap(F), for every monomial q in C4 x C4.  Results are written
incrementally and an existing run directory is never overwritten.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import pathlib
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


X_ORDER = 4
Y_ORDER = 4
BOTTOM_ORDER = X_ORDER * Y_ORDER
TOP_DIMENSION = 2
BLOCK_SIZE = TOP_DIMENSION * BOTTOM_ORDER
NUM_DATA_BLOCKS = 4
NUM_CHECK_BLOCKS = 2
NUM_QUBITS = NUM_DATA_BLOCKS * BLOCK_SIZE
NUM_CHECKS = NUM_CHECK_BLOCKS * BLOCK_SIZE
NUM_SHEETS = NUM_DATA_BLOCKS * TOP_DIMENSION
NUM_LOGICALS = BOTTOM_ORDER
TARGET_DIMENSION = NUM_LOGICALS
TARGET_CHECK_RANK = (NUM_QUBITS - TARGET_DIMENSION) // 2
ENTRY_NAMES = ("F0", "F1", "G0", "G1")

# Four GL(2,2) elements whose binary lifts span M_2(F_2).  This avoids the
# two-dimensional kernel introduced by treating all six S3 elements as
# independent polynomial coefficients.
TOP_MATRIX_BASIS = (
    np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
    np.asarray([[1, 1], [0, 1]], dtype=np.uint8),
    np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
    np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
)
TERMS_PER_ENTRY = len(TOP_MATRIX_BASIS) * BOTTOM_ORDER
NUM_COEFFICIENTS = len(ENTRY_NAMES) * TERMS_PER_ENTRY


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    row = 0
    pivots: list[int] = []
    for column in range(reduced.shape[1]):
        choices = np.flatnonzero(reduced[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        other = np.flatnonzero(reduced[:, column])
        other = other[other != row]
        if len(other):
            reduced[other] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced[:row], tuple(pivots)


def gf2_rank(matrix: np.ndarray) -> int:
    return len(gf2_rref(matrix)[1])


def gf2_nullspace_from_rref(
    reduced: np.ndarray, pivots: Sequence[int], num_columns: int
) -> np.ndarray:
    pivot_set = set(pivots)
    free = [column for column in range(num_columns) if column not in pivot_set]
    basis = np.zeros((len(free), num_columns), dtype=np.uint8)
    for basis_row, free_column in enumerate(free):
        basis[basis_row, free_column] = 1
        for row, pivot in enumerate(pivots):
            basis[basis_row, pivot] = reduced[row, free_column]
    return basis


def packed_row_rank(matrices: Iterable[np.ndarray]) -> int:
    basis: dict[int, int] = {}
    for matrix in matrices:
        for row in np.asarray(matrix, dtype=np.uint8):
            value = int.from_bytes(np.packbits(row).tobytes())
            while value:
                pivot = value.bit_length() - 1
                if pivot in basis:
                    value ^= basis[pivot]
                else:
                    basis[pivot] = value
                    break
    return len(basis)


def bottom_index(x: int, y: int) -> int:
    return (x % X_ORDER) * Y_ORDER + (y % Y_ORDER)


def bottom_point(index: int) -> tuple[int, int]:
    return divmod(index, Y_ORDER)


@functools.lru_cache(maxsize=None)
def bottom_shift_matrix(dx: int, dy: int) -> np.ndarray:
    matrix = np.zeros((BOTTOM_ORDER, BOTTOM_ORDER), dtype=np.uint8)
    for source in range(BOTTOM_ORDER):
        x, y = bottom_point(source)
        target = bottom_index(x + dx, y + dy)
        matrix[target, source] = 1
    return matrix


@functools.lru_cache(maxsize=None)
def coefficient_lift(top: int, x: int, y: int) -> np.ndarray:
    return np.kron(TOP_MATRIX_BASIS[top], bottom_shift_matrix(x, y)).astype(
        np.uint8
    )


def _basis_checks(entry: int, lift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        _basis_checks(entry, coefficient_lift(top, x, y))
        for entry in range(4)
        for top in range(4)
        for x in range(X_ORDER)
        for y in range(Y_ORDER)
    )


def coefficient_index(entry: int, top: int, x: int, y: int) -> int:
    return (
        entry * TERMS_PER_ENTRY
        + top * BOTTOM_ORDER
        + bottom_index(x, y)
    )


def _permute_columns(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, np.asarray(permutation, dtype=int)] = matrix
    return output


def _permute_rows(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[np.asarray(permutation, dtype=int)] = matrix
    return output


@dataclass(frozen=True)
class StructuredGLFold:
    """Translation-commuting top swap with coupled protograph permutations."""

    forward_blocks: tuple[int, ...]
    backward_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = list(range(len(self.forward_blocks)))
        if sorted(self.forward_blocks) != expected:
            raise ValueError("forward block map must be a permutation")
        if sorted(self.backward_blocks) != expected:
            raise ValueError("backward block map must be a permutation")

    @property
    def num_blocks(self) -> int:
        return len(self.forward_blocks)

    @property
    def permutation(self) -> np.ndarray:
        output = np.empty(self.num_blocks * BLOCK_SIZE, dtype=int)
        bottom = np.arange(BOTTOM_ORDER)
        for block in range(self.num_blocks):
            base = block * BLOCK_SIZE
            forward = self.forward_blocks[block] * BLOCK_SIZE
            backward = self.backward_blocks[block] * BLOCK_SIZE
            output[base + bottom] = forward + BOTTOM_ORDER + bottom
            output[base + BOTTOM_ORDER + bottom] = backward + bottom
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_blocks": list(self.forward_blocks),
            "backward_blocks": list(self.backward_blocks),
            "top_independent": self.forward_blocks == self.backward_blocks,
        }


def iter_structured_gl_folds(num_blocks: int) -> Iterable[StructuredGLFold]:
    permutations = tuple(itertools.permutations(range(num_blocks)))
    for forward in permutations:
        for backward in permutations:
            yield StructuredGLFold(forward, backward)


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    rows = []
    for dx in range(X_ORDER):
        for dy in range(Y_ORDER):
            translated = np.zeros_like(seed)
            for sheet in range(NUM_SHEETS):
                section = seed[
                    sheet * BOTTOM_ORDER : (sheet + 1) * BOTTOM_ORDER
                ]
                translated[
                    sheet * BOTTOM_ORDER : (sheet + 1) * BOTTOM_ORDER
                ] = bottom_shift_matrix(dx, dy) @ section
            rows.append(translated)
    return np.asarray(rows, dtype=np.uint8)


def canonical_support(support: Sequence[int]) -> tuple[int, ...]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return min(
        tuple(np.flatnonzero(row).astype(int).tolist())
        for row in translation_orbit(seed)
    )


def random_graph_seed(rng: random.Random, weight: int) -> tuple[int, ...]:
    sheets = rng.sample(range(NUM_SHEETS), weight)
    support = [
        sheet * BOTTOM_ORDER + rng.randrange(BOTTOM_ORDER) for sheet in sheets
    ]
    return canonical_support(support)


def analyze_seed_fold(
    support: Sequence[int], fold: StructuredGLFold
) -> dict[str, Any]:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = translation_orbit(seed)
    folded = _permute_columns(orbit, fold.permutation)
    pairing = (orbit @ folded.T) % 2
    occupancies = [
        int(np.count_nonzero(seed[s * BOTTOM_ORDER : (s + 1) * BOTTOM_ORDER]))
        for s in range(NUM_SHEETS)
    ]
    return {
        "seed_support": list(canonical_support(support)),
        "weight": int(np.count_nonzero(seed)),
        "sheet_occupancies": occupancies,
        "graph_supported": max(occupancies, default=0) <= 1,
        "pairwise_disjoint": bool(np.all(np.sum(orbit, axis=0) <= 1)),
        "physical_orbit_rank": gf2_rank(orbit),
        "zx_pairing_rank": gf2_rank(pairing),
        "zx_pairing_weight": int(np.count_nonzero(pairing)),
        "pairing_is_permutation": bool(
            np.all(np.count_nonzero(pairing, axis=0) == 1)
            and np.all(np.count_nonzero(pairing, axis=1) == 1)
        ),
        "pairing_row_augmentation": int(np.sum(pairing[0]) % 2),
    }


def search_geometry(
    *, rng_seed: int, weight: int, trials_per_fold: int, target: int
) -> dict[str, Any]:
    rng = random.Random(rng_seed)
    ranks: Counter[int] = Counter()
    witnesses: list[dict[str, Any]] = []
    tested = 0
    standard_max = 0
    started = time.perf_counter()
    for fold_index, fold in enumerate(iter_structured_gl_folds(NUM_DATA_BLOCKS)):
        for _ in range(trials_per_fold):
            support = random_graph_seed(rng, weight)
            result = analyze_seed_fold(support, fold)
            tested += 1
            rank = result["zx_pairing_rank"]
            ranks[rank] += 1
            if fold.forward_blocks == fold.backward_blocks:
                standard_max = max(standard_max, rank)
            if rank == NUM_LOGICALS:
                witnesses.append(
                    {
                        "witness_index": len(witnesses),
                        "fold_index": fold_index,
                        "fold": fold.to_dict(),
                        **result,
                    }
                )
                if len(witnesses) >= target:
                    return {
                        "rng_seed": rng_seed,
                        "weight": weight,
                        "trials_per_fold": trials_per_fold,
                        "tested": tested,
                        "complete_fold_sweep": False,
                        "rank_counts": dict(sorted(ranks.items())),
                        "standard_fold_maximum_rank": standard_max,
                        "seconds": time.perf_counter() - started,
                        "witnesses": witnesses,
                    }
    return {
        "rng_seed": rng_seed,
        "weight": weight,
        "trials_per_fold": trials_per_fold,
        "tested": tested,
        "complete_fold_sweep": True,
        "rank_counts": dict(sorted(ranks.items())),
        "standard_fold_maximum_rank": standard_max,
        "seconds": time.perf_counter() - started,
        "witnesses": witnesses,
    }


def fold_from_dict(record: dict[str, Any]) -> StructuredGLFold:
    return StructuredGLFold(
        tuple(record["forward_blocks"]), tuple(record["backward_blocks"])
    )


@functools.lru_cache(maxsize=None)
def forward_fold_columns(
    data_forward: tuple[int, ...],
    data_backward: tuple[int, ...],
    check_forward: tuple[int, ...],
    check_backward: tuple[int, ...],
) -> np.ndarray:
    data_fold = StructuredGLFold(data_forward, data_backward)
    check_fold = StructuredGLFold(check_forward, check_backward)
    columns = []
    for matrix_x, matrix_z in coefficient_checks():
        folded_x = _permute_rows(
            _permute_columns(matrix_x, data_fold.permutation),
            check_fold.permutation,
        )
        columns.append((matrix_z ^ folded_x).ravel())
    return np.asarray(columns, dtype=np.uint8).T


def seed_constraint_columns(support: Sequence[int]) -> np.ndarray:
    seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
    seed[list(support)] = 1
    return np.asarray(
        [(matrix_x @ seed) % 2 for matrix_x, _ in coefficient_checks()],
        dtype=np.uint8,
    ).T


def automatic_css_equations(
    relation: str, q_support: Sequence[tuple[int, int]]
) -> np.ndarray:
    if relation not in {"identity", "swap"}:
        raise ValueError("relation must be identity or swap")
    if not q_support:
        raise ValueError("q must be nonzero")
    equations = np.zeros((2 * TERMS_PER_ENTRY, NUM_COEFFICIENTS), dtype=np.uint8)
    sources = (0, 1) if relation == "identity" else (1, 0)
    for target_offset, source_entry in enumerate(sources):
        target_entry = 2 + target_offset
        for top in range(4):
            for tx in range(X_ORDER):
                for ty in range(Y_ORDER):
                    row = target_offset * TERMS_PER_ENTRY + top * BOTTOM_ORDER + bottom_index(tx, ty)
                    equations[row, coefficient_index(target_entry, top, tx, ty)] = 1
                    for qx, qy in q_support:
                        equations[
                            row,
                            coefficient_index(source_entry, top, tx - qx, ty - qy),
                        ] ^= 1
    return equations


def rank_capacity(constraints: np.ndarray) -> dict[str, int]:
    reduced, pivots = gf2_rref(constraints)
    basis = gf2_nullspace_from_rref(reduced, pivots, NUM_COEFFICIENTS)
    capacity_x = packed_row_rank(
        np.bitwise_xor.reduce(
            np.asarray([coefficient_checks()[index][0] for index in np.flatnonzero(vector)]),
            axis=0,
        )
        if np.any(vector)
        else np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
        for vector in basis
    )
    capacity_z = packed_row_rank(
        np.bitwise_xor.reduce(
            np.asarray([coefficient_checks()[index][1] for index in np.flatnonzero(vector)]),
            axis=0,
        )
        if np.any(vector)
        else np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
        for vector in basis
    )
    return {
        "constraint_rank": len(pivots),
        "constraint_nullity": len(basis),
        "rank_capacity_x": capacity_x,
        "rank_capacity_z": capacity_z,
        "required_rank": TARGET_CHECK_RANK,
        "passes": int(min(capacity_x, capacity_z) >= TARGET_CHECK_RANK),
    }


def append_jsonl(path: pathlib.Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", default="gl2-c4xc4-l4-j2-preflight-v1")
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--weight", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--trials-per-fold", type=int, default=20)
    parser.add_argument("--target-witnesses", type=int, default=20)
    parser.add_argument("--capacity-witnesses", type=int, default=10)
    parser.add_argument("--skip-automatic-css", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()

    geometry = search_geometry(
        rng_seed=args.seed,
        weight=args.weight,
        trials_per_fold=args.trials_per_fold,
        target=args.target_witnesses,
    )
    witnesses = geometry.pop("witnesses")
    for witness in witnesses:
        append_jsonl(output_dir / "geometry-witnesses.jsonl", witness)
    write_json(output_dir / "geometry-summary.json", geometry)
    print(
        json.dumps(
            {
                "stage": "geometry",
                "tested": geometry["tested"],
                "witnesses": len(witnesses),
                "rank_counts": geometry["rank_counts"],
                "standard_fold_maximum_rank": geometry["standard_fold_maximum_rank"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    independent_passes: list[tuple[dict[str, Any], StructuredGLFold, np.ndarray]] = []
    capacity_file = output_dir / "capacity.jsonl"
    check_folds = tuple(iter_structured_gl_folds(NUM_CHECK_BLOCKS))
    selected = witnesses[: args.capacity_witnesses]
    attempt = 0
    for witness in selected:
        data_fold = fold_from_dict(witness["fold"])
        seed_constraints = seed_constraint_columns(witness["seed_support"])
        for check_fold_index, check_fold in enumerate(check_folds):
            attempt += 1
            base = np.vstack(
                [
                    seed_constraints,
                    forward_fold_columns(
                        data_fold.forward_blocks,
                        data_fold.backward_blocks,
                        check_fold.forward_blocks,
                        check_fold.backward_blocks,
                    ),
                ]
            )
            capacity = rank_capacity(base)
            record = {
                "stage": "independent_exact_fold",
                "attempt": attempt,
                "witness_index": witness["witness_index"],
                "check_fold_index": check_fold_index,
                "check_fold": check_fold.to_dict(),
                **capacity,
            }
            append_jsonl(capacity_file, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if capacity["passes"]:
                independent_passes.append((witness, check_fold, base))

    automatic_records = 0
    automatic_passes = 0
    if not args.skip_automatic_css:
        for witness, check_fold, base in independent_passes:
            for relation in ("identity", "swap"):
                for qx in range(X_ORDER):
                    for qy in range(Y_ORDER):
                        automatic_records += 1
                        capacity = rank_capacity(
                            np.vstack(
                                [base, automatic_css_equations(relation, ((qx, qy),))]
                            )
                        )
                        automatic_passes += capacity["passes"]
                        record = {
                            "stage": "automatic_css_monomial",
                            "attempt": automatic_records,
                            "witness_index": witness["witness_index"],
                            "check_fold": check_fold.to_dict(),
                            "relation": relation,
                            "q": {"x": qx, "y": qy},
                            **capacity,
                        }
                        append_jsonl(capacity_file, record)
                        if automatic_records % 16 == 0 or capacity["passes"]:
                            print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "target": {
            "n": NUM_QUBITS,
            "target_k": TARGET_DIMENSION,
            "target_distance": 6,
            "target_check_rank_each_sector": TARGET_CHECK_RANK,
            "logical_translation_group": "C4 x C4",
            "represented_top": "GL(2,2) ~= S3",
            "protograph": {"L": 4, "J": 2},
        },
        "arguments": {**vars(args), "output_root": str(args.output_root)},
        "geometry": geometry,
        "geometry_witnesses": len(witnesses),
        "independent_spaces_tested": attempt,
        "independent_spaces_passing_rank_capacity": len(independent_passes),
        "automatic_css_monomial_spaces_tested": automatic_records,
        "automatic_css_monomial_spaces_passing_rank_capacity": automatic_passes,
        "seconds": time.perf_counter() - started,
        "interpretation": {
            "geometry_gate_passed": bool(witnesses),
            "independent_exact_fold_capacity_gate_passed": bool(independent_passes),
            "simple_automatic_css_gate_passed": bool(automatic_passes),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
