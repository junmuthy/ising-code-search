"""Sparse-``F``, affine-solved-``G`` row-space ZX search at ``n=192``."""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from qldpc import codes

from gala_search.s3_ising import find_logical_up_to_weight_four
from gala_search.single_row import gf2_rank, gf2_rref
from searches.reverse_geometry.n192 import (
    BLOCK_SIZE,
    LATTICE_ORDER,
    MAXIMUM_CHECK_WEIGHT,
    NUM_HALVES,
    NUM_QUBITS,
    TARGET_LOGICALS,
    NaturalS3Fold,
    _has_nonzero_top_augmentation,
    _permutation_matrix,
    _serialize_term,
    fold_internal_matrix,
    represented_terms,
    standard_transposition_fold,
    tanner_connected,
    translation_orbit,
)


def _xor_terms(coefficients: np.ndarray) -> np.ndarray:
    selected = np.flatnonzero(coefficients)
    return (
        np.bitwise_xor.reduce(represented_terms()[selected], axis=0)
        if len(selected)
        else np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.uint8)
    )


def data_fold_matrix(fold: NaturalS3Fold | None = None) -> np.ndarray:
    fold = fold or standard_transposition_fold()
    qq = fold_internal_matrix(fold)
    output = np.zeros((NUM_QUBITS, NUM_QUBITS), dtype=np.uint8)
    output[:BLOCK_SIZE, :BLOCK_SIZE] = qq
    output[BLOCK_SIZE:, BLOCK_SIZE:] = qq
    return output


def checks_from_coefficients(
    coefficients_f: np.ndarray, coefficients_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ff = _xor_terms(coefficients_f)
    gg = _xor_terms(coefficients_g)
    return (
        np.hstack([ff, gg]).astype(np.uint8),
        np.hstack([gg.T, ff.T]).astype(np.uint8),
    )


def displayed_partner_coefficients(coefficients_f: np.ndarray) -> np.ndarray:
    """Return coefficients for ``G=Q F^T Q`` under the standard fold."""
    qq = fold_internal_matrix(standard_transposition_fold())
    terms = represented_terms()
    lookup = {matrix.tobytes(): index for index, matrix in enumerate(terms)}
    output = np.zeros(len(terms), dtype=np.uint8)
    for index in np.flatnonzero(coefficients_f):
        transformed = (qq @ terms[int(index)].T @ qq) % 2
        output[lookup[transformed.tobytes()]] ^= 1
    return output


def rowspace_zx_dual(
    matrix_x: np.ndarray, matrix_z: np.ndarray, *, fold: NaturalS3Fold | None = None
) -> bool:
    transformed_x = (matrix_x @ data_fold_matrix(fold).T) % 2
    transformed_z = (matrix_z @ data_fold_matrix(fold).T) % 2
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    return bool(
        gf2_rank(np.vstack([matrix_z, transformed_x])) == rank_z
        and gf2_rank(np.vstack([matrix_x, transformed_z])) == rank_x
    )


def affine_g_system(
    seed: np.ndarray, coefficients_f: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A g = b`` for logical kernels and ``FG+GF=0``."""
    ff = _xor_terms(coefficients_f)
    zz = translation_orbit(seed)
    xx = (zz @ data_fold_matrix().T) % 2
    z_left, z_right = zz[:, :BLOCK_SIZE], zz[:, BLOCK_SIZE:]
    x_left, x_right = xx[:, :BLOCK_SIZE], xx[:, BLOCK_SIZE:]
    target = np.concatenate(
        [
            (ff @ z_left.T % 2).reshape(-1),
            (ff.T @ x_right.T % 2).reshape(-1),
            np.zeros(BLOCK_SIZE * BLOCK_SIZE, dtype=np.uint8),
        ]
    )
    columns = []
    for term in represented_terms():
        columns.append(
            np.concatenate(
                [
                    (term @ z_right.T % 2).reshape(-1),
                    (term.T @ x_left.T % 2).reshape(-1),
                    ((ff @ term + term @ ff) % 2).reshape(-1),
                ]
            )
        )
    return np.asarray(columns, dtype=np.uint8).T, target


def logical_compatibility_constraints(seed: np.ndarray) -> np.ndarray:
    """Return exact linear constraints on ``F`` for existence of logical ``G``.

    CSS is deliberately omitted here.  Eliminating the ``G`` variables from
    the two logical-kernel systems removes random ``F`` choices that could
    never have any completion.
    """
    zz = translation_orbit(seed)
    xx = (zz @ data_fold_matrix().T) % 2
    z_left, z_right = zz[:, :BLOCK_SIZE], zz[:, BLOCK_SIZE:]
    x_left, x_right = xx[:, :BLOCK_SIZE], xx[:, BLOCK_SIZE:]
    columns_g = []
    columns_f = []
    for term in represented_terms():
        columns_g.append(
            np.concatenate(
                [
                    (term @ z_right.T % 2).reshape(-1),
                    (term.T @ x_left.T % 2).reshape(-1),
                ]
            )
        )
        columns_f.append(
            np.concatenate(
                [
                    (term @ z_left.T % 2).reshape(-1),
                    (term.T @ x_right.T % 2).reshape(-1),
                ]
            )
        )
    joint = np.hstack(
        [
            np.asarray(columns_g, dtype=np.uint8).T,
            np.asarray(columns_f, dtype=np.uint8).T,
        ]
    )
    reduced, _pivots = gf2_rref(joint)
    constraints = reduced[~np.any(reduced[:, : len(represented_terms())], axis=1)]
    return constraints[:, len(represented_terms()) :]


def _packed_columns(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sum(int(matrix[row, column]) << row for row in range(matrix.shape[0]))
        for column in range(matrix.shape[1])
    )


def compatible_sparse_f_supports(
    seed: np.ndarray,
    *,
    weights: Sequence[int] = (2, 3, 4, 5, 6),
    restarts: int = 5,
    maximum_supports: int = 100,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Find sparse ``F`` supports inside the logical compatibility code."""
    started = time.perf_counter()
    constraints = logical_compatibility_constraints(seed)
    augmentation = np.zeros(
        (LATTICE_ORDER, constraints.shape[1]), dtype=np.uint8
    )
    for bottom in range(LATTICE_ORDER):
        for top in range(6):
            augmentation[bottom, top * LATTICE_ORDER + bottom] = 1
    augmentation_rank_gain = int(
        gf2_rank(np.vstack([constraints, augmentation])) - gf2_rank(constraints)
    )
    if augmentation_rank_gain == 0:
        return {
            "constraint_rows": int(constraints.shape[0]),
            "constraint_rank": gf2_rank(constraints),
            "augmentation_rank_gain": 0,
            "weights": list(weights),
            "restarts": restarts,
            "maximum_supports": maximum_supports,
            "supports": [],
            "counters": {"augmentation_obstructed": 1},
            "seconds": round(time.perf_counter() - started, 6),
        }
    syndromes = _packed_columns(constraints)
    rng = random.Random(random_seed)
    supports = []
    seen: set[tuple[int, ...]] = set()
    counters: Counter[str] = Counter()
    stop = False
    for weight in weights:
        if not 2 <= weight <= 6:
            raise ValueError("guided F weights must lie between two and six")
        left_size = weight // 2
        right_size = weight - left_size
        for _restart in range(restarts):
            order = list(range(len(syndromes)))
            rng.shuffle(order)
            left = order[: len(order) // 2]
            right = order[len(order) // 2 :]
            left_by_key: dict[int, list[tuple[int, ...]]] = {}
            for combination in __import__("itertools").combinations(left, left_size):
                key = 0
                for index in combination:
                    key ^= syndromes[index]
                left_by_key.setdefault(key, []).append(combination)
            counters[f"weight_{weight}_left_combinations"] += sum(
                len(value) for value in left_by_key.values()
            )
            for combination in __import__("itertools").combinations(right, right_size):
                key = 0
                for index in combination:
                    key ^= syndromes[index]
                for match in left_by_key.get(key, ()):
                    support = tuple(sorted((*match, *combination)))
                    if support in seen:
                        counters["duplicates"] += 1
                        continue
                    seen.add(support)
                    coefficients = np.zeros(len(syndromes), dtype=np.uint8)
                    coefficients[list(support)] = 1
                    if not _has_nonzero_top_augmentation(coefficients):
                        counters["zero_top_augmentation"] += 1
                        continue
                    supports.append(support)
                    counters[f"weight_{weight}_retained"] += 1
                    if len(supports) >= maximum_supports:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    return {
        "constraint_rows": int(constraints.shape[0]),
        "constraint_rank": gf2_rank(constraints),
        "augmentation_rank_gain": augmentation_rank_gain,
        "weights": list(weights),
        "restarts": restarts,
        "maximum_supports": maximum_supports,
        "supports": [list(support) for support in supports],
        "counters": dict(sorted(counters.items())),
        "seconds": round(time.perf_counter() - started, 6),
    }


def solve_minimum_augmented_f(
    seed: np.ndarray,
    *,
    maximum_terms: int,
    time_limit: float,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Minimize ``F`` weight inside the logical-compatibility code.

    The auxiliary binary variables are the 32 coefficients of the top-group
    augmentation.  Requiring at least one of them to be nonzero excludes the
    invisible natural-representation kernel exactly, without choosing the
    nonzero coefficient in advance.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, eye, hstack, vstack

    started = time.perf_counter()
    compatibility = logical_compatibility_constraints(seed)
    compatibility, _pivots = gf2_rref(compatibility)
    compatibility = compatibility[np.any(compatibility, axis=1)]
    num_rows, num_coefficients = compatibility.shape
    augmentation = np.zeros((LATTICE_ORDER, num_coefficients), dtype=np.uint8)
    for bottom in range(LATTICE_ORDER):
        for top in range(6):
            augmentation[bottom, top * LATTICE_ORDER + bottom] = 1
    augmentation_rank_gain = int(
        gf2_rank(np.vstack([compatibility, augmentation]))
        - gf2_rank(compatibility)
    )
    if augmentation_rank_gain == 0:
        return {
            "status": "augmentation_obstructed",
            "solver_status": None,
            "constraint_rows": int(num_rows),
            "augmentation_rank_gain": 0,
            "seconds": round(time.perf_counter() - started, 6),
        }

    # Variables are (F bits, augmentation bits, compatibility quotients,
    # augmentation quotients).  All are integral; the first two blocks are
    # binary.  The quotient blocks implement equality modulo two.
    num_augmentation = LATTICE_ORDER
    offset_y = num_coefficients
    offset_compatibility_q = offset_y + num_augmentation
    offset_augmentation_q = offset_compatibility_q + num_rows
    num_variables = offset_augmentation_q + num_augmentation
    compatibility_parity = hstack(
        [
            csr_matrix(compatibility.astype(float)),
            csr_matrix((num_rows, num_augmentation)),
            -2 * eye(num_rows, format="csr"),
            csr_matrix((num_rows, num_augmentation)),
        ],
        format="csr",
    )
    augmentation_parity = hstack(
        [
            csr_matrix(augmentation.astype(float)),
            -eye(num_augmentation, format="csr"),
            csr_matrix((num_augmentation, num_rows)),
            -2 * eye(num_augmentation, format="csr"),
        ],
        format="csr",
    )
    nonzero_augmentation = np.zeros((1, num_variables), dtype=float)
    nonzero_augmentation[0, offset_y:offset_compatibility_q] = 1
    maximum_weight = np.zeros((1, num_variables), dtype=float)
    maximum_weight[0, :num_coefficients] = 1
    constraint_matrix = vstack(
        [
            compatibility_parity,
            augmentation_parity,
            csr_matrix(nonzero_augmentation),
            csr_matrix(maximum_weight),
        ],
        format="csr",
    )
    lower_constraints = np.concatenate(
        [
            np.zeros(num_rows + num_augmentation),
            [1.0, 1.0],
        ]
    )
    upper_constraints = np.concatenate(
        [
            np.zeros(num_rows + num_augmentation),
            [float(num_augmentation), float(maximum_terms)],
        ]
    )
    lower = np.zeros(num_variables)
    upper = np.ones(num_variables)
    upper[offset_compatibility_q:offset_augmentation_q] = max(
        1, num_coefficients // 2
    )
    upper[offset_augmentation_q:] = 3
    rng = random.Random(random_seed)
    objective = np.zeros(num_variables)
    objective[:num_coefficients] = 1 + np.asarray(
        [rng.random() for _ in range(num_coefficients)]
    ) / (1000 * num_coefficients)
    result = milp(
        c=objective,
        integrality=np.ones(num_variables),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            constraint_matrix, lower_constraints, upper_constraints
        ),
        options={"time_limit": float(time_limit), "presolve": True},
    )
    if result.x is None:
        return {
            "status": "infeasible" if result.status == 2 else "no_solution",
            "solver_status": int(result.status),
            "constraint_rows": int(num_rows),
            "augmentation_rank_gain": augmentation_rank_gain,
            "seconds": round(time.perf_counter() - started, 6),
        }
    coefficients_f = np.rint(result.x[:num_coefficients]).astype(np.uint8)
    return {
        "status": "solved",
        "solver_status": int(result.status),
        "constraint_rows": int(num_rows),
        "augmentation_rank_gain": augmentation_rank_gain,
        "coefficient_weight": int(np.count_nonzero(coefficients_f)),
        "coefficients": coefficients_f,
        "terms": [
            _serialize_term(int(index)) for index in np.flatnonzero(coefficients_f)
        ],
        "seconds": round(time.perf_counter() - started, 6),
    }


def independent_affine_rows(
    matrix: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reduce a binary affine system using packed Python integers."""
    matrix = np.asarray(matrix, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    if matrix.shape[0] != len(target):
        raise ValueError("affine matrix and target dimensions disagree")
    byte_count = (matrix.shape[1] + 7) // 8
    basis: dict[int, tuple[int, int]] = {}
    for packed, rhs in zip(np.packbits(matrix, axis=1), target, strict=True):
        value = int.from_bytes(packed.tobytes())
        bit = int(rhs)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (value, bit)
                break
            reducer, reducer_bit = basis[pivot]
            value ^= reducer
            bit ^= reducer_bit
        if not value and bit:
            return None
    rows = []
    rhs_values = []
    for value, bit in basis.values():
        packed = np.frombuffer(value.to_bytes(byte_count), dtype=np.uint8)
        rows.append(np.unpackbits(packed)[: matrix.shape[1]])
        rhs_values.append(bit)
    return np.asarray(rows, dtype=np.uint8), np.asarray(rhs_values, dtype=np.uint8)


def solve_sparse_g(
    seed: np.ndarray,
    coefficients_f: np.ndarray,
    *,
    minimum_terms: int,
    maximum_terms: int,
    time_limit: float,
    random_seed: int,
) -> dict[str, Any]:
    """Minimize ``G`` support in the exact affine CSS/kernel system."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, eye, hstack, vstack

    started = time.perf_counter()
    matrix, target = affine_g_system(seed, coefficients_f)
    reduced = independent_affine_rows(matrix, target)
    if reduced is None:
        return {
            "status": "inconsistent",
            "solver_status": None,
            "seconds": round(time.perf_counter() - started, 6),
        }
    rows, rhs = reduced
    num_rows, num_coefficients = rows.shape
    parity = hstack(
        [csr_matrix(rows.astype(float)), -2 * eye(num_rows, format="csr")],
        format="csr",
    )
    weight = np.zeros((1, num_coefficients + num_rows), dtype=float)
    weight[0, :num_coefficients] = 1
    kernel = np.zeros(
        (LATTICE_ORDER, num_coefficients + num_rows), dtype=float
    )
    for bottom in range(LATTICE_ORDER):
        for top in range(6):
            kernel[bottom, top * LATTICE_ORDER + bottom] = 1
    constraints = vstack(
        [parity, csr_matrix(weight), csr_matrix(kernel)], format="csr"
    )
    lower_constraints = np.concatenate(
        [rhs.astype(float), [float(minimum_terms)], np.full(LATTICE_ORDER, -np.inf)]
    )
    upper_constraints = np.concatenate(
        [rhs.astype(float), [float(maximum_terms)], np.full(LATTICE_ORDER, 5.0)]
    )
    lower = np.zeros(num_coefficients + num_rows)
    upper = np.ones(num_coefficients + num_rows)
    upper[num_coefficients:] = np.maximum(
        1, np.count_nonzero(rows, axis=1) // 2
    )
    rng = random.Random(random_seed)
    objective = np.zeros(num_coefficients + num_rows)
    objective[:num_coefficients] = 10_000 + np.asarray(
        [rng.randrange(1_000) for _ in range(num_coefficients)]
    )
    result = milp(
        c=objective,
        integrality=np.ones(num_coefficients + num_rows, dtype=int),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            constraints, lower_constraints, upper_constraints
        ),
        options={"time_limit": time_limit, "presolve": True},
    )
    if result.x is None:
        return {
            "status": "infeasible" if result.status == 2 else "no_incumbent",
            "solver_status": int(result.status),
            "solver_message": result.message,
            "constraint_rank": num_rows,
            "seconds": round(time.perf_counter() - started, 6),
        }
    coefficients_g = (result.x[:num_coefficients] > 0.5).astype(np.uint8)
    if np.any((matrix @ coefficients_g) % 2 != target):
        raise AssertionError("affine G solution failed exact validation")
    return {
        "status": "solved",
        "solver_status": int(result.status),
        "constraint_rank": num_rows,
        "coefficient_weight": int(np.count_nonzero(coefficients_g)),
        "coefficients": coefficients_g,
        "terms": [
            _serialize_term(int(index)) for index in np.flatnonzero(coefficients_g)
        ],
        "seconds": round(time.perf_counter() - started, 6),
    }


def analyze_pair(
    seed: np.ndarray,
    coefficients_f: np.ndarray,
    coefficients_g: np.ndarray,
    *,
    screen_distance_four: bool = True,
) -> dict[str, Any]:
    matrix_x, matrix_z = checks_from_coefficients(coefficients_f, coefficients_g)
    css = bool(not np.any((matrix_x @ matrix_z.T) % 2))
    zz = translation_orbit(seed)
    xx = (zz @ data_fold_matrix().T) % 2
    rank_x = gf2_rank(matrix_x)
    rank_z = gf2_rank(matrix_z)
    grid_rank = gf2_rank(np.vstack([matrix_z, zz])) - rank_z if css else None
    row_weights_x = np.count_nonzero(matrix_x, axis=1)
    row_weights_z = np.count_nonzero(matrix_z, axis=1)
    column_weights_x = np.count_nonzero(matrix_x, axis=0)
    column_weights_z = np.count_nonzero(matrix_z, axis=0)
    rowspace_fold = rowspace_zx_dual(matrix_x, matrix_z) if css else False
    structural = bool(
        css
        and not np.any((matrix_x @ zz.T) % 2)
        and not np.any((matrix_z @ xx.T) % 2)
        and grid_rank == TARGET_LOGICALS
        and rowspace_fold
        and max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        <= MAXIMUM_CHECK_WEIGHT
        and np.all(column_weights_x > 0)
        and np.all(column_weights_z > 0)
        and _has_nonzero_top_augmentation(coefficients_f)
        and _has_nonzero_top_augmentation(coefficients_g)
    )
    low_z = low_x = None
    connected = tanner_connected(matrix_x, matrix_z) if structural else False
    if structural and connected and screen_distance_four:
        code = codes.CSSCode(matrix_x, matrix_z)
        low_z = find_logical_up_to_weight_four(code, pauli="Z")
        low_x = find_logical_up_to_weight_four(code, pauli="X")
    return {
        "terms_f": [
            _serialize_term(int(index)) for index in np.flatnonzero(coefficients_f)
        ],
        "terms_g": [
            _serialize_term(int(index)) for index in np.flatnonzero(coefficients_g)
        ],
        "coefficient_weight_f": int(np.count_nonzero(coefficients_f)),
        "coefficient_weight_g": int(np.count_nonzero(coefficients_g)),
        "css_orthogonal": css,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "k_if_css": NUM_QUBITS - rank_x - rank_z if css else None,
        "target_grid_rank": grid_rank,
        "rowspace_zx_dual": rowspace_fold,
        "maximum_check_weight": int(
            max(row_weights_x.max(initial=0), row_weights_z.max(initial=0))
        ),
        "row_weights_x": sorted(set(row_weights_x.astype(int).tolist())),
        "row_weights_z": sorted(set(row_weights_z.astype(int).tolist())),
        "column_degrees_x": sorted(set(column_weights_x.astype(int).tolist())),
        "column_degrees_z": sorted(set(column_weights_z.astype(int).tolist())),
        "zero_columns_x": int(np.count_nonzero(column_weights_x == 0)),
        "zero_columns_z": int(np.count_nonzero(column_weights_z == 0)),
        "top_augmentation_f_nonzero": _has_nonzero_top_augmentation(coefficients_f),
        "top_augmentation_g_nonzero": _has_nonzero_top_augmentation(coefficients_g),
        "even_syndrome_parity": bool(
            np.all(column_weights_x % 2 == 0)
            and np.all(column_weights_z % 2 == 0)
        ),
        "structural_hit": structural,
        "tanner_connected": connected,
        "logical_z_up_to_weight_four": low_z,
        "logical_x_up_to_weight_four": low_x,
        "passes_weight_four_screen": bool(
            structural and connected and low_z is None and low_x is None
        ),
    }


def random_f_coefficients(
    rng: random.Random, *, weight: int
) -> np.ndarray:
    while True:
        selected = rng.sample(range(len(represented_terms())), weight)
        coefficients = np.zeros(len(represented_terms()), dtype=np.uint8)
        coefficients[selected] = 1
        if _has_nonzero_top_augmentation(coefficients):
            return coefficients


def run_rowspace_pilot(
    seed: np.ndarray,
    *,
    f_trials: int,
    f_weights: Sequence[int] = (3, 4, 5, 6),
    maximum_total_terms: int = 12,
    seconds_per_solve: float = 1,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Run a bounded sparse-``F``, affine-solved-``G`` pilot."""
    if f_trials < 1 or not f_weights:
        raise ValueError("F trial count and weights must be nonempty")
    rng = random.Random(random_seed)
    counters: Counter[str] = Counter()
    records = []
    structural_hits = []
    started = time.perf_counter()
    for trial in range(f_trials):
        weight_f = int(f_weights[trial % len(f_weights)])
        coefficients_f = random_f_coefficients(rng, weight=weight_f)
        solve = solve_sparse_g(
            seed,
            coefficients_f,
            minimum_terms=1,
            maximum_terms=maximum_total_terms - weight_f,
            time_limit=seconds_per_solve,
            random_seed=random_seed + trial,
        )
        counters[solve["status"]] += 1
        if solve["status"] != "solved":
            continue
        coefficients_g = solve.pop("coefficients")
        analysis = analyze_pair(seed, coefficients_f, coefficients_g)
        record = {"trial": trial, "solve": solve, "analysis": analysis}
        records.append(record)
        if analysis["rowspace_zx_dual"]:
            counters["rowspace_zx_hits"] += 1
        if analysis["structural_hit"]:
            counters["structural_hits"] += 1
            structural_hits.append(record)
        if analysis["passes_weight_four_screen"]:
            counters["weight_four_survivors"] += 1
    return {
        "method": "random-sparse-F-affine-solved-G",
        "f_trials": f_trials,
        "f_weights": list(f_weights),
        "maximum_total_terms": maximum_total_terms,
        "seconds_per_solve": seconds_per_solve,
        "counters": dict(sorted(counters.items())),
        "records": records,
        "structural_hits": structural_hits,
        "seconds": round(time.perf_counter() - started, 6),
    }


def run_guided_rowspace_pilot(
    seed: np.ndarray,
    *,
    f_supports: int = 100,
    compatibility_restarts: int = 5,
    maximum_total_terms: int = 12,
    seconds_per_solve: float = 1,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Enumerate compatible sparse ``F`` and solve the full affine CSS ``G``."""
    started = time.perf_counter()
    compatibility = compatible_sparse_f_supports(
        seed,
        restarts=compatibility_restarts,
        maximum_supports=f_supports,
        random_seed=random_seed,
    )
    counters: Counter[str] = Counter()
    records = []
    structural_hits = []
    for trial, support in enumerate(compatibility["supports"]):
        coefficients_f = np.zeros(len(represented_terms()), dtype=np.uint8)
        coefficients_f[support] = 1
        solve = solve_sparse_g(
            seed,
            coefficients_f,
            minimum_terms=1,
            maximum_terms=maximum_total_terms - len(support),
            time_limit=seconds_per_solve,
            random_seed=random_seed + trial,
        )
        counters[solve["status"]] += 1
        if solve["status"] != "solved":
            continue
        coefficients_g = solve.pop("coefficients")
        analysis = analyze_pair(seed, coefficients_f, coefficients_g)
        record = {"support_f": support, "solve": solve, "analysis": analysis}
        records.append(record)
        if analysis["rowspace_zx_dual"]:
            counters["rowspace_zx_hits"] += 1
        if analysis["structural_hit"]:
            counters["structural_hits"] += 1
            structural_hits.append(record)
        if analysis["passes_weight_four_screen"]:
            counters["weight_four_survivors"] += 1
    return {
        "method": "logical-compatible-sparse-F-affine-solved-G",
        "compatibility": compatibility,
        "maximum_total_terms": maximum_total_terms,
        "seconds_per_solve": seconds_per_solve,
        "counters": dict(sorted(counters.items())),
        "records": records,
        "structural_hits": structural_hits,
        "seconds": round(time.perf_counter() - started, 6),
    }


def run_minimum_f_rowspace_pilot(
    seed: np.ndarray,
    *,
    f_trials: int = 3,
    maximum_total_terms: int = 12,
    seconds_per_f_solve: float = 2,
    seconds_per_g_solve: float = 2,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Minimize compatible augmented ``F``, then complete it with sparse ``G``."""
    started = time.perf_counter()
    counters: Counter[str] = Counter()
    minimum_f_records = []
    records = []
    structural_hits = []
    seen_f: set[tuple[int, ...]] = set()
    for trial in range(f_trials):
        solve_f = solve_minimum_augmented_f(
            seed,
            maximum_terms=maximum_total_terms - 1,
            time_limit=seconds_per_f_solve,
            random_seed=random_seed + trial,
        )
        counters[f"f_{solve_f['status']}"] += 1
        if solve_f["status"] != "solved":
            minimum_f_records.append({"trial": trial, "solve_f": solve_f})
            if solve_f["status"] == "augmentation_obstructed":
                break
            continue
        coefficients_f = solve_f.pop("coefficients")
        support_f = tuple(np.flatnonzero(coefficients_f).astype(int).tolist())
        if support_f in seen_f:
            counters["duplicate_f"] += 1
        seen_f.add(support_f)
        minimum_f_records.append({"trial": trial, "solve_f": solve_f})
        remaining_terms = maximum_total_terms - len(support_f)
        solve_g = solve_sparse_g(
            seed,
            coefficients_f,
            minimum_terms=1,
            maximum_terms=remaining_terms,
            time_limit=seconds_per_g_solve,
            random_seed=random_seed + trial,
        )
        counters[f"g_{solve_g['status']}"] += 1
        if solve_g["status"] != "solved":
            records.append(
                {
                    "trial": trial,
                    "support_f": support_f,
                    "solve_f": solve_f,
                    "solve_g": solve_g,
                }
            )
            continue
        coefficients_g = solve_g.pop("coefficients")
        analysis = analyze_pair(seed, coefficients_f, coefficients_g)
        record = {
            "trial": trial,
            "support_f": support_f,
            "solve_f": solve_f,
            "solve_g": solve_g,
            "analysis": analysis,
        }
        records.append(record)
        if analysis["rowspace_zx_dual"]:
            counters["rowspace_zx_hits"] += 1
        if analysis["structural_hit"]:
            counters["structural_hits"] += 1
            structural_hits.append(record)
        if analysis["passes_weight_four_screen"]:
            counters["weight_four_survivors"] += 1
    return {
        "method": "minimum-compatible-augmented-F-affine-solved-G",
        "f_trials": f_trials,
        "maximum_total_terms": maximum_total_terms,
        "seconds_per_f_solve": seconds_per_f_solve,
        "seconds_per_g_solve": seconds_per_g_solve,
        "counters": dict(sorted(counters.items())),
        "minimum_f_records": minimum_f_records,
        "records": records,
        "structural_hits": structural_hits,
        "seconds": round(time.perf_counter() - started, 6),
    }
