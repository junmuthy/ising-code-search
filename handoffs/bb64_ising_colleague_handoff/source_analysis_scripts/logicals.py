"""Logical ``C4 x C2`` modules and batchwise support optimization."""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

from .model import (
    GRID_ORDER,
    GRID_X_ORDER,
    GRID_Y_ORDER,
    Candidate,
    gf2_rank,
    lift_block_permutation,
    translation_permutations,
    zx_fold_permutation,
)


def _permute_vector(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vector)
    output[permutation] = vector
    return output


def translation_orbit(candidate: Candidate, seed: np.ndarray) -> np.ndarray:
    """Translate one physical support over the logical grid in x-major order."""
    seed = np.asarray(seed, dtype=np.uint8)
    if seed.shape != (candidate.num_qubits,):
        raise ValueError("logical seed has the wrong length")
    px, py = translation_permutations(candidate.top_representation)
    full_px = lift_block_permutation(px, candidate.num_blocks)
    full_py = lift_block_permutation(py, candidate.num_blocks)
    rows = []
    translated_x = seed.copy()
    for _xx in range(GRID_X_ORDER):
        translated_y = translated_x.copy()
        for _yy in range(GRID_Y_ORDER):
            rows.append(translated_y)
            translated_y = _permute_vector(translated_y, full_py)
        translated_x = _permute_vector(translated_x, full_px)
    return np.asarray(rows, dtype=np.uint8)


def grid_index(xx: int, yy: int) -> int:
    return (xx % GRID_X_ORDER) * GRID_Y_ORDER + yy % GRID_Y_ORDER


def _generated_subgroup(generators: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    subgroup = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        xx, yy = frontier.pop()
        for dx, dy in generators:
            point = ((xx + dx) % GRID_X_ORDER, (yy + dy) % GRID_Y_ORDER)
            if point not in subgroup:
                subgroup.add(point)
                frontier.append(point)
    return subgroup


def _coset_partition(generators: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    subgroup = _generated_subgroup(generators)
    remaining = {
        (xx, yy)
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    }
    cosets = []
    while remaining:
        anchor = min(remaining)
        coset = {
            ((anchor[0] + xx) % GRID_X_ORDER, (anchor[1] + yy) % GRID_Y_ORDER)
            for xx, yy in subgroup
        }
        remaining -= coset
        cosets.append(tuple(sorted(grid_index(*point) for point in coset)))
    return tuple(sorted(cosets))


def batch_schemes() -> dict[str, tuple[tuple[int, ...], ...]]:
    """All subgroup-coset partitions into two or four translation-covariant batches."""
    return {
        "two_C4_rows": _coset_partition(((1, 0),)),
        "two_C4_diagonals": _coset_partition(((1, 1),)),
        "two_C2xC2_columns": _coset_partition(((2, 0), (0, 1))),
        "four_C2_x_squared": _coset_partition(((2, 0),)),
        "four_C2_y": _coset_partition(((0, 1),)),
        "four_C2_x_squared_y": _coset_partition(((2, 1),)),
    }


def batch_disjointness(logicals: np.ndarray) -> dict[str, Any]:
    logicals = np.asarray(logicals, dtype=np.uint8)
    if logicals.shape[0] != GRID_ORDER:
        raise ValueError("expected eight logical representatives")
    output: dict[str, Any] = {}
    for name, batches in batch_schemes().items():
        multiplicities = [
            int(np.sum(logicals[list(batch)], axis=0).max(initial=0))
            for batch in batches
        ]
        output[name] = {
            "num_batches": len(batches),
            "batch_size": len(batches[0]),
            "batches": [list(batch) for batch in batches],
            "maximum_multiplicity": multiplicities,
            "disjoint": max(multiplicities, default=0) <= 1,
        }
    return output


def analyze_logical_seed(
    code: codes.CSSCode, candidate: Candidate, support: Sequence[int]
) -> dict[str, Any]:
    """Check whether one seed generates a ZX-closed regular logical module."""
    seed = np.zeros(code.num_qubits, dtype=np.uint8)
    seed[list(support)] = 1
    orbit = translation_orbit(candidate, seed)
    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    stabilizer_z = np.asarray(code.matrix_z, dtype=np.uint8)
    fold = zx_fold_permutation(candidate)
    x_orbit = np.zeros_like(orbit)
    x_orbit[:, fold] = orbit
    pairing = (orbit @ x_orbit.T) % 2
    rank_mod_stabilizers = (
        gf2_rank(np.vstack([stabilizer_z, orbit]), code.field) - code.code_z.rank
    )
    unique_supports = {
        tuple(np.flatnonzero(row).astype(int).tolist()) for row in orbit
    }
    batches = batch_disjointness(orbit)
    return {
        "seed_support": list(map(int, support)),
        "seed_weight": len(support),
        "translation_orbit_size": len(unique_supports),
        "translated_supports": [
            np.flatnonzero(row).astype(int).tolist() for row in orbit
        ],
        "z_orbit_in_kernel": not bool(np.any((hx @ orbit.T) % 2)),
        "x_partner_orbit_in_kernel": not bool(np.any((hz @ x_orbit.T) % 2)),
        "orbit_rank_mod_stabilizers": rank_mod_stabilizers,
        "zx_pairing_rank": gf2_rank(pairing, code.field),
        "zx_pairing": pairing.astype(int).tolist(),
        "regular_logical_module": bool(
            len(unique_supports) == GRID_ORDER
            and not np.any((hx @ orbit.T) % 2)
            and not np.any((hz @ x_orbit.T) % 2)
            and rank_mod_stabilizers == GRID_ORDER
            and gf2_rank(pairing, code.field) == GRID_ORDER
        ),
        "raw_batch_schemes": batches,
        "raw_batch_successes": [
            name for name, result in batches.items() if result["disjoint"]
        ],
    }


def random_logical_seeds(
    code: codes.CSSCode, *, count: int, seed: int
) -> Iterable[np.ndarray]:
    """Sample physical representatives from random Z-logical classes."""
    basis = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    rng = random.Random(seed)
    seen: set[bytes] = set()
    attempts = 0
    while len(seen) < count and attempts < 50 * max(1, count):
        attempts += 1
        coefficients = np.asarray(
            [rng.getrandbits(1) for _ in range(len(basis))], dtype=np.uint8
        )
        if not np.any(coefficients):
            continue
        vector = (coefficients @ basis) % 2
        key = np.packbits(vector).tobytes()
        if key in seen:
            continue
        seen.add(key)
        yield vector


def exhaustive_logical_seeds(
    code: codes.CSSCode, *, maximum_dimension: int = 12
) -> Iterable[np.ndarray]:
    """Enumerate every nonzero Z-logical class for a small code."""
    basis = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    if len(basis) > maximum_dimension:
        raise ValueError(
            f"logical dimension {len(basis)} exceeds exhaustive limit "
            f"{maximum_dimension}"
        )
    for mask in range(1, 1 << len(basis)):
        coefficients = np.fromiter(
            ((mask >> index) & 1 for index in range(len(basis))),
            dtype=np.uint8,
            count=len(basis),
        )
        yield (coefficients @ basis) % 2


def logical_orbit_rank(
    code: codes.CSSCode, candidate: Candidate, vector: np.ndarray
) -> int:
    """Rank of one translation orbit in the logical quotient.

    Pairing translated Z representatives with a full X-logical basis gives
    quotient coordinates up to an invertible basis change, so this avoids
    repeated large stabilizer-row-space reductions.
    """
    conjugate = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    return _logical_orbit_rank_with_conjugate(
        code, candidate, vector, conjugate
    )


def _logical_orbit_rank_with_conjugate(
    code: codes.CSSCode,
    candidate: Candidate,
    vector: np.ndarray,
    conjugate: np.ndarray,
) -> int:
    orbit = translation_orbit(candidate, vector)
    coordinates = (orbit @ conjugate.T) % 2
    return gf2_rank(coordinates, code.field)


def logical_action_profile(
    code: codes.CSSCode,
    candidate: Candidate,
    *,
    random_trials: int = 0,
    seed: int = 0,
    exhaustive_dimension: int = 12,
) -> dict[str, Any]:
    """Measure how large a cyclic `C4 x C2` logical orbit can become."""
    exact = code.dimension <= exhaustive_dimension
    vectors = (
        exhaustive_logical_seeds(code, maximum_dimension=exhaustive_dimension)
        if exact
        else random_logical_seeds(code, count=random_trials, seed=seed)
    )
    histogram: Counter[int] = Counter()
    conjugate = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    for vector in vectors:
        histogram[
            _logical_orbit_rank_with_conjugate(
                code, candidate, vector, conjugate
            )
        ] += 1
    return {
        "exact": exact,
        "tested_classes": sum(histogram.values()),
        "maximum_orbit_rank": max(histogram, default=0),
        "full_rank_classes": histogram[GRID_ORDER],
        "rank_histogram": {
            str(rank): count for rank, count in sorted(histogram.items())
        },
    }


def _matrix_power_mod2(matrix: np.ndarray, exponent: int) -> np.ndarray:
    output = np.eye(len(matrix), dtype=np.uint8)
    base = np.asarray(matrix, dtype=np.uint8)
    while exponent:
        if exponent & 1:
            output = (output @ base) % 2
        base = (base @ base) % 2
        exponent //= 2
    return output


def induced_translation_actions(
    code: codes.CSSCode, candidate: Candidate
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact logical-quotient matrices induced by physical x/y shifts."""
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    pairing = (logical_z @ logical_x.T) % 2
    pairing_inverse = np.asarray(
        np.linalg.inv(pairing.view(code.field)), dtype=np.uint8
    )
    px, py = translation_permutations(candidate.top_representation)
    physical = (
        lift_block_permutation(px, candidate.num_blocks),
        lift_block_permutation(py, candidate.num_blocks),
    )
    output = []
    for permutation in physical:
        translated = np.zeros_like(logical_z)
        translated[:, permutation] = logical_z
        output.append(((translated @ logical_x.T) @ pairing_inverse) % 2)
    return output[0], output[1]


def translation_algebra_profile(
    code: codes.CSSCode, candidate: Candidate
) -> dict[str, Any]:
    """Measure the algebra generated by the eight logical grid translations.

    A regular cyclic ``C4 x C2`` logical module requires this algebra to have
    dimension eight.  A smaller dimension is therefore an exact, inexpensive
    rejection before distance certification.
    """
    tx, ty = induced_translation_actions(code, candidate)
    identity = np.eye(code.dimension, dtype=np.uint8)
    powers_x = [_matrix_power_mod2(tx, exponent) for exponent in range(4)]
    actions = [
        (powers_x[xx] @ _matrix_power_mod2(ty, yy)) % 2
        for xx in range(GRID_X_ORDER)
        for yy in range(GRID_Y_ORDER)
    ]
    relation = next(
        (
            f"T_y=T_x^{exponent}"
            for exponent, power in enumerate(powers_x)
            if np.array_equal(ty, power)
        ),
        "independent",
    )
    dimension = gf2_rank(
        np.asarray(actions, dtype=np.uint8).reshape(GRID_ORDER, -1), code.field
    )
    return {
        "dimension": dimension,
        "required_dimension": GRID_ORDER,
        "passes_regular_grid_necessary_gate": dimension == GRID_ORDER,
        "translation_relation": relation,
        "group_relations_verified": bool(
            np.array_equal(_matrix_power_mod2(tx, GRID_X_ORDER), identity)
            and np.array_equal(_matrix_power_mod2(ty, GRID_Y_ORDER), identity)
            and np.array_equal((tx @ ty) % 2, (ty @ tx) % 2)
        ),
    }


def find_regular_logical_modules(
    code: codes.CSSCode,
    candidate: Candidate,
    *,
    supports: Sequence[Sequence[int]] = (),
    random_trials: int = 0,
    seed: int = 0,
    limit: int = 10,
    exhaustive_dimension: int = 12,
) -> list[dict[str, Any]]:
    """Test logical classes for an eight-site regular module.

    All nonzero classes are enumerated when ``k <= exhaustive_dimension``;
    larger codes retain the reproducible random search.
    """
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for support in supports:
        vector = np.zeros(code.num_qubits, dtype=np.uint8)
        vector[list(support)] = 1
        vectors.append(vector)
    if code.dimension <= exhaustive_dimension:
        vectors.extend(
            exhaustive_logical_seeds(
                code, maximum_dimension=exhaustive_dimension
            )
        )
    else:
        vectors.extend(random_logical_seeds(code, count=random_trials, seed=seed))
    seen_orbits: set[tuple[tuple[int, ...], ...]] = set()
    conjugate = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    for vector in vectors:
        if (
            _logical_orbit_rank_with_conjugate(
                code, candidate, vector, conjugate
            )
            != GRID_ORDER
        ):
            continue
        support = np.flatnonzero(vector).astype(int).tolist()
        record = analyze_logical_seed(code, candidate, support)
        canonical = min(
            tuple(sorted(row)) for row in record["translated_supports"]
        )
        orbit_key = tuple(
            sorted(tuple(sorted(row)) for row in record["translated_supports"])
        )
        if orbit_key in seen_orbits:
            continue
        seen_orbits.add(orbit_key)
        record["canonical_seed_support"] = list(canonical)
        if record["regular_logical_module"]:
            records.append(record)
            if len(records) >= limit:
                break
    return records


def dress_logicals_for_batches(
    code: codes.CSSCode,
    base_logicals: np.ndarray,
    scheme_name: str,
    *,
    solver: str = "HIGHS",
    time_limit: float | None = 30,
) -> dict[str, Any]:
    """Add Z stabilizers to make representatives disjoint inside each batch.

    The eight logical classes are fixed.  Each representative is optimized
    independently within its stabilizer coset, while the packing constraints
    are imposed jointly.  Overlap between different injection batches is
    intentionally allowed.
    """
    import cvxpy as cp

    schemes = batch_schemes()
    if scheme_name not in schemes:
        raise ValueError(f"unknown batching scheme: {scheme_name}")
    base = np.asarray(base_logicals, dtype=np.uint8)
    if base.shape != (GRID_ORDER, code.num_qubits):
        raise ValueError("base logical array has the wrong shape")
    stabilizers = np.asarray(
        np.asarray(code.matrix_z, dtype=np.uint8).view(code.field).row_space(),
        dtype=int,
    )
    coefficients = cp.Variable((GRID_ORDER, len(stabilizers)), boolean=True)
    representatives = cp.Variable((GRID_ORDER, code.num_qubits), boolean=True)
    parity_slack = cp.Variable((GRID_ORDER, code.num_qubits), integer=True)
    constraints = [
        base + coefficients @ stabilizers == representatives + 2 * parity_slack,
        parity_slack >= 0,
        parity_slack <= (len(stabilizers) + 1) // 2,
    ]
    for batch in schemes[scheme_name]:
        constraints.append(cp.sum(representatives[list(batch)], axis=0) <= 1)
    problem = cp.Problem(cp.Minimize(cp.sum(representatives)), constraints)
    started = time.perf_counter()
    solver_options = (
        {"highs_options": {"time_limit": float(time_limit)}}
        if solver == "HIGHS" and time_limit is not None
        else {}
    )
    value = problem.solve(solver=solver, **solver_options)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "scheme": scheme_name,
        "batches": [list(batch) for batch in schemes[scheme_name]],
        "solver": solver,
        "solver_status": problem.status,
        "time_limit": time_limit,
        "seconds": round(elapsed, 6),
    }
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return {**result, "feasible": False}
    if problem.status != cp.OPTIMAL or representatives.value is None:
        return {**result, "feasible": None}
    dressed = np.rint(representatives.value).astype(np.uint8)
    batch_check = batch_disjointness(dressed)[scheme_name]
    return {
        **result,
        "feasible": True,
        "objective": int(round(float(value))),
        "weights": np.count_nonzero(dressed, axis=1).astype(int).tolist(),
        "supports": [
            np.flatnonzero(row).astype(int).tolist() for row in dressed
        ],
        "verified_disjoint": batch_check["disjoint"],
        "maximum_multiplicity": batch_check["maximum_multiplicity"],
    }


def orbit_array(record: dict[str, Any], num_qubits: int) -> np.ndarray:
    """Reconstruct the binary orbit stored by :func:`analyze_logical_seed`."""
    output = np.zeros((GRID_ORDER, num_qubits), dtype=np.uint8)
    for row, support in zip(output, record["translated_supports"]):
        row[list(support)] = 1
    return output
