"""Exact low-distance MILP screens for the batched logical-grid search."""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli


def _column_bitmasks(matrix: np.ndarray) -> list[int]:
    matrix = np.asarray(matrix, dtype=np.uint8)
    return [
        sum(int(matrix[row, col]) << row for row in range(matrix.shape[0]))
        for col in range(matrix.shape[1])
    ]


def find_logical_through_weight_four(
    code: codes.CSSCode, *, pauli: Literal["X", "Z"] = "Z"
) -> dict[str, Any]:
    """Find the minimum logical of weight at most four by syndrome hashing."""
    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be X or Z")
    started = time.perf_counter()
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=np.uint8
    )
    conjugate = np.asarray(
        code.get_logical_ops(Pauli.Z if pauli == "X" else Pauli.X),
        dtype=np.uint8,
    )
    syndromes = _column_bitmasks(check)
    logicals = _column_bitmasks(conjugate)
    num_qubits = code.num_qubits

    def result(support: list[int] | None) -> dict[str, Any]:
        return {
            "pauli": pauli,
            "solver": "SYNDROME_HASH",
            "solver_status": "witness" if support is not None else "not_found",
            "maximum_weight": 4,
            "exact_weight": None,
            "time_limit": None,
            "seconds": round(time.perf_counter() - started, 6),
            "support": support,
            "weight": None if support is None else len(support),
            "infeasible": support is None,
        }

    for qubit in range(num_qubits):
        if syndromes[qubit] == 0 and logicals[qubit] != 0:
            return result([qubit])

    pairs_by_syndrome: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for left in range(num_qubits):
        for right in range(left + 1, num_qubits):
            syndrome = syndromes[left] ^ syndromes[right]
            logical = logicals[left] ^ logicals[right]
            if syndrome == 0 and logical != 0:
                return result([left, right])
            pairs_by_syndrome[syndrome].append((left, right, logical))

    for pair_syndrome, pairs in pairs_by_syndrome.items():
        for left, right, pair_logical in pairs:
            for qubit in range(num_qubits):
                if qubit in (left, right):
                    continue
                if syndromes[qubit] == pair_syndrome and (
                    pair_logical ^ logicals[qubit]
                ) != 0:
                    return result([left, right, qubit])

    for pairs in pairs_by_syndrome.values():
        for first_index, (aa, bb, first_logical) in enumerate(pairs):
            for cc, dd, second_logical in pairs[first_index + 1 :]:
                if len({aa, bb, cc, dd}) != 4:
                    continue
                if (first_logical ^ second_logical) != 0:
                    return result([aa, bb, cc, dd])
    return result(None)


def low_weight_logical_spectrum(
    code: codes.CSSCode,
    *,
    pauli: Literal["X", "Z"] = "Z",
    maximum_weight: int = 4,
) -> dict[str, Any]:
    """Count every nontrivial logical support through a small weight.

    The complete spectrum supplies a useful optimization signal: two codes
    can have the same minimum distance while one has far fewer offending
    supports.  Syndrome and logical signatures are represented as Python
    integers, making the exhaustive scan through weight four inexpensive for
    the 48-qubit compact candidates.
    """
    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be X or Z")
    if not 1 <= maximum_weight <= 4:
        raise ValueError("maximum weight must lie between one and four")
    started = time.perf_counter()
    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=np.uint8
    )
    conjugate = np.asarray(
        code.get_logical_ops(Pauli.Z if pauli == "X" else Pauli.X),
        dtype=np.uint8,
    )
    syndromes = _column_bitmasks(check)
    logicals = _column_bitmasks(conjugate)
    counts: dict[int, int] = {}
    witnesses: dict[int, list[int] | None] = {}
    minimum: int | None = None
    for weight in range(1, maximum_weight + 1):
        count = 0
        witness = None
        for support in itertools.combinations(range(code.num_qubits), weight):
            syndrome = 0
            logical = 0
            for qubit in support:
                syndrome ^= syndromes[qubit]
                logical ^= logicals[qubit]
            if syndrome == 0 and logical != 0:
                count += 1
                if witness is None:
                    witness = list(support)
        counts[weight] = count
        witnesses[weight] = witness
        if minimum is None and count:
            minimum = weight
    return {
        "pauli": pauli,
        "maximum_weight": maximum_weight,
        "minimum_logical_weight_found": minimum,
        "counts": {str(weight): count for weight, count in counts.items()},
        "witnesses": {
            str(weight): witness for weight, witness in witnesses.items()
        },
        "clear_through_maximum_weight": minimum is None,
        "seconds": round(time.perf_counter() - started, 6),
    }


def find_logical_milp(
    code: codes.CSSCode,
    *,
    pauli: Literal["X", "Z"] = "Z",
    maximum_weight: int | None = None,
    exact_weight: int | None = None,
    solver: str = "HIGHS",
    time_limit: float | None = None,
) -> dict[str, Any]:
    """Find a nontrivial logical, or certify that none obeys the weight bound.

    Unlike a zero-syndrome search, this formulation explicitly requires odd
    pairing with at least one conjugate logical basis vector.  Low-weight
    stabilizers therefore do not cause false distance rejections.
    """
    import cvxpy as cp

    if pauli not in {"X", "Z"}:
        raise ValueError("pauli must be X or Z")
    if exact_weight is not None and maximum_weight is not None:
        raise ValueError("pass exact_weight or maximum_weight, not both")
    if exact_weight is not None and exact_weight < 1:
        raise ValueError("exact weight must be positive")
    if maximum_weight is not None and maximum_weight < 1:
        raise ValueError("maximum weight must be positive")

    check = np.asarray(
        code.matrix_z if pauli == "X" else code.matrix_x, dtype=int
    )
    conjugate = np.asarray(
        code.get_logical_ops(Pauli.Z if pauli == "X" else Pauli.X), dtype=int
    )
    error = cp.Variable(code.num_qubits, boolean=True)
    syndrome_slack = cp.Variable(check.shape[0], integer=True)
    logical_parities = cp.Variable(conjugate.shape[0], boolean=True)
    logical_slack = cp.Variable(conjugate.shape[0], integer=True)
    constraints = [
        check @ error == 2 * syndrome_slack,
        syndrome_slack >= 0,
        syndrome_slack <= np.sum(check, axis=1) // 2,
        conjugate @ error == logical_parities + 2 * logical_slack,
        logical_slack >= 0,
        logical_slack <= np.sum(conjugate, axis=1) // 2,
        cp.sum(logical_parities) >= 1,
    ]
    if exact_weight is not None:
        constraints.append(cp.sum(error) == exact_weight)
    elif maximum_weight is not None:
        constraints.append(cp.sum(error) <= maximum_weight)

    objective = cp.Minimize(cp.sum(error))
    problem = cp.Problem(objective, constraints)
    started = time.perf_counter()
    solver_options = (
        {"highs_options": {"time_limit": float(time_limit)}}
        if solver == "HIGHS" and time_limit is not None
        else {}
    )
    value = problem.solve(solver=solver, **solver_options)
    elapsed = time.perf_counter() - started
    base = {
        "pauli": pauli,
        "solver": solver,
        "solver_status": problem.status,
        "maximum_weight": maximum_weight,
        "exact_weight": exact_weight,
        "time_limit": time_limit,
        "seconds": round(elapsed, 6),
    }
    if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
        return {**base, "support": None, "weight": None, "infeasible": True}
    if problem.status != cp.OPTIMAL or error.value is None or not np.isfinite(value):
        return {**base, "support": None, "weight": None, "infeasible": False}
    solution = np.rint(error.value).astype(np.uint8)
    return {
        **base,
        "support": np.flatnonzero(solution).astype(int).tolist(),
        "weight": int(np.count_nonzero(solution)),
        "infeasible": False,
    }


def certify_distance_at_least(
    code: codes.CSSCode,
    minimum_distance: int = 6,
    *,
    solver: str = "HIGHS",
    time_limit: float | None = None,
    equal_xz_by_permutation: bool = False,
) -> dict[str, Any]:
    """Certify both CSS distances are at least ``minimum_distance``."""
    if minimum_distance < 2:
        raise ValueError("minimum distance must be at least two")
    equal_by_identity = np.array_equal(code.matrix_x, code.matrix_z)
    directions = (
        ("Z",)
        if equal_by_identity or equal_xz_by_permutation
        else ("X", "Z")
    )
    results = {}
    for pauli in directions:
        quick = (
            find_logical_through_weight_four(
                code, pauli=pauli  # type: ignore[arg-type]
            )
            if minimum_distance >= 5
            else None
        )
        if quick is not None and quick["support"] is not None:
            results[pauli] = quick
        else:
            results[pauli] = find_logical_milp(
                code,
                pauli=pauli,  # type: ignore[arg-type]
                maximum_weight=minimum_distance - 1,
                solver=solver,
                time_limit=time_limit,
            )
    return {
        "minimum_distance": minimum_distance,
        "directions": results,
        "equal_xz_distance_by_identity_fold": equal_by_identity,
        "equal_xz_distance_by_permutation_fold": bool(
            equal_xz_by_permutation and not equal_by_identity
        ),
        "certified": all(result["infeasible"] for result in results.values()),
    }
