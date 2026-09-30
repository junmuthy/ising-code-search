#!/usr/bin/env python3
"""Use SciPy/HiGHS to certify the exact static distance of the BB64 code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    with np.load(DIRECTORY / "bb64_two_batch_basis.npz") as archive:
        check = np.asarray(archive["matrix_x"], dtype=float)
        conjugate_logicals = np.asarray(archive["logical_x"], dtype=float)

    num_qubits = check.shape[1]
    num_checks = check.shape[0]
    num_logicals = conjugate_logicals.shape[0]
    error = slice(0, num_qubits)
    syndrome_slack = slice(num_qubits, num_qubits + num_checks)
    logical_parity = slice(
        num_qubits + num_checks, num_qubits + num_checks + num_logicals
    )
    logical_slack = slice(num_qubits + num_checks + num_logicals, None)
    size = num_qubits + num_checks + 2 * num_logicals

    objective = np.zeros(size)
    objective[error] = 1
    lower = np.zeros(size)
    upper = np.ones(size)
    upper[syndrome_slack] = check.sum(axis=1) // 2
    upper[logical_slack] = conjugate_logicals.sum(axis=1) // 2

    equalities = np.zeros((num_checks + num_logicals, size))
    equalities[:num_checks, error] = check
    equalities[:num_checks, syndrome_slack] = -2 * np.eye(num_checks)
    equalities[num_checks:, error] = conjugate_logicals
    equalities[num_checks:, logical_parity] = -np.eye(num_logicals)
    equalities[num_checks:, logical_slack] = -2 * np.eye(num_logicals)

    nontrivial = np.zeros((1, size))
    nontrivial[0, logical_parity] = 1
    result = milp(
        objective,
        integrality=np.ones(size),
        bounds=Bounds(lower, upper),
        constraints=(
            LinearConstraint(equalities, np.zeros(len(equalities)), np.zeros(len(equalities))),
            LinearConstraint(nontrivial, np.asarray([1.0]), np.asarray([np.inf])),
        ),
        options={"presolve": True},
    )
    if not result.success or result.fun is None or result.x is None:
        raise AssertionError(f"HiGHS did not finish optimally: {result.message}")
    solution = np.rint(result.x[error]).astype(np.uint8)
    weight = int(solution.sum())
    support = np.flatnonzero(solution).astype(int).tolist()
    assert weight == 8
    assert not np.any((check.astype(np.uint8) @ solution) % 2)
    assert np.any((conjugate_logicals.astype(np.uint8) @ solution) % 2)
    print(f"HiGHS optimum: weight {weight}, support {support}")
    print("Because H_X=H_Z, the X and Z distances agree. Certified [[64,8,8]].")


if __name__ == "__main__":
    main()
