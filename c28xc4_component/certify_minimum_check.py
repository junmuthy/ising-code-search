#!/usr/bin/env python3
"""Certify that the C28 x C4 component has no stabilizer of weight <= 12."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import cvxpy as cp
import numpy as np

from component_code import check_space_basis, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    check = check_space_basis().astype(int)

    coefficients = cp.Variable(len(check), boolean=True)
    word = cp.Variable(check.shape[1], boolean=True)
    slack = cp.Variable(check.shape[1], integer=True)
    constraints = [
        check.T @ coefficients == word + 2 * slack,
        slack >= 0,
        slack <= check.sum(axis=0) // 2,
        word[0] == 1,
        cp.sum(word) <= 12,
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    started = time.perf_counter()
    problem.solve(solver="HIGHS")
    elapsed = time.perf_counter() - started
    properties = validate(check)
    result = {
        "solver": "HIGHS",
        "status": problem.status,
        "tested_maximum_weight": 12,
        "coordinate_zero_fixed": True,
        "coordinate_fix_justification": "C28 x C4 translations are transitive",
        "doubly_even_check_space": properties["doubly_even_check_space"],
        "known_check_upper_bound": 16,
        "certified_minimum_check_weight": (
            16
            if problem.status == cp.INFEASIBLE
            and properties["doubly_even_check_space"]
            else None
        ),
        "elapsed_seconds": elapsed,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
