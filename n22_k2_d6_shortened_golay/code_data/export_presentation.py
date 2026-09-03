#!/usr/bin/env python3
"""Export exact matrices, supports, fold data, and validation metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .code import (
    A_SUPPORT,
    B_SUPPORT,
    F_SUPPORT,
    LOGICAL_X_SUPPORTS,
    LOGICAL_Z_SUPPORTS,
    P_SUPPORT,
    Q_SUPPORT,
    build_checks,
    fold_check_action,
    fold_permutation,
    validate_code,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()
    checks_x, checks_z = build_checks()
    logical_x = np.zeros((2, 22), dtype=np.uint8)
    logical_z = np.zeros((2, 22), dtype=np.uint8)
    for row, support in enumerate(LOGICAL_X_SUPPORTS):
        logical_x[row, list(support)] = 1
    for row, support in enumerate(LOGICAL_Z_SUPPORTS):
        logical_z[row, list(support)] = 1
    record = {
        "schema_version": 1,
        "code": "[[22,2,6]]",
        "polynomial_supports": {
            "f": list(F_SUPPORT),
            "p": list(P_SUPPORT),
            "q": list(Q_SUPPORT),
            "a_equals_pf": list(A_SUPPORT),
            "b_equals_qf": list(B_SUPPORT),
        },
        "checks_x": [np.flatnonzero(row).astype(int).tolist() for row in checks_x],
        "checks_z": [np.flatnonzero(row).astype(int).tolist() for row in checks_z],
        "redundant_relations": [list(range(11))],
        "logical_supports_x": [list(row) for row in LOGICAL_X_SUPPORTS],
        "logical_supports_z": [list(row) for row in LOGICAL_Z_SUPPORTS],
        "zx_fold": list(fold_permutation()),
        "zx_fold_check_action": list(fold_check_action()),
        "validation": validate_code(exhaustive=True),
    }
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.npz,
        matrix_x=checks_x,
        matrix_z=checks_z,
        logical_x=logical_x,
        logical_z=logical_z,
        zx_fold=np.asarray(fold_permutation(), dtype=np.int64),
        zx_fold_check_action=np.asarray(fold_check_action(), dtype=np.int64),
    )


if __name__ == "__main__":
    main()

