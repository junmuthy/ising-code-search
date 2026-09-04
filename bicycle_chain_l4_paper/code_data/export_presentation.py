#!/usr/bin/env python3
"""Export the exact paper matrices, logicals, and symmetries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .code import (
    A_SUPPORT,
    B_SUPPORT,
    ELL,
    LOGICAL_X_SUPPORTS,
    LOGICAL_Z_SUPPORTS,
    M,
    build_checks,
    fold_check_action,
    fold_permutation,
    redundant_relations,
    support_matrix,
    translation_x_permutation,
    validate_code,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    if args.json.exists() or args.npz.exists():
        raise SystemExit("refusing to overwrite an existing frozen artifact")
    checks_x, checks_z = build_checks()
    logical_x = support_matrix(LOGICAL_X_SUPPORTS)
    logical_z = support_matrix(LOGICAL_Z_SUPPORTS)
    record = {
        "schema_version": 1,
        "code": "[[56,8,6]]",
        "source": "Ismail et al., arXiv:2606.25011v1, Appendix B, ell=4, m=7",
        "coordinate_convention": "q(half,i,j)=half*ell*m+i*m+j; half 0=L, 1=R",
        "group_orders": {"ell": ELL, "m": M},
        "polynomial_supports": {
            "a": [list(row) for row in A_SUPPORT],
            "b_equals_a_dagger": [list(row) for row in B_SUPPORT],
        },
        "checks_x": [np.flatnonzero(row).astype(int).tolist() for row in checks_x],
        "checks_z": [np.flatnonzero(row).astype(int).tolist() for row in checks_z],
        "redundant_relations": [list(row) for row in redundant_relations(checks_x)],
        "logical_supports_x": [list(row) for row in LOGICAL_X_SUPPORTS],
        "logical_supports_z": [list(row) for row in LOGICAL_Z_SUPPORTS],
        "zx_fold": list(fold_permutation()),
        "zx_fold_check_action": list(fold_check_action()),
        "translation_x": list(translation_x_permutation()),
        "validation": validate_code(exhaustive=args.exhaustive),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.npz,
        matrix_x=checks_x,
        matrix_z=checks_z,
        logical_x=logical_x,
        logical_z=logical_z,
        zx_fold=np.asarray(fold_permutation(), dtype=np.int64),
        zx_fold_check_action=np.asarray(fold_check_action(), dtype=np.int64),
        translation_x=np.asarray(translation_x_permutation(), dtype=np.int64),
    )


if __name__ == "__main__":
    main()
