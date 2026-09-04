#!/usr/bin/env python3
"""Materialize Table II of Ismail et al. at ``ell=4, m=7``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bicycle_chain_l4_paper.code_data.code import (
    A_SUPPORT,
    B_SUPPORT,
    CELL_COUNT,
    ELL,
    M,
    NUM_DATA_QUBITS,
    cell_index,
    data_index,
    fold_check_action,
    fold_permutation,
)


A_TERMS = ((1, 4), (1, 2), (0, 3), (0, 0))
B_TERMS = tuple(((-ii) % ELL, (-jj) % M) for ii, jj in A_TERMS)
TABLE_II = (
    (0, A_TERMS[0], 1, B_TERMS[3]),
    (1, B_TERMS[0], 0, A_TERMS[3]),
    (0, A_TERMS[1], 1, B_TERMS[2]),
    (1, B_TERMS[1], 0, A_TERMS[1]),
    (1, B_TERMS[2], 0, A_TERMS[2]),
    (0, A_TERMS[2], 1, B_TERMS[1]),
    (1, B_TERMS[3], 0, A_TERMS[0]),
    (0, A_TERMS[3], 1, B_TERMS[0]),
)


def materialize() -> dict[str, object]:
    if set(A_TERMS) != set(A_SUPPORT) or set(B_TERMS) != set(B_SUPPORT):
        raise AssertionError("Table-II dictionary does not match the code")
    layers = []
    for layer_index, (x_half, x_shift, z_half, z_shift) in enumerate(TABLE_II):
        gates = []
        for ii in range(ELL):
            for jj in range(M):
                check = cell_index(ii, jj)
                gates.append(
                    {
                        "type": "X",
                        "check": check,
                        "data": data_index(x_half, ii + x_shift[0], jj + x_shift[1]),
                    }
                )
                gates.append(
                    {
                        "type": "Z",
                        "check": check,
                        "data": data_index(z_half, ii + z_shift[0], jj + z_shift[1]),
                    }
                )
        if len(gates) != NUM_DATA_QUBITS:
            raise AssertionError("paper layer is not a perfect matching")
        layers.append(
            {
                "layer": layer_index,
                "paper_round": layer_index + 1,
                "x_half": "L" if x_half == 0 else "R",
                "x_displacement": list(x_shift),
                "z_half": "L" if z_half == 0 else "R",
                "z_displacement": list(z_shift),
                "gates": gates,
            }
        )
    return {
        "schema_version": 1,
        "name": "Ismail-et-al Table-II bicycle-chain schedule at ell=4,m=7",
        "source": "arXiv:2606.25011v1 Appendix B Table II",
        "code": "[[56,8,6]]",
        "cnot_depth": 8,
        "cnot_count": 448,
        "dedicated_x_ancillas": CELL_COUNT,
        "dedicated_z_ancillas": CELL_COUNT,
        "dedicated_ancillas": 2 * CELL_COUNT,
        "translation_invariant_layers": True,
        "collision_free": True,
        "idle_free_entangling_layers": True,
        "zx_fold": list(fold_permutation()),
        "zx_fold_check_action": list(fold_check_action()),
        "a_terms_paper_order": [list(row) for row in A_TERMS],
        "b_terms_paper_order": [list(row) for row in B_TERMS],
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(materialize(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
