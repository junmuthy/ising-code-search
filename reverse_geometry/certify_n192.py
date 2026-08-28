#!/usr/bin/env python3
"""Apply cheap exact distance and structural tests to saved n=192 hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qldpc import codes

from gala_search.s3_ising import find_logical_up_to_weight_four
from reverse_geometry.n192 import (
    NUM_QUBITS,
    matrices_from_terms,
    tanner_connected,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results/reverse-geometry") / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    source = json.loads(args.source.read_text())
    output_dir.mkdir(parents=True)
    hits = [
        hit
        for run in source["runs"]
        for hit in run["search"]["structural_pilot_hits"]
    ]
    records = []
    for index, hit in enumerate(hits):
        matrix_x, matrix_z = matrices_from_terms(hit["terms"])
        code = codes.CSSCode(matrix_x, matrix_z)
        low_z = find_logical_up_to_weight_four(code, pauli="Z")
        low_x = find_logical_up_to_weight_four(code, pauli="X")
        records.append(
            {
                "candidate_index": index,
                "terms": hit["terms"],
                "n": code.num_qubits,
                "k": code.dimension,
                "rank_x": code.code_x.rank,
                "rank_z": code.code_z.rank,
                "css_orthogonal": bool(not np.any((matrix_x @ matrix_z.T) % 2)),
                "tanner_connected": tanner_connected(matrix_x, matrix_z),
                "logical_z_up_to_weight_four": low_z,
                "logical_x_up_to_weight_four": low_x,
                "passes_weight_four_screen": low_z is None and low_x is None,
                "distance_at_most": min(
                    [
                        record["minimum_weight_upper_bound"]
                        for record in (low_z, low_x)
                        if record is not None
                    ],
                    default=None,
                ),
                "source_record": hit,
            }
        )
    result = {
        "source": str(args.source),
        "candidates_tested": len(records),
        "passes_weight_four_screen": sum(
            record["passes_weight_four_screen"] for record in records
        ),
        "records": records,
    }
    output = output_dir / "weight-four-certification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "candidates_tested": result["candidates_tested"],
        "passes_weight_four_screen": result["passes_weight_four_screen"],
        "parameters": [[record["n"], record["k"]] for record in records],
        "distance_upper_bounds": [record["distance_at_most"] for record in records],
    }, indent=2))


if __name__ == "__main__":
    main()
