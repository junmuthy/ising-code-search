#!/usr/bin/env python3
"""Analyze logical translation representations of half-swap distance survivors."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from collections import Counter
from typing import Any

import numpy as np
from qldpc.objects import Pauli

from .halfswap_model import HalfSwapCandidate, build_half_swap_code
from .model import gf2_rank, lift_block_permutation, translation_permutations

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_DIR / "results" / "batched-c4xc2-search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        temporary = pathlib.Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def load_distance_survivors(source: pathlib.Path) -> list[dict[str, Any]]:
    records_path = source / "candidates.jsonl" if source.is_dir() else source
    output = []
    with records_path.open() as records:
        for line in records:
            record = json.loads(line)
            if record.get("distance", {}).get("certified"):
                output.append(record)
    return output


def permute_rows(rows: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(rows)
    output[:, permutation] = rows
    return output


def matrix_power_mod2(matrix: np.ndarray, exponent: int) -> np.ndarray:
    output = np.eye(len(matrix), dtype=np.uint8)
    base = np.asarray(matrix, dtype=np.uint8)
    while exponent:
        if exponent & 1:
            output = (output @ base) % 2
        base = (base @ base) % 2
        exponent //= 2
    return output


def induced_action(code: Any, physical_permutation: np.ndarray) -> np.ndarray:
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    pairing = (logical_z @ logical_x.T) % 2
    pairing_inverse = np.asarray(
        np.linalg.inv(pairing.view(code.field)), dtype=np.uint8
    )
    translated = permute_rows(logical_z, physical_permutation)
    return ((translated @ logical_x.T) @ pairing_inverse) % 2


def action_record(source: pathlib.Path, record: dict[str, Any]) -> dict[str, Any]:
    candidate = HalfSwapCandidate.from_dict(record["candidate"])
    code = build_half_swap_code(candidate)
    px, py = translation_permutations(candidate.top_representation)
    tx = induced_action(code, lift_block_permutation(px, candidate.num_blocks))
    ty = induced_action(code, lift_block_permutation(py, candidate.num_blocks))
    identity = np.eye(code.dimension, dtype=np.uint8)
    powers_x = [matrix_power_mod2(tx, exponent) for exponent in range(4)]
    group_actions = [
        matrix_power_mod2(tx, xx) @ matrix_power_mod2(ty, yy) % 2
        for xx in range(4)
        for yy in range(2)
    ]
    relation = next(
        (
            f"T_y=T_x^{exponent}"
            for exponent, power in enumerate(powers_x)
            if np.array_equal(ty, power)
        ),
        "independent",
    )
    nx = tx ^ identity
    ny = ty ^ identity
    return {
        "source_run": str(source),
        "candidate_id": candidate.candidate_id,
        "n": code.num_qubits,
        "k": code.dimension,
        "maximum_cyclic_orbit_rank": record["logical_action"][
            "maximum_orbit_rank"
        ],
        "logical_action_was_exhaustive": record["logical_action"]["exact"],
        "translation_relation": relation,
        "translation_group_relations_verified": bool(
            np.array_equal(matrix_power_mod2(tx, 4), identity)
            and np.array_equal(matrix_power_mod2(ty, 2), identity)
            and np.array_equal((tx @ ty) % 2, (ty @ tx) % 2)
        ),
        "translation_image_algebra_dimension": gf2_rank(
            np.asarray(group_actions, dtype=np.uint8).reshape(8, -1), code.field
        ),
        "rank_Tx_plus_I": gf2_rank(nx, code.field),
        "rank_Tx_plus_I_squared": gf2_rank((nx @ nx) % 2, code.field),
        "rank_Tx_plus_I_cubed": gf2_rank((nx @ nx @ nx) % 2, code.field),
        "rank_Ty_plus_I": gf2_rank(ny, code.field),
        "rank_product_nilpotents": gf2_rank((nx @ ny) % 2, code.field),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.results_root / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    output_path = run_dir / "candidates.jsonl"
    records = [
        (source, record)
        for source in args.sources
        for record in load_distance_survivors(source)
    ]
    counts: Counter[str] = Counter()
    for source, record in records:
        analysis = action_record(source, record)
        append_jsonl(output_path, analysis)
        counts[f"k={analysis['k']}"] += 1
        counts[
            f"maximum_orbit_rank={analysis['maximum_cyclic_orbit_rank']}"
        ] += 1
        counts[f"relation:{analysis['translation_relation']}"] += 1
        counts[
            "action_algebra_dimension="
            f"{analysis['translation_image_algebra_dimension']}"
        ] += 1
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "sources": [str(source) for source in args.sources],
        "distance_survivors": len(records),
        "all_logical_screens_exhaustive": all(
            json.loads(line)["logical_action_was_exhaustive"]
            for line in output_path.read_text().splitlines()
        ),
        "all_translation_relations_verified": all(
            json.loads(line)["translation_group_relations_verified"]
            for line in output_path.read_text().splitlines()
        ),
        "counts": dict(counts),
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
