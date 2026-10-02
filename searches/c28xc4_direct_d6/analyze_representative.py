#!/usr/bin/env python3
"""Extract and certify a compact connected ``[[112,16,6]]`` representative."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import Counter
from typing import Any, Sequence

import networkx as nx
import numpy as np

from algebra import (
    CHECK_RANK,
    GROUP_ORDER,
    QUOTIENT_Y_ORDER,
    build_stabilizer_basis,
    displacement_subgroup_size,
    fibre_logicals,
    gf2_rref,
    quotient_annihilator_basis,
    validate_stabilizer,
)
from search import (
    build_sparse_catalog,
    enumerate_ideal_candidates,
    find_logical_through_weight_five,
    find_weight_six_logical,
)

CANDIDATE_ID = "ideal-6c21985b5722a169"
SEED_SUPPORTS = (
    (
        (10, 0), (10, 1), (14, 0), (14, 1),
        (17, 0), (17, 1), (17, 2), (17, 3),
        (18, 2), (18, 3),
        (21, 0), (21, 1), (21, 2), (21, 3),
        (26, 2), (26, 3),
    ),
    (
        (2, 0), (3, 0), (4, 0), (6, 0),
        (9, 0), (10, 0), (11, 0), (13, 0),
        (16, 0), (17, 0), (18, 0), (20, 0),
        (23, 0), (24, 0), (25, 0), (27, 0),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--basis-trials", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def vector(support: Sequence[tuple[int, int]]) -> np.ndarray[Any, Any]:
    result = np.zeros((28, 4), dtype=np.uint8)
    for power_x, power_y in support:
        result[power_x, power_y] ^= 1
    return result.reshape(-1)


def unique_orbit(seed: np.ndarray[Any, Any]) -> list[tuple[int, int, np.ndarray[Any, Any]]]:
    grid = seed.reshape(28, 4)
    records: dict[bytes, tuple[int, int, np.ndarray[Any, Any]]] = {}
    for shift_x in range(28):
        for shift_y in range(4):
            row = np.roll(grid, (shift_x, shift_y), axis=(0, 1)).reshape(-1)
            key = np.packbits(row, bitorder="little").tobytes()
            records.setdefault(key, (shift_x, shift_y, row))
    return list(records.values())


def row_catalog(check: np.ndarray[Any, Any], ceiling: int = 16) -> np.ndarray[Any, Any]:
    records: dict[bytes, np.ndarray[Any, Any]] = {}
    for seed in check:
        if int(seed.sum()) > ceiling:
            continue
        for _shift_x, _shift_y, row in unique_orbit(seed):
            key = np.packbits(row, bitorder="little").tobytes()
            records.setdefault(key, row)
    return np.asarray(list(records.values()), dtype=np.uint8)


def independent_selection(rows: np.ndarray[Any, Any], order: Sequence[int]) -> list[int]:
    row_integers = [
        int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        for row in rows
    ]
    pivots: dict[int, int] = {}
    selected = []
    for index in order:
        value = row_integers[index]
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                selected.append(int(index))
                break
        if len(selected) == CHECK_RANK:
            return selected
    return selected


def balanced_minimum_cnot_basis(
    rows: np.ndarray[Any, Any], *, trials: int, random_seed: int
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    generator = random.Random(random_seed)
    weights = rows.sum(axis=1).astype(int)
    groups = {
        weight: np.flatnonzero(weights == weight).astype(int).tolist()
        for weight in sorted(set(map(int, weights)))
    }
    best = None
    for trial in range(trials):
        order = []
        for weight in sorted(groups):
            group = groups[weight].copy()
            generator.shuffle(group)
            order.extend(group)
        selected = independent_selection(rows, order)
        if len(selected) != CHECK_RANK:
            raise RuntimeError("weight-16 catalog failed to span the check space")
        basis = rows[selected]
        degrees = basis.sum(axis=0).astype(int)
        row_weights = basis.sum(axis=1).astype(int)
        score = (
            int(row_weights.sum()),
            int(degrees.max()),
            int(np.sum(degrees**2)),
            int(degrees.max() - degrees.min()),
        )
        if best is None or score < best[0]:
            best = (score, selected, basis.copy(), degrees.copy(), trial)
    assert best is not None
    score, selected, basis, degrees, trial = best
    return basis, {
        "random_trials": trials,
        "random_seed": random_seed,
        "best_trial": trial,
        "score": list(score),
        "selected_catalog_rows": selected,
        "check_weight_histogram": dict(sorted(Counter(map(int, basis.sum(axis=1))).items())),
        "qubit_degree_histogram": dict(sorted(Counter(map(int, degrees)).items())),
        "minimum_qubit_degree": int(degrees.min()),
        "maximum_qubit_degree": int(degrees.max()),
        "average_qubit_degree": float(degrees.mean()),
        "cnot_count_one_css_type": int(basis.sum()),
    }


def edge_schedule(check: np.ndarray[Any, Any]) -> dict[str, Any]:
    num_checks, num_qubits = check.shape
    delta = int(max(check.sum(axis=1).max(), check.sum(axis=0).max()))
    size = max(num_checks, num_qubits)
    left = list(range(size))
    right = list(range(size, 2 * size))
    original = {(int(row), int(column)) for row, column in np.argwhere(check)}

    graph = nx.Graph()
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)
    graph.add_edges_from((row, size + column) for row, column in original)

    left_deficit = [delta - graph.degree(row) for row in left]
    right_deficit = [delta - graph.degree(size + column) for column in range(size)]
    flow = nx.DiGraph()
    source, sink = "source", "sink"
    for row, deficit in enumerate(left_deficit):
        flow.add_edge(source, ("l", row), capacity=deficit)
    for column, deficit in enumerate(right_deficit):
        flow.add_edge(("r", column), sink, capacity=deficit)
    for row in range(size):
        for column in range(size):
            if not graph.has_edge(row, size + column):
                flow.add_edge(("l", row), ("r", column), capacity=1)
    value, flows = nx.maximum_flow(flow, source, sink)
    if value != sum(left_deficit):
        raise RuntimeError("could not regularize Tanner graph")
    for row in range(size):
        for node, amount in flows[("l", row)].items():
            if amount and isinstance(node, tuple) and node[0] == "r":
                graph.add_edge(row, size + int(node[1]))
    if any(graph.degree(node) != delta for node in graph):
        raise RuntimeError("Tanner regularization has the wrong degree")

    layers = []
    for _layer in range(delta):
        matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left)
        matched = [(row, matching[row]) for row in left]
        if len(matched) != size:
            raise RuntimeError("regularized graph lacks a perfect matching")
        layers.append(
            [
                {"check": row, "qubit": target - size}
                for row, target in matched
                if row < num_checks and target - size < num_qubits and (row, target - size) in original
            ]
        )
        graph.remove_edges_from(matched)
    scheduled = {
        (edge["check"], edge["qubit"])
        for layer in layers
        for edge in layer
    }
    if scheduled != original:
        raise RuntimeError("edge schedule does not equal the Tanner graph")
    return {
        "scope": "one CSS check type with one ancilla per listed check",
        "checks": num_checks,
        "qubits": num_qubits,
        "layers": delta,
        "optimal": True,
        "optimality_reason": f"maximum Tanner degree is {delta}",
        "cnot_count": len(original),
        "cnots_per_layer": [len(layer) for layer in layers],
        "schedule": layers,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    catalog = build_sparse_catalog()
    candidates, _counts = enumerate_ideal_candidates(catalog)
    candidate = next(item for item in candidates if item.candidate_id == CANDIDATE_ID)
    annihilator = quotient_annihilator_basis(list(candidate.ideal_basis))
    stabilizer = build_stabilizer_basis(candidate.ideal_basis, annihilator)
    properties = validate_stabilizer(stabilizer)
    logicals = fibre_logicals()

    seeds = [vector(support) for support in SEED_SUPPORTS]
    orbits = [unique_orbit(seed) for seed in seeds]
    full_rows = np.asarray([item[2] for orbit in orbits for item in orbit], dtype=np.uint8)
    if len(gf2_rref(full_rows)[0]) != CHECK_RANK:
        raise RuntimeError("two-seed presentation does not generate rank 48")
    if displacement_subgroup_size(seeds) != GROUP_ORDER:
        raise RuntimeError("two-seed presentation is disconnected")
    if gf2_rref(full_rows)[0].tobytes() != stabilizer.tobytes():
        raise RuntimeError("two-seed presentation generates the wrong row space")

    light_catalog = row_catalog(stabilizer)
    independent, basis_summary = balanced_minimum_cnot_basis(
        light_catalog, trials=args.basis_trials, random_seed=args.seed
    )
    if gf2_rref(independent)[0].tobytes() != stabilizer.tobytes():
        raise RuntimeError("independent presentation generates the wrong row space")

    low_weight, _support, distance_seconds = find_logical_through_weight_five(
        stabilizer, logicals[0]
    )
    if low_weight is not None:
        raise RuntimeError(f"unexpected logical of weight {low_weight}")
    six_support, six_seconds = find_weight_six_logical(stabilizer, logicals[0])
    distance_seconds += six_seconds
    if six_support is None:
        raise RuntimeError("failed to reproduce the weight-six logical")

    independent_schedule = edge_schedule(independent)
    full_schedule = edge_schedule(full_rows)
    write_json(args.output / "independent-schedule-one-css-type.json", independent_schedule)
    write_json(args.output / "translation-symmetric-schedule-one-css-type.json", full_schedule)

    seed_summary = {
        "ring": "F2[u^+-1,v^+-1]/(u^28-1,v^4-1)",
        "candidate_id": CANDIDATE_ID,
        "gf8_ideal": "I=(U^3,V) in GF(8)[U,V]/(U^4,V^4), U=1+X, V=1+Y",
        "gf8_annihilator": "Ann(I)=(U V^3)",
        "binary_check_seeds": [
            {
                "polynomial": " + ".join(
                    "1" if xx == 0 and yy == 0 else
                    f"u^{xx}" if yy == 0 else
                    f"v^{yy}" if xx == 0 else f"u^{xx}v^{yy}"
                    for xx, yy in support
                ),
                "support": [list(term) for term in support],
                "weight": len(support),
                "orbit_size": len(orbit),
                "orbit_rank": len(gf2_rref(np.asarray([item[2] for item in orbit], dtype=np.uint8))[0]),
            }
            for support, orbit in zip(SEED_SUPPORTS, orbits)
        ],
        "combined_orbit_rank": len(gf2_rref(full_rows)[0]),
        "combined_displacement_subgroup_size": displacement_subgroup_size(seeds),
    }
    write_json(args.output / "polynomials.json", seed_summary)
    write_json(args.output / "balanced-independent-basis.json", basis_summary)

    full_degrees = full_rows.sum(axis=0).astype(int)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "n": GROUP_ORDER,
        "k": GROUP_ORDER - 2 * CHECK_RANK,
        "distance": 6,
        "distance_method": "exact exclusion through five plus weight-six meet-in-the-middle witness",
        "distance_support": six_support,
        "distance_seconds": distance_seconds,
        **properties,
        "exact_zx_self_dual": True,
        "doubly_even_check_space": bool(np.all(stabilizer.sum(axis=1) % 4 == 0)),
        "logical_count": len(logicals),
        "logical_weights": sorted(set(map(int, logicals.sum(axis=1)))),
        "logical_supports_pairwise_disjoint": bool(np.max(logicals.sum(axis=0)) == 1),
        "logical_translation_group": "C4 x C4",
        "minimum_complete_presentation_maximum_check_weight": 16,
        "weight_12_exclusion": "exact mixed-sector MILP: light span rank 36; no outside stabilizer through weight 12",
        "translation_symmetric_seed_types": len(seeds),
        "translation_symmetric_checks": len(full_rows),
        "translation_symmetric_check_weight_histogram": dict(sorted(Counter(map(int, full_rows.sum(axis=1))).items())),
        "translation_symmetric_qubit_degree_histogram": dict(sorted(Counter(map(int, full_degrees)).items())),
        "translation_symmetric_syndrome_parity_even": bool(np.all(full_degrees % 2 == 0)),
        "translation_symmetric_ancillas_per_css_type": len(full_rows),
        "translation_symmetric_cnot_depth_per_css_type": full_schedule["layers"],
        "independent_ancillas_per_css_type": len(independent),
        "independent_check_weight_histogram": basis_summary["check_weight_histogram"],
        "independent_qubit_degree_histogram": basis_summary["qubit_degree_histogram"],
        "independent_cnot_count_per_css_type": basis_summary["cnot_count_one_css_type"],
        "independent_cnot_depth_per_css_type": independent_schedule["layers"],
    }
    write_json(args.output / "summary.json", summary)
    np.savez_compressed(
        args.output / "code-matrices.npz",
        check_x=independent,
        check_z=independent,
        stabilizer_rref=stabilizer,
        logical_x=logicals,
        logical_z=logicals,
        translation_symmetric_checks=full_rows,
        check_seeds=np.asarray(seeds, dtype=np.uint8),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
