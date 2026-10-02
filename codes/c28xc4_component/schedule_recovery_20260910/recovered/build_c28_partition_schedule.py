#!/usr/bin/env python3
"""Construct a clean depth-16 schedule from an exact early/late partition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import z3

ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
sys.path.insert(0, str(ROOT))
from component_code import gf2_rref, translated_check_catalog  # noqa: E402


def parity(values):
    values = list(values)
    if not values:
        return z3.BoolVal(False)
    while len(values) > 1:
        reduced = [
            z3.Xor(values[index], values[index + 1])
            for index in range(0, len(values) - 1, 2)
        ]
        if len(values) % 2:
            reduced.append(values[-1])
        values = reduced
    return values[0]


def select_basis(path):
    record = json.loads(path.read_text())
    selected = [int(i) for i in record["selected_indices"]]
    rows, metadata, _ = translated_check_catalog()
    check = rows[selected]
    assert len(gf2_rref(check)[0]) == 48
    return check, selected, metadata


def find_partition(check, timeout_seconds):
    early = [z3.Bool(f"early_{data}") for data in range(check.shape[1])]
    solver = z3.Solver()
    solver.set(timeout=timeout_seconds * 1000)
    for row in range(check.shape[0]):
        support = np.flatnonzero(check[row]).astype(int)
        solver.add(z3.PbEq([(early[data], 1) for data in support], 8))
    parity_constraints = 0
    for left in range(check.shape[0]):
        for right in range(left + 1, check.shape[0]):
            overlap = np.flatnonzero(check[left] & check[right]).astype(int)
            if len(overlap):
                solver.add(z3.Not(parity(early[data] for data in overlap)))
                parity_constraints += 1
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    if status != z3.sat:
        return str(status), None, parity_constraints, elapsed
    model = solver.model()
    values = np.asarray([z3.is_true(model.eval(value, model_completion=True)) for value in early])
    return str(status), values, parity_constraints, elapsed


def edge_color(check, active_data, color_offset):
    num_checks, num_data = check.shape
    degree = 8
    graph = nx.Graph()
    padded = num_data
    dummy_checks = padded - num_checks
    left = list(range(padded))
    right = list(range(padded, 2 * padded))
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)
    lookup = {}
    for row, data in zip(*np.nonzero(check)):
        row = int(row)
        data = int(data)
        if not active_data[data]:
            continue
        graph.add_edge(row, padded + data)
        lookup[(row, padded + data)] = (row, data)
    real_degrees = np.asarray([graph.degree(padded + data) for data in range(num_data)])
    deficits = (degree - real_degrees).astype(int).tolist()
    dummy = nx.algorithms.bipartite.havel_hakimi_graph(
        [degree] * dummy_checks, deficits, create_using=nx.Graph
    )
    for dummy_left, dummy_right in dummy.edges():
        if dummy_left >= dummy_checks:
            dummy_left, dummy_right = dummy_right, dummy_left
        data = dummy_right - dummy_checks
        graph.add_edge(num_checks + dummy_left, padded + data)
    if any(graph.degree(node) != degree for node in graph):
        raise AssertionError("8-regular padding failed")
    colors = {}
    for local_color in range(degree):
        matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left)
        edges = [(node, matching[node]) for node in left]
        if len(edges) != padded:
            raise AssertionError("perfect matching failed")
        for edge in edges:
            if edge in lookup:
                colors[lookup[edge]] = color_offset + local_color
        graph.remove_edges_from(edges)
    return colors


def verify(check, partition, colors):
    expected = set(map(tuple, np.argwhere(check)))
    assert set(colors) == expected
    for row in range(48):
        row_colors = [colors[(row, int(data))] for data in np.flatnonzero(check[row])]
        assert sorted(row_colors) == list(range(16))
    for data in range(112):
        incident = [colors[(int(row), data)] for row in np.flatnonzero(check[:, data])]
        assert len(incident) == len(set(incident))
        assert all((color < 8) == bool(partition[data]) for color in incident)
    failures = []
    for xrow in range(48):
        for zrow in range(48):
            overlap = np.flatnonzero(check[xrow] & check[zrow]).astype(int)
            inversions = sum(colors[(xrow, data)] < 15 - colors[(zrow, data)] for data in overlap)
            if inversions % 2:
                failures.append((xrow, zrow))
    assert not failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path("/tmp/c28_basis_search.json"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check, selected, metadata = select_basis(args.basis)
    status, partition, parity_constraints, elapsed = find_partition(check, args.timeout_seconds)
    print(json.dumps({
        "event": "partition",
        "status": status,
        "parity_constraints": parity_constraints,
        "elapsed_seconds": round(elapsed, 3),
    }), flush=True)
    if partition is None:
        return
    low = edge_color(check, partition, 0)
    high = edge_color(check, ~partition, 8)
    colors = {**low, **high}
    verify(check, partition, colors)
    layers = [[] for _ in range(16)]
    for (row, data), color in sorted(colors.items()):
        layers[color].append({"type": "X", "check": row, "data": data})
        layers[15 - color].append({"type": "Z", "check": row, "data": data})
    result = {
        "schema_version": 1,
        "name": "depth-optimal partitioned exact-self-dual C28 x C4 schedule",
        "code": "[[112,16,7]]",
        "depth": 16,
        "depth_optimal": True,
        "cnot_count": 1536,
        "layer_sizes": [len(layer) for layer in layers],
        "peak_layer_load_optimal": True,
        "collision_free": True,
        "clean_cross_ancilla_backaction": True,
        "fold_plus_time_reversal": True,
        "x_early_data": np.flatnonzero(partition).astype(int).tolist(),
        "x_late_data": np.flatnonzero(~partition).astype(int).tolist(),
        "partition_constraints": {
            "early_qubits_per_check": 8,
            "even_early_parity_on_each_pairwise_check_overlap": True,
            "nonzero_pairwise_overlap_constraints": parity_constraints,
        },
        "basis_selected_indices": selected,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selected
        ],
        "basis_degree_histogram": {
            str(value): int(np.count_nonzero(check.sum(axis=0) == value))
            for value in sorted(set(map(int, check.sum(axis=0))))
        },
        "layers": layers,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("layers", "basis_selected_checks", "basis_selected_indices", "x_early_data", "x_late_data")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
