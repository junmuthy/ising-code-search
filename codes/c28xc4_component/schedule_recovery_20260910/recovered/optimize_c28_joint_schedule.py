#!/usr/bin/env python3
"""Build and improve a depth-16 simultaneous schedule for the C28xC4 code."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path("/home/judah_unmuth/gala-code-search/c28xc4_component")
sys.path.insert(0, str(ROOT))
from component_code import translated_check_catalog, gf2_rref  # noqa: E402

DEPTH = 16


def selected_check(basis_path: Path):
    selection = json.loads(basis_path.read_text())["selected_indices"]
    rows, metadata, _orbits = translated_check_catalog()
    check = rows[selection]
    if len(gf2_rref(check)[0]) != 48:
        raise AssertionError("selected rows are not independent")
    return check, selection, metadata


def initial_coloring(check):
    num_checks, num_data = check.shape
    real_left = 2 * num_checks
    padded = max(real_left, num_data)
    combined_degrees = 2 * check.sum(axis=0).astype(int)
    deficits = (DEPTH - combined_degrees).tolist()
    dummy_checks = padded - real_left
    dummy = nx.algorithms.bipartite.havel_hakimi_graph(
        [DEPTH] * dummy_checks, deficits, create_using=nx.Graph
    )
    graph = nx.Graph()
    left = list(range(padded))
    right = list(range(padded, 2 * padded))
    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)
    edge_records = []
    edge_lookup = {}
    for kind in range(2):
        for row, data in zip(*np.nonzero(check)):
            left_node = kind * num_checks + int(row)
            right_node = padded + int(data)
            edge_index = len(edge_records)
            edge_records.append((kind, int(row), int(data), left_node, right_node))
            edge_lookup[(left_node, right_node)] = edge_index
            graph.add_edge(left_node, right_node)
    for dummy_left, dummy_right in dummy.edges():
        if dummy_left >= dummy_checks:
            dummy_left, dummy_right = dummy_right, dummy_left
        data = dummy_right - dummy_checks
        graph.add_edge(real_left + dummy_left, padded + data)
    if any(graph.degree(node) != DEPTH for node in graph):
        raise AssertionError("regularization failure")
    colors = np.full(len(edge_records), -1, dtype=np.int16)
    for color in range(DEPTH):
        matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left)
        matched = [(node, matching[node]) for node in left]
        if len(matched) != padded:
            raise AssertionError("perfect matching failure")
        for left_node, right_node in matched:
            edge_index = edge_lookup.get((left_node, right_node))
            if edge_index is not None:
                colors[edge_index] = color
        graph.remove_edges_from(matched)
    if np.any(colors < 0):
        raise AssertionError("uncolored real edge")
    return edge_records, colors


def time_arrays(check, edge_records, colors):
    tx = np.full(check.shape, -1, dtype=np.int16)
    tz = np.full(check.shape, -1, dtype=np.int16)
    for edge, color in zip(edge_records, colors, strict=True):
        kind, row, data, _left, _right = edge
        (tx if kind == 0 else tz)[row, data] = color
    return tx, tz


def parity_matrix(check, tx, tz):
    active = check.astype(bool)
    before = tx[:, None, :] < tz[None, :, :]
    overlap = active[:, None, :] & active[None, :, :]
    return np.sum(before & overlap, axis=2, dtype=np.int16) % 2


def objective(check, edge_records, colors):
    tx, tz = time_arrays(check, edge_records, colors)
    failures = parity_matrix(check, tx, tz)
    return int(failures.sum()), tx, tz, failures


def incidence(edge_records, colors):
    by_node_color = {}
    for edge_index, edge in enumerate(edge_records):
        _kind, _row, _data, left, right = edge
        color = int(colors[edge_index])
        by_node_color[(left, color)] = edge_index
        by_node_color[(right, color)] = edge_index
    return by_node_color


def kempe_component(edge_records, colors, start, first, second, by_node_color):
    todo = [start]
    seen_edges = set()
    seen_nodes = set()
    while todo:
        edge_index = todo.pop()
        if edge_index in seen_edges:
            continue
        seen_edges.add(edge_index)
        edge = edge_records[edge_index]
        for node in edge[3:5]:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            for color in (first, second):
                neighbor = by_node_color.get((node, color))
                if neighbor is not None and neighbor not in seen_edges:
                    todo.append(neighbor)
    return list(seen_edges)


def verify_coloring(check, edge_records, colors):
    used = {}
    for edge_index, (kind, row, data, left, right) in enumerate(edge_records):
        color = int(colors[edge_index])
        for node in (left, right):
            key = (node, color)
            if key in used:
                raise AssertionError(f"collision {key}")
            used[key] = edge_index
    score, tx, tz, failures = objective(check, edge_records, colors)
    return score, tx, tz, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path("/tmp/c28_basis_search.json"))
    parser.add_argument("--moves", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=281607)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check, selection, metadata = selected_check(args.basis)
    edge_records, colors = initial_coloring(check)
    rng = random.Random(args.seed)
    current_score, _tx, _tz, _failures = verify_coloring(check, edge_records, colors)
    best_score = current_score
    best_colors = colors.copy()
    started = time.perf_counter()
    print(json.dumps({"event": "start", "parity_failures": current_score}), flush=True)
    for move in range(1, args.moves + 1):
        by_node_color = incidence(edge_records, colors)
        start = rng.randrange(len(edge_records))
        first = int(colors[start])
        second = rng.randrange(DEPTH - 1)
        if second >= first:
            second += 1
        component = kempe_component(edge_records, colors, start, first, second, by_node_color)
        old_score = current_score
        for edge_index in component:
            colors[edge_index] = second if colors[edge_index] == first else first
        new_score, _tx, _tz, _failures = objective(check, edge_records, colors)
        temperature = max(0.02, 2.0 * (1.0 - move / args.moves))
        accept = new_score <= old_score or rng.random() < math.exp((old_score - new_score) / temperature)
        if accept:
            current_score = new_score
        else:
            for edge_index in component:
                colors[edge_index] = second if colors[edge_index] == first else first
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            print(json.dumps({
                "event": "best",
                "move": move,
                "parity_failures": best_score,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }), flush=True)
            if best_score == 0:
                break
    score, tx, tz, failures = verify_coloring(check, edge_records, best_colors)
    layers = [[] for _ in range(DEPTH)]
    for edge, color in zip(edge_records, best_colors, strict=True):
        kind, row, data, _left, _right = edge
        layers[int(color)].append({"type": "X" if kind == 0 else "Z", "check": row, "data": data})
    result = {
        "code": "[[112,16,7]]",
        "depth": DEPTH,
        "cnot_count": len(edge_records),
        "layer_sizes": [len(layer) for layer in layers],
        "parity_failures": score,
        "clean_cross_ancilla_backaction": score == 0,
        "basis_selected_indices": selection,
        "basis_selected_checks": [
            {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
            for i in selection
        ],
        "basis_degree_histogram": {
            str(value): int(np.count_nonzero(check.sum(axis=0) == value))
            for value in sorted(set(map(int, check.sum(axis=0))))
        },
        "failed_cross_pairs": np.argwhere(failures).astype(int).tolist(),
        "colors": best_colors.astype(int).tolist(),
        "layers": layers,
        "moves": move,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("colors", "layers", "basis_selected_checks", "basis_selected_indices", "failed_cross_pairs")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
