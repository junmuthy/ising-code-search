#!/usr/bin/env python3
"""Build an optimal 16-layer Tanner-edge schedule for one CSS check type."""

from __future__ import annotations

import argparse
import json
import pathlib

import networkx as nx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with np.load(args.matrices) as archive:
        check = np.asarray(archive["check_x"], dtype=np.uint8)
    num_checks, num_qubits = check.shape
    maximum_degree = int(max(check.sum(axis=1).max(), check.sum(axis=0).max()))
    if maximum_degree != 16:
        raise RuntimeError(f"expected maximum Tanner degree 16, found {maximum_degree}")

    # Pad to a 16-regular 112-by-112 bipartite graph.  Repeated perfect
    # matchings then provide an exact Delta-edge-coloring (Konig's theorem).
    padded_size = max(num_checks, num_qubits)
    dummy_checks = padded_size - num_checks
    qubit_deficits = (maximum_degree - check.sum(axis=0)).astype(int).tolist()
    dummy = nx.algorithms.bipartite.havel_hakimi_graph(
        [maximum_degree] * dummy_checks,
        qubit_deficits,
        create_using=nx.Graph,
    )

    graph = nx.Graph()
    left_nodes = list(range(padded_size))
    right_nodes = list(range(padded_size, 2 * padded_size))
    graph.add_nodes_from(left_nodes, bipartite=0)
    graph.add_nodes_from(right_nodes, bipartite=1)
    for check_index, qubit_index in zip(*np.nonzero(check)):
        graph.add_edge(int(check_index), padded_size + int(qubit_index))
    # The dummy graph labels left nodes 0..dummy_checks-1 and right nodes
    # dummy_checks..dummy_checks+num_qubits-1.
    for dummy_left, dummy_right in dummy.edges():
        if dummy_left >= dummy_checks:
            dummy_left, dummy_right = dummy_right, dummy_left
        qubit_index = dummy_right - dummy_checks
        graph.add_edge(num_checks + dummy_left, padded_size + qubit_index)

    if any(graph.degree(node) != maximum_degree for node in graph):
        raise RuntimeError("regularization failed")

    layers = []
    for layer_index in range(maximum_degree):
        matching = nx.algorithms.bipartite.maximum_matching(
            graph, top_nodes=left_nodes
        )
        matched = [(left, matching[left]) for left in left_nodes]
        if len(matched) != padded_size:
            raise RuntimeError("failed to find a perfect matching")
        real_edges = [
            {"check": left, "qubit": right - padded_size}
            for left, right in matched
            if left < num_checks and right - padded_size < num_qubits
        ]
        layers.append(real_edges)
        graph.remove_edges_from(matched)

    if graph.number_of_edges():
        raise RuntimeError("edge decomposition left unused edges")
    scheduled = {
        (edge["check"], edge["qubit"])
        for layer in layers
        for edge in layer
    }
    expected = set(map(tuple, np.argwhere(check)))
    if scheduled != expected:
        raise RuntimeError("schedule does not equal the Tanner edge set")

    result = {
        "scope": "one CSS check type with one ancilla per independent check",
        "layers": len(layers),
        "optimal": True,
        "optimality_reason": "maximum check degree is 16",
        "checks": num_checks,
        "qubits": num_qubits,
        "cnot_count": len(expected),
        "cnots_per_layer": [len(layer) for layer in layers],
        "schedule": layers,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "schedule"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
