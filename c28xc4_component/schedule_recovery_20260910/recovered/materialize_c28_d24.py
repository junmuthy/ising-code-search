#!/usr/bin/env python3
"""Materialize and independently verify the constructive depth-24 schedule."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
from optimize_c28_joint_schedule import initial_coloring, selected_check, verify_coloring


def main():
    basis = Path("/tmp/c28_basis_search.json")
    source = Path("/tmp/c28_span_coloring_241129.json")
    output = Path("/tmp/c28_clean_d24.json")
    check, selection, metadata = selected_check(basis)
    records, _base = initial_coloring(check)
    source_payload = json.loads(source.read_text())
    source_colors = np.asarray(source_payload["colors"], dtype=np.int16)
    one_type = {
        (row, data): int(source_colors[index])
        for index, (kind, row, data, _left, _right) in enumerate(records)
        if kind == 0
    }
    colors = np.asarray([
        one_type[row, data] + (8 if kind == 0 else 0)
        for kind, row, data, _left, _right in records
    ], dtype=np.int16)
    failures, tx, tz, matrix = verify_coloring(check, records, colors)
    if failures:
        raise AssertionError(np.argwhere(matrix).tolist())
    for data in range(112):
        active_x = tx[:, data][tx[:, data] >= 0]
        active_z = tz[:, data][tz[:, data] >= 0]
        if int(active_z.max()) >= int(active_x.min()):
            raise AssertionError((data, active_z.tolist(), active_x.tolist()))
    layers = [[] for _ in range(24)]
    seen = set()
    for edge, color in zip(records, colors, strict=True):
        kind, row, data, _left, _right = edge
        gate = {"type": "X" if kind == 0 else "Z", "check": row, "data": data}
        layers[int(color)].append(gate)
        key = (gate["type"], row, data)
        if key in seen:
            raise AssertionError(key)
        seen.add(key)
    if len(seen) != 1536:
        raise AssertionError(len(seen))
    for time_index, layer in enumerate(layers):
        data_seen = set()
        ancilla_seen = set()
        for gate in layer:
            ancilla = (gate["type"], gate["check"])
            if gate["data"] in data_seen or ancilla in ancilla_seen:
                raise AssertionError((time_index, gate))
            data_seen.add(gate["data"])
            ancilla_seen.add(ancilla)
    payload = {
        "schema_version": 1,
        "name": "verified clean 24-layer syndrome schedule for C28 x C4 component",
        "code": "[[112,16,7]]",
        "basis": {
            "rank": 48,
            "check_weight_histogram": {"16": 48},
            "data_degree_histogram_per_css_type": dict(sorted(Counter(map(int, check.sum(axis=0))).items())),
            "selected_catalog_indices": selection,
            "selected_checks": [
                {"seed": metadata[i][0], "shift_u": metadata[i][1], "shift_v": metadata[i][2]}
                for i in selection
            ],
        },
        "resources": {
            "dedicated_ancillas": 96,
            "cnot_count": 1536,
            "cnot_depth": 24,
            "layer_sizes": [len(layer) for layer in layers],
            "peak_layer_load": max(map(len, layers)),
        },
        "construction": {
            "one_type_colors_zero_based": [one_type[row, data] for kind, row, data, _left, _right in records if kind == 0],
            "z_time": "color",
            "x_time": "color + 8",
            "maximum_data_color_span": int(max(source_payload["spans"])),
            "local_order": "all Z-check CNOTs precede all X-check CNOTs at every data qubit",
        },
        "verified": {
            "collision_free": True,
            "all_tanner_edges_once": True,
            "clean_cross_ancilla_backaction": True,
            "local_z_before_x": True,
            "same_check_space_as_saved_code": True,
        },
        "layers": layers,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "basis_degree_histogram": payload["basis"]["data_degree_histogram_per_css_type"],
        "resources": payload["resources"],
        "verified": payload["verified"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
