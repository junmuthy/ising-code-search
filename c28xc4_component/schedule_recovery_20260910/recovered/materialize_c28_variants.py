#!/usr/bin/env python3
"""Make clean sequential variants of the selected C28 schedule."""

import json
from pathlib import Path

source = json.loads(Path("/tmp/c28_clean_d24.json").read_text())
base = source["construction"]["one_type_colors_zero_based"]
supports = []
for layer in source["layers"]:
    for gate in layer:
        if gate["type"] == "Z":
            supports.append((int(gate["check"]), int(gate["data"]), int(layer.index(gate))))
# Recover color by the saved Z gates, avoiding dependence on traversal ordering.
color = {}
for time, layer in enumerate(source["layers"]):
    for gate in layer:
        if gate["type"] == "Z":
            color[int(gate["check"]), int(gate["data"])] = time

for name, x_time in (
    ("same", lambda c: 16 + c),
    ("reversed", lambda c: 31 - c),
):
    layers = [[] for _ in range(32)]
    for (check, data), c in sorted(color.items()):
        layers[c].append({"type": "Z", "check": check, "data": data})
        layers[x_time(c)].append({"type": "X", "check": check, "data": data})
    payload = {**source, "name": f"clean sequential {name}-order schedule", "resources": {**source["resources"], "cnot_depth": 32, "layer_sizes": [len(layer) for layer in layers], "peak_layer_load": 48}, "layers": layers}
    Path(f"/tmp/c28_clean_d32_{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
