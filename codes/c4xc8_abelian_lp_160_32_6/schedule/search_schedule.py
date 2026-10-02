#!/usr/bin/env python3
"""Find translation-invariant fold/time-reversed schedules for the LP160 code."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import z3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import code  # noqa: E402


DEPTH = 12


def sub(left, right):
    return ((left[0] - right[0]) % 4, (left[1] - right[1]) % 8)


def classify(row: int, data: int):
    family, anchor_index = divmod(row, code.GROUP_ORDER)
    block, within = divmod(data, code.GROUP_ORDER)
    return family, block, sub(code.GROUP[within], code.GROUP[anchor_index])


def collect_orbits(matrix: np.ndarray):
    keys = sorted(
        {
            classify(row, int(data))
            for row in range(matrix.shape[0])
            for data in np.flatnonzero(matrix[row])
        }
    )
    if len(keys) != 18:
        raise AssertionError(len(keys))
    lookup = {key: index for index, key in enumerate(keys)}
    return keys, lookup


def fold_row_map(hx: np.ndarray, hz: np.ndarray):
    permutation = code.fold_permutation()
    mapped = code.permute_rows(hx, permutation)
    rows = {np.packbits(row).tobytes(): index for index, row in enumerate(hz)}
    result = []
    for row in mapped:
        result.append(rows[np.packbits(row).tobytes()])
    return result


def build_problem():
    hx, hz = code.build_checks()
    xkeys, xlookup = collect_orbits(hx)
    zkeys, zlookup = collect_orbits(hz)
    row_map = fold_row_map(hx, hz)
    fold = code.fold_permutation()
    partners = [None] * len(xkeys)
    for row in range(hx.shape[0]):
        for data in np.flatnonzero(hx[row]):
            xorbit = xlookup[classify(row, int(data))]
            zorbit = zlookup[classify(row_map[row], int(fold[int(data)]))]
            if partners[xorbit] is not None and partners[xorbit] != zorbit:
                raise AssertionError("fold does not map an X orbit to one Z orbit")
            partners[xorbit] = zorbit
    if sorted(partners) != list(range(len(zkeys))):
        raise AssertionError("fold orbit matching is not bijective")
    inverse_partners = {zorbit: xorbit for xorbit, zorbit in enumerate(partners)}

    colors = [z3.Int(f"c_{index}") for index in range(len(xkeys))]
    solver = z3.Solver()
    for color in colors:
        solver.add(color >= 0, color < DEPTH)

    # Each X-check ancilla uses its nine incident edge-orbit colors once.
    for family in range(2):
        solver.add(z3.Distinct(*[colors[i] for i, key in enumerate(xkeys) if key[0] == family]))

    # The Z schedule is the fold image in time reverse.  The fold makes Z
    # ancilla collisions equivalent to the X constraints above.
    def ztime(zorbit):
        return DEPTH - 1 - colors[inverse_partners[zorbit]]

    # Every data block has a translation-invariant incident-orbit pattern.
    # All simultaneous X and Z interactions on one data qubit must be distinct.
    for block in range(5):
        data = block * code.GROUP_ORDER
        incident = []
        for row in np.flatnonzero(hx[:, data]):
            incident.append(colors[xlookup[classify(int(row), data)]])
        for row in np.flatnonzero(hz[:, data]):
            incident.append(ztime(zlookup[classify(int(row), data)]))
        solver.add(z3.Distinct(*incident))

    # Exact cancellation of X-ancilla -> Z-ancilla back-action.  By
    # translation invariance, fix each X anchor to identity and scan relative
    # Z anchors.
    parity_constraints = 0
    for xfamily in range(2):
        xrow = xfamily * code.GROUP_ORDER
        xsupport = set(np.flatnonzero(hx[xrow]).astype(int))
        for zrow in range(hz.shape[0]):
            overlap = sorted(xsupport & set(np.flatnonzero(hz[zrow]).astype(int)))
            if not overlap:
                continue
            terms = []
            for data in overlap:
                xtime = colors[xlookup[classify(xrow, data)]]
                ztime_value = ztime(zlookup[classify(zrow, data)])
                terms.append(z3.If(xtime < ztime_value, 1, 0))
            solver.add(z3.Sum(*terms) % 2 == 0)
            parity_constraints += 1

    # Fix a harmless layer-label symmetry: one chosen edge can be placed in
    # the first/last time-reversal pair.
    solver.add(colors[0] == 0)
    # Balance the translation orbits across each time-reversal layer pair.
    # Each X orbit contributes 32 gates in its own layer and another 32 Z
    # gates in the reversed layer, so this makes every layer exactly 96 CNOTs.
    for layer in range(DEPTH // 2):
        reverse = DEPTH - 1 - layer
        solver.add(
            z3.Sum(*[z3.If(z3.Or(color == layer, color == reverse), 1, 0) for color in colors])
            == 3
        )
    return solver, colors, xkeys, zkeys, partners, parity_constraints


def schedule_from_colors(values, xkeys, zkeys, partners):
    hx, hz = code.build_checks()
    xlookup = {key: index for index, key in enumerate(xkeys)}
    zlookup = {key: index for index, key in enumerate(zkeys)}
    inverse = {zorbit: xorbit for xorbit, zorbit in enumerate(partners)}
    fold = code.fold_permutation()
    row_map = fold_row_map(hx, hz)
    layers = [[] for _ in range(DEPTH)]
    for row in range(hx.shape[0]):
        for data in np.flatnonzero(hx[row]).astype(int):
            orbit = xlookup[classify(row, data)]
            layers[values[orbit]].append({"type": "X", "check": row, "data": int(data)})
    for row in range(hz.shape[0]):
        for data in np.flatnonzero(hz[row]).astype(int):
            orbit = zlookup[classify(row, data)]
            layer = DEPTH - 1 - values[inverse[orbit]]
            layers[layer].append({"type": "Z", "check": row, "data": int(data)})

    # Structural verification.
    expected = {
        (kind, row, int(data))
        for kind, matrix in (("X", hx), ("Z", hz))
        for row in range(matrix.shape[0])
        for data in np.flatnonzero(matrix[row])
    }
    actual = {(gate["type"], gate["check"], gate["data"]) for layer in layers for gate in layer}
    assert expected == actual and len(actual) == 1152
    times = {}
    for layer_index, layer in enumerate(layers):
        assert len({gate["data"] for gate in layer}) == len(layer)
        assert len({(gate["type"], gate["check"]) for gate in layer}) == len(layer)
        for gate in layer:
            times[(gate["type"], gate["check"], gate["data"])] = layer_index
    for xrow in range(hx.shape[0]):
        for zrow in range(hz.shape[0]):
            overlap = set(np.flatnonzero(hx[xrow]).astype(int)) & set(np.flatnonzero(hz[zrow]).astype(int))
            inversions = sum(times[("X", xrow, data)] < times[("Z", zrow, data)] for data in overlap)
            assert inversions % 2 == 0
    # Fold plus time reversal edge equality.
    zedges_by_layer = [
        {(gate["check"], gate["data"]) for gate in layer if gate["type"] == "Z"}
        for layer in layers
    ]
    for layer_index, layer in enumerate(layers):
        for gate in layer:
            if gate["type"] == "X":
                assert (row_map[gate["check"]], int(fold[gate["data"]])) in zedges_by_layer[DEPTH - 1 - layer_index]
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    solver, colors, xkeys, zkeys, partners, parity_count = build_problem()
    print(json.dumps({"stage": "solve", "variables": len(colors), "parity_constraints": parity_count}), flush=True)
    status = solver.check()
    if status != z3.sat:
        print(json.dumps({"status": str(status)}))
        return
    model = solver.model()
    values = tuple(model.eval(color).as_long() for color in colors)
    layers = schedule_from_colors(values, xkeys, zkeys, partners)
    result = {
        "status": "sat",
        "depth": DEPTH,
        "colors": list(values),
        "x_orbits": [[family, block, list(delta)] for family, block, delta in xkeys],
        "z_orbits": [[family, block, list(delta)] for family, block, delta in zkeys],
        "fold_x_to_z_orbit": partners,
        "layer_sizes": [len(layer) for layer in layers],
        "layers": layers,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "layers"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
