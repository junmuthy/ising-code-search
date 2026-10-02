"""Guarded-bulk Stim circuits for the LP160 syndrome schedule."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import stim


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
import code  # noqa: E402

NUM_DATA = 160
NUM_CHECKS = 64
X_ANCILLA_START = NUM_DATA
Z_ANCILLA_START = NUM_DATA + NUM_CHECKS
DATA = tuple(range(NUM_DATA))
X_ANCILLAS = tuple(range(X_ANCILLA_START, Z_ANCILLA_START))
Z_ANCILLAS = tuple(range(Z_ANCILLA_START, Z_ANCILLA_START + NUM_CHECKS))
DEFAULT_SCHEDULE = Path(__file__).resolve().parent / "optimal_depth12_schedule.json"


def load_layers(path: Path = DEFAULT_SCHEDULE) -> list[list[dict]]:
    return [layer["gates"] for layer in json.loads(path.read_text())["layers"]]


def _records(circuit: stim.Circuit, measurements: Iterable[int]):
    now = circuit.num_measurements
    return [stim.target_rec(int(index) - now) for index in measurements]


def _syndrome_round(
    circuit: stim.Circuit,
    layers: list[list[dict]],
    *,
    noisy: bool,
    probability: float,
    injected_errors: dict[int, list[tuple[str, int]]] | None = None,
):
    circuit.append("RX", X_ANCILLAS)
    circuit.append("R", Z_ANCILLAS)
    circuit.append("TICK")
    for layer_index, layer in enumerate(layers):
        pairs: list[int] = []
        for gate in layer:
            if gate["type"] == "X":
                pairs.extend((X_ANCILLA_START + gate["check"], gate["data"]))
            else:
                pairs.extend((gate["data"], Z_ANCILLA_START + gate["check"]))
        circuit.append("CX", pairs)
        if noisy:
            circuit.append("DEPOLARIZE2", pairs, probability)
        for pauli, data in (injected_errors or {}).get(layer_index, ()):
            circuit.append(f"{pauli}_ERROR", [data], probability)
        circuit.append("TICK")
    start_x = circuit.num_measurements
    circuit.append("MX", X_ANCILLAS)
    start_z = circuit.num_measurements
    circuit.append("M", Z_ANCILLAS)
    circuit.append("TICK")
    return (
        tuple(range(start_x, start_x + NUM_CHECKS)),
        tuple(range(start_z, start_z + NUM_CHECKS)),
    )


def build_guarded_bulk_circuit(
    basis: str,
    *,
    probability: float = 1e-3,
    schedule_path: Path = DEFAULT_SCHEDULE,
    injected_middle_errors: dict[int, list[tuple[str, int]]] | None = None,
) -> stim.Circuit:
    """Put one noisy extraction round between two noiseless guard rounds."""
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    hx, hz = code.build_checks()
    logical_z, logical_x = code.build_logical_bases()
    checks = {"X": hx, "Z": hz}
    logicals = {"X": logical_x, "Z": logical_z}
    layers = load_layers(schedule_path)
    circuit = stim.Circuit()
    circuit.append("RX" if basis == "X" else "R", DATA)
    circuit.append("TICK")
    previous = None
    for round_index in range(3):
        current = _syndrome_round(
            circuit,
            layers,
            noisy=round_index == 1 and injected_middle_errors is None,
            probability=probability,
            injected_errors=injected_middle_errors if round_index == 1 else None,
        )
        if round_index == 0:
            selected = current[0 if basis == "X" else 1]
            for check, measurement in enumerate(selected):
                circuit.append("DETECTOR", _records(circuit, (measurement,)))
        else:
            for kind in range(2):
                for check in range(NUM_CHECKS):
                    circuit.append(
                        "DETECTOR",
                        _records(circuit, (current[kind][check], previous[kind][check])),
                    )
        previous = current

    start = circuit.num_measurements
    circuit.append("MX" if basis == "X" else "M", DATA)
    data_measurements = tuple(range(start, start + NUM_DATA))
    kind = 0 if basis == "X" else 1
    for check, row in enumerate(checks[basis]):
        support = np.flatnonzero(row).astype(int)
        circuit.append(
            "DETECTOR",
            _records(
                circuit,
                (previous[kind][check], *(data_measurements[data] for data in support)),
            ),
        )
    for logical, row in enumerate(logicals[basis]):
        support = np.flatnonzero(row).astype(int)
        circuit.append(
            "OBSERVABLE_INCLUDE",
            _records(circuit, (data_measurements[data] for data in support)),
            logical,
        )
    return circuit


def last_cnot_injections(support: list[int], pauli: str, schedule_path=DEFAULT_SCHEDULE):
    """Place a data-only Pauli after each support qubit's final middle-round CNOT."""
    layers = load_layers(schedule_path)
    last = {}
    for layer_index, layer in enumerate(layers):
        for gate in layer:
            last[int(gate["data"])] = layer_index
    result: dict[int, list[tuple[str, int]]] = {}
    for data in support:
        result.setdefault(last[data], []).append((pauli, data))
    return result
