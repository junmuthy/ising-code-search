"""Enumerate clean C4-invariant fold/time-reversal schedules with Z3."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import z3


NUM_QUBITS = 32
NUM_CHECKS = 16
LOGICAL_ORDER = 4
THICKNESS = 8


def matrix_from_supports(supports: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(supports), NUM_QUBITS), dtype=np.uint8)
    for row, support in enumerate(supports):
        matrix[row, support] = 1
    return matrix


def act_rows(matrix: np.ndarray, permutation: list[int] | tuple[int, ...]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, list(permutation)] = matrix
    return output


def build_edge_orbits(
    checks: np.ndarray, check_action: list[int]
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    """Partition X-check Tanner edges into the 32 free physical-C4 orbits."""

    data_action = [
        ((data // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + data % THICKNESS
        for data in range(NUM_QUBITS)
    ]
    edges = {
        (check, int(data))
        for check, row in enumerate(checks)
        for data in np.flatnonzero(row)
    }
    orbits: list[list[tuple[int, int]]] = []
    edge_orbit: dict[tuple[int, int], int] = {}
    for edge in sorted(edges):
        if edge in edge_orbit:
            continue
        orbit: list[tuple[int, int]] = []
        current = edge
        while current not in orbit:
            if current not in edges:
                raise RuntimeError("translation leaves the Tanner graph")
            orbit.append(current)
            current = (check_action[current[0]], data_action[current[1]])
        if len(orbit) != LOGICAL_ORDER:
            raise RuntimeError("Tanner-edge orbit is not free under C4")
        index = len(orbits)
        orbits.append(orbit)
        for member in orbit:
            edge_orbit[member] = index
    return orbits, edge_orbit


def reverse_colors(colors: Iterable[int], depth: int) -> tuple[int, ...]:
    return tuple(depth - 1 - int(color) for color in colors)


def canonical_colors(colors: Iterable[int], depth: int) -> tuple[int, ...]:
    direct = tuple(int(color) for color in colors)
    reverse = reverse_colors(direct, depth)
    return min(direct, reverse)


def schedule_id(fold_index: int, colors: Iterable[int]) -> str:
    payload = json.dumps(
        {"fold_index": int(fold_index), "colors": list(colors)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class EnumerationResult:
    status: str
    colors: tuple[int, ...] | None
    solver_seconds: float
    reason_unknown: str | None = None


class ScheduleEnumerator:
    """Incremental enumerator for one fold and one fixed circuit depth."""

    def __init__(
        self,
        checks_x: np.ndarray,
        permutation: list[int],
        check_action: list[int],
        *,
        depth: int = 12,
        timeout_seconds: float = 10.0,
        random_seed: int = 0,
        minimum_color_changes: int = 1,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        if minimum_color_changes < 1:
            raise ValueError("minimum_color_changes must be positive")
        self.checks_x = np.asarray(checks_x, dtype=np.uint8)
        self.permutation = [int(value) for value in permutation]
        self.depth = int(depth)
        self.minimum_color_changes = int(minimum_color_changes)
        self.orbits, self.edge_orbit = build_edge_orbits(self.checks_x, check_action)
        if len(self.orbits) != 32:
            raise RuntimeError(f"expected 32 translated edge orbits, found {len(self.orbits)}")
        self.variables = [z3.Int(f"orbit_{index}") for index in range(len(self.orbits))]
        self.solver = z3.Solver()
        self.solver.set(timeout=max(1, round(timeout_seconds * 1000)))
        self.solver.set(random_seed=int(random_seed))
        self._add_structural_constraints()

    def _add_structural_constraints(self) -> None:
        checks_z = act_rows(self.checks_x, self.permutation)
        for variable in self.variables:
            self.solver.add(variable >= 0, variable < self.depth)

        # X/X ancilla and data collisions. Z/Z follows from the folded reverse.
        for check in range(NUM_CHECKS):
            incident = [
                self.variables[self.edge_orbit[(check, int(data))]]
                for data in np.flatnonzero(self.checks_x[check])
            ]
            self.solver.add(z3.Distinct(*incident))
        for data in range(NUM_QUBITS):
            incident = [
                self.variables[self.edge_orbit[(check, data)]]
                for check in range(NUM_CHECKS)
                if self.checks_x[check, data]
            ]
            self.solver.add(z3.Distinct(*incident))

        # Simultaneous X/Z data collisions.
        for data in range(NUM_QUBITS):
            x_incident = [
                self.variables[self.edge_orbit[(check, data)]]
                for check in range(NUM_CHECKS)
                if self.checks_x[check, data]
            ]
            source = self.permutation[data]
            z_incident = [
                self.depth - 1 - self.variables[self.edge_orbit[(check, source)]]
                for check in range(NUM_CHECKS)
                if self.checks_x[check, source]
            ]
            self.solver.add(z3.Distinct(*(x_incident + z_incident)))

        # Exact even cross-ancilla backaction parity.
        for check_x in range(NUM_CHECKS):
            for check_z in range(NUM_CHECKS):
                comparisons: list[z3.BoolRef] = []
                for data in range(NUM_QUBITS):
                    if self.checks_x[check_x, data] and checks_z[check_z, data]:
                        x_time = self.variables[self.edge_orbit[(check_x, data)]]
                        source = self.permutation[data]
                        z_time = (
                            self.depth
                            - 1
                            - self.variables[self.edge_orbit[(check_z, source)]]
                        )
                        comparisons.append(x_time < z_time)
                if len(comparisons) % 2:
                    raise RuntimeError("input checks are not CSS orthogonal")
                if comparisons:
                    parity = comparisons[0]
                    for comparison in comparisons[1:]:
                        parity = z3.Xor(parity, comparison)
                    self.solver.add(z3.Not(parity))

    def exclude(self, colors: Iterable[int]) -> None:
        """Exclude a coloring and its globally time-reversed equivalent."""

        direct = tuple(int(color) for color in colors)
        if len(direct) != len(self.variables):
            raise ValueError("wrong number of edge-orbit colors")
        alternatives = {direct, reverse_colors(direct, self.depth)}
        for alternative in alternatives:
            changed = [
                (variable != value, 1)
                for variable, value in zip(self.variables, alternative, strict=True)
            ]
            self.solver.add(z3.PbGe(changed, self.minimum_color_changes))

    def next(self) -> EnumerationResult:
        started = time.perf_counter()
        status = self.solver.check()
        seconds = time.perf_counter() - started
        if status != z3.sat:
            return EnumerationResult(
                status=str(status),
                colors=None,
                solver_seconds=seconds,
                reason_unknown=self.solver.reason_unknown() if status == z3.unknown else None,
            )
        model = self.solver.model()
        colors = canonical_colors(
            (model.eval(variable).as_long() for variable in self.variables),
            self.depth,
        )
        self.exclude(colors)
        return EnumerationResult("sat", colors, seconds)


def build_schedule_record(
    checks_x: np.ndarray,
    permutation: list[int],
    edge_orbit: dict[tuple[int, int], int],
    colors: Iterable[int],
    *,
    depth: int,
    fold_index: int,
    fold: dict[str, Any],
) -> dict[str, Any]:
    """Materialize and independently validate a folded simultaneous schedule."""

    color_values = tuple(int(color) for color in colors)
    checks_z = act_rows(checks_x, permutation)
    layers: list[dict[str, Any]] = []
    for time_index in range(depth):
        gates: list[dict[str, Any]] = []
        for check, row in enumerate(checks_x):
            for data_value in np.flatnonzero(row):
                data = int(data_value)
                if color_values[edge_orbit[(check, data)]] == time_index:
                    gates.append({"type": "X", "check": check, "data": data})
                target = permutation[data]
                if depth - 1 - color_values[edge_orbit[(check, data)]] == time_index:
                    gates.append({"type": "Z", "check": check, "data": target})
        layers.append({"round": time_index + 1, "gates": gates})

    expected = {
        ("X", check, int(data))
        for check, row in enumerate(checks_x)
        for data in np.flatnonzero(row)
    } | {
        ("Z", check, int(data))
        for check, row in enumerate(checks_z)
        for data in np.flatnonzero(row)
    }
    observed = {
        (gate["type"], gate["check"], gate["data"])
        for layer in layers
        for gate in layer["gates"]
    }
    if expected != observed:
        raise RuntimeError("schedule does not cover every Tanner edge exactly once")
    for layer in layers:
        ancillas: set[tuple[str, int]] = set()
        data_seen: set[int] = set()
        for gate in layer["gates"]:
            ancilla = (str(gate["type"]), int(gate["check"]))
            data = int(gate["data"])
            if ancilla in ancillas or data in data_seen:
                raise RuntimeError("schedule collision")
            ancillas.add(ancilla)
            data_seen.add(data)

    times = {
        (gate["type"], gate["check"], gate["data"]): time_index
        for time_index, layer in enumerate(layers)
        for gate in layer["gates"]
    }
    for check_x in range(NUM_CHECKS):
        for check_z in range(NUM_CHECKS):
            overlap = np.flatnonzero(checks_x[check_x] & checks_z[check_z])
            inversions = sum(
                times[("X", check_x, int(data))]
                < times[("Z", check_z, int(data))]
                for data in overlap
            )
            if inversions % 2:
                raise RuntimeError("cross-ancilla backaction parity failure")

    return {
        "schema_version": 1,
        "name": "alternative clean simultaneous C4-invariant fold/time-reversal schedule",
        "schedule_id": schedule_id(fold_index, color_values),
        "fold_index": int(fold_index),
        "fold": fold,
        "cnot_depth": depth,
        "cnot_count": len(expected),
        "dedicated_ancillas": 32,
        "translation_invariant_layers": True,
        "fold_P_plus_time_reversal": True,
        "clean_cross_ancilla_backaction": True,
        "edge_orbit_colors_zero_based": list(color_values),
        "layers": layers,
    }
