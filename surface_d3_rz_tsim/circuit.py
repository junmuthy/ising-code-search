"""TMR angle formulas and tsim circuits for the d=3 surface-code demo."""

from __future__ import annotations

from dataclasses import dataclass
import math

import tsim

from surface_code import (
    ANCILLA_QUBITS,
    DATA_QUBITS,
    LOGICAL_Z,
    QUBIT_COORDS,
    X_ANCILLAS,
    X_CHECKS,
    Z_CHECKS,
    encoded_plus_physical_circuit,
    logical_pauli_text,
    physical_pauli_text,
)


# Canonical four-layer schedule emitted by Stim's rotated_memory_x generator.
CNOT_LAYERS = (
    ((2, 3), (16, 17), (11, 12), (15, 14), (10, 9), (19, 18)),
    ((2, 1), (16, 15), (11, 10), (8, 14), (3, 9), (12, 18)),
    ((16, 10), (11, 5), (25, 19), (8, 9), (17, 18), (12, 13)),
    ((16, 8), (11, 3), (25, 17), (1, 9), (10, 18), (5, 13)),
)


@dataclass(frozen=True)
class CircuitBundle:
    circuit: tsim.Circuit
    mode: str
    basis: str
    theta: float
    theta_star: float
    partitions: int
    detector_groups: dict[str, tuple[int, ...]]


def physical_tmr_angle(theta: float, partitions: int = 3) -> float:
    """Return the common physical angle preparing logical RZ(theta)|+>.

    For odd M, (-i)^M tan(theta_star/2)^M = -i tan(theta/2).
    This includes the otherwise easy-to-miss sign reversal at M=3.
    """

    if partitions <= 0 or partitions % 2 == 0:
        raise ValueError("This introductory implementation requires odd M")
    tangent = math.tan(theta / 2)
    signed_target = ((-1) ** (partitions // 2)) * tangent
    signed_root = math.copysign(
        abs(signed_target) ** (1 / partitions),
        signed_target,
    )
    return 2 * math.atan(signed_root)


def ideal_tmr_success(theta: float, partitions: int = 3) -> float:
    """Return the noise-free probability of trivial-syndrome projection."""

    theta_star = physical_tmr_angle(theta, partitions)
    return math.cos(theta_star / 2) ** (2 * partitions) + math.sin(
        theta_star / 2
    ) ** (2 * partitions)


def _coordinate_lines() -> list[str]:
    return [
        f"QUBIT_COORDS({x}, {y}) {qubit}"
        for qubit, (x, y) in sorted(QUBIT_COORDS.items())
    ]


def _encoder_lines() -> list[str]:
    return [str(instruction) for instruction in encoded_plus_physical_circuit()]


def _tmr_line(theta_star: float) -> str:
    alpha = theta_star / math.pi
    targets = " ".join(str(qubit) for qubit in LOGICAL_Z)
    return f"R_Z({alpha:.17g}) {targets}"


def _append_logical_readout(lines: list[str], basis: str) -> None:
    lines.append(f"MPP {logical_pauli_text(basis)}")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")


def build_projection_circuit(theta: float, basis: str = "X") -> CircuitBundle:
    """Build the simplest TMR demo using ideal MPP code-space projection."""

    basis = basis.upper()
    partitions = 3
    theta_star = physical_tmr_angle(theta, partitions)
    lines = [*_coordinate_lines(), *_encoder_lines(), "TICK", _tmr_line(theta_star), "TICK"]

    detector_indices: list[int] = []
    for support in X_CHECKS:
        lines.append(f"MPP {physical_pauli_text(axis='X', support=support)}")
        lines.append("DETECTOR rec[-1]")
        detector_indices.append(len(detector_indices))
    for support in Z_CHECKS:
        lines.append(f"MPP {physical_pauli_text(axis='Z', support=support)}")
        lines.append("DETECTOR rec[-1]")
        detector_indices.append(len(detector_indices))
    _append_logical_readout(lines, basis)

    return CircuitBundle(
        circuit=tsim.Circuit("\n".join(lines)),
        mode="projection",
        basis=basis,
        theta=theta,
        theta_star=theta_star,
        partitions=partitions,
        detector_groups={"projection": tuple(detector_indices)},
    )


def _syndrome_round_lines() -> list[str]:
    lines = [f"H {' '.join(map(str, X_ANCILLAS))}", "TICK"]
    for layer in CNOT_LAYERS:
        flat_targets = " ".join(str(qubit) for pair in layer for qubit in pair)
        lines.extend((f"CX {flat_targets}", "TICK"))
    lines.extend(
        (
            f"H {' '.join(map(str, X_ANCILLAS))}",
            "TICK",
            f"MR {' '.join(map(str, ANCILLA_QUBITS))}",
        )
    )
    # The encoded state has a fixed all-+1 stabilizer frame, so raw syndrome
    # bits are meaningful and make the postselection rule especially clear.
    lines.extend(f"DETECTOR rec[-{offset}]" for offset in range(8, 0, -1))
    lines.append("TICK")
    return lines


def build_scheduled_circuit(
    theta: float,
    basis: str = "X",
    initialization_rounds: int = 2,
    post_tmr_rounds: int = 2,
) -> CircuitBundle:
    """Build the eight-ancilla, four-CNOT-layer syndrome-extraction version."""

    if initialization_rounds < 1 or post_tmr_rounds < 1:
        raise ValueError("At least one initialization and post-TMR round is required")
    basis = basis.upper()
    partitions = 3
    theta_star = physical_tmr_angle(theta, partitions)
    lines = [
        *_coordinate_lines(),
        *_encoder_lines(),
        f"R {' '.join(map(str, ANCILLA_QUBITS))}",
        "TICK",
    ]

    detector_groups: dict[str, tuple[int, ...]] = {}
    detector_cursor = 0
    for phase, rounds in (
        ("initialization", initialization_rounds),
        ("post_tmr", post_tmr_rounds),
    ):
        phase_detectors: list[int] = []
        if phase == "post_tmr":
            lines.extend((_tmr_line(theta_star), "TICK"))
        for _ in range(rounds):
            lines.extend(_syndrome_round_lines())
            phase_detectors.extend(range(detector_cursor, detector_cursor + 8))
            detector_cursor += 8
        detector_groups[phase] = tuple(phase_detectors)

    _append_logical_readout(lines, basis)
    return CircuitBundle(
        circuit=tsim.Circuit("\n".join(lines)),
        mode="scheduled",
        basis=basis,
        theta=theta,
        theta_star=theta_star,
        partitions=partitions,
        detector_groups=detector_groups,
    )


def build_circuit(mode: str, theta: float, basis: str) -> CircuitBundle:
    if mode == "projection":
        return build_projection_circuit(theta, basis)
    if mode == "scheduled":
        return build_scheduled_circuit(theta, basis)
    raise ValueError(f"Unknown circuit mode {mode!r}")
