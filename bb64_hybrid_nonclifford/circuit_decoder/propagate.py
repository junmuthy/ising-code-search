"""Exact Pauli-difference propagation through the scheduled BB64 circuit."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from .ledger import FaultMechanism, parse_instruction


_RECORD = re.compile(r"^rec\[(-?\d+)\]$")


@dataclass(frozen=True)
class ParsedInstruction:
    index: int
    name: str
    arguments: tuple[float, ...]
    targets: tuple[str, ...]


@dataclass(frozen=True)
class ParsedCircuit:
    instructions: tuple[ParsedInstruction, ...]
    num_qubits: int
    num_measurements: int
    num_detectors: int
    num_rotations: int


@dataclass(frozen=True)
class PauliSignature:
    detector_mask: int
    final_x_mask: int
    final_z_mask: int
    rotation_sign_mask: int


def _record_value(records: list[int], token: str) -> int:
    match = _RECORD.match(token)
    if match is None:
        raise ValueError(f"invalid measurement-record target: {token}")
    offset = int(match.group(1))
    index = offset if offset >= 0 else len(records) + offset
    if not 0 <= index < len(records):
        raise ValueError(f"measurement-record reference is out of range: {token}")
    return records[index]


def parse_circuit(circuit_text: str) -> ParsedCircuit:
    instructions: list[ParsedInstruction] = []
    max_qubit = -1
    measurements = 0
    detectors = 0
    rotations = 0
    non_qubit = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS"}
    for index, line in enumerate(circuit_text.splitlines()):
        if not line.strip():
            continue
        name, arguments, targets = parse_instruction(line)
        instructions.append(ParsedInstruction(index, name, arguments, targets))
        if name in ("M", "MX"):
            measurements += len(targets)
        elif name == "MPP":
            measurements += 1
        elif name == "DETECTOR":
            detectors += 1
        elif name == "R_Z":
            rotations += len(targets)
        if name not in non_qubit:
            for token in targets:
                if token.startswith("rec[") or "*" in token:
                    continue
                try:
                    max_qubit = max(max_qubit, int(token))
                except ValueError:
                    pass
    return ParsedCircuit(tuple(instructions), max_qubit + 1, measurements, detectors, rotations)


def _toggle_mask(bits: np.ndarray, mask: int) -> None:
    while mask:
        least = mask & -mask
        bits[least.bit_length() - 1] ^= 1
        mask ^= least


def propagate_faults(
    circuit: ParsedCircuit,
    faults: Iterable[FaultMechanism],
) -> PauliSignature:
    """Propagate one or more inserted Pauli faults to detectors and boundary.

    The state is a *difference* from the corresponding ideal coherent
    trajectory.  At an ``R_Z`` target, an X component toggles that rotation's
    sign according to ``R_Z(phi) X = X R_Z(-phi)``.  The Pauli itself remains
    in the frame and continues through the subsequent Clifford circuit.
    """

    by_instruction: dict[int, list[FaultMechanism]] = {}
    for fault in faults:
        by_instruction.setdefault(fault.instruction_index, []).append(fault)
    x = np.zeros(circuit.num_qubits, dtype=np.uint8)
    z = np.zeros(circuit.num_qubits, dtype=np.uint8)
    records: list[int] = []
    detectors: list[int] = []
    rotation_index = 0
    sign_mask = 0
    noise_names = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}

    for instruction in circuit.instructions:
        name = instruction.name
        targets = instruction.targets
        if name in ("R", "RX"):
            for token in targets:
                qubit = int(token)
                x[qubit] = 0
                z[qubit] = 0
        elif name == "CX":
            if len(targets) % 2:
                raise ValueError(f"odd CX target count at instruction {instruction.index}")
            for offset in range(0, len(targets), 2):
                control, raw_target = targets[offset : offset + 2]
                target = int(raw_target)
                if control.startswith("rec["):
                    x[target] ^= _record_value(records, control)
                else:
                    control_qubit = int(control)
                    x[target] ^= x[control_qubit]
                    z[control_qubit] ^= z[target]
        elif name == "R_Z":
            for token in targets:
                qubit = int(token)
                if x[qubit]:
                    # There are only 24 rotations, so one Python integer is
                    # a compact exact sign-history record.
                    sign_mask ^= 1 << rotation_index
                rotation_index += 1
        elif name == "M":
            for token in targets:
                qubit = int(token)
                records.append(int(x[qubit]))
                x[qubit] = z[qubit] = 0
        elif name == "MX":
            for token in targets:
                qubit = int(token)
                records.append(int(z[qubit]))
                x[qubit] = z[qubit] = 0
        elif name == "DETECTOR":
            value = 0
            for token in targets:
                value ^= _record_value(records, token)
            detectors.append(value)
        elif name == "OBSERVABLE_INCLUDE":
            pass
        elif name == "MPP":
            raise NotImplementedError("the scheduled history circuit must not contain MPP")
        elif name in noise_names or name in ("TICK", "QUBIT_COORDS", "X", "Z"):
            pass
        else:
            raise NotImplementedError(f"unsupported instruction {name}")

        # Noise channels in the text occur exactly where their sampled Pauli
        # acts.  Inject the selected elementary outcome after interpreting the
        # otherwise inert noise instruction.
        for fault in by_instruction.get(instruction.index, ()):
            _toggle_mask(x, fault.x_mask)
            _toggle_mask(z, fault.z_mask)

    detector_mask = sum(int(value) << index for index, value in enumerate(detectors))
    final_x_mask = sum(int(value) << index for index, value in enumerate(x[:64]))
    final_z_mask = sum(int(value) << index for index, value in enumerate(z[:64]))
    return PauliSignature(detector_mask, final_x_mask, final_z_mask, sign_mask)
