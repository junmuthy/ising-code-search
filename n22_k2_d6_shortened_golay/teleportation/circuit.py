"""Fig. 10 teleportation-fidelity estimator for the ``[[22,2,6]]`` code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from n22_k2_d6_shortened_golay.tmr_postselection.circuit import (
    DATA_QUBITS,
    NoiseModel,
    build_circuit as build_preparation_circuit,
    encoded_plus_circuit,
)
from n22_k2_d6_shortened_golay.tmr_postselection.model import CodeArtifact
from n22_k2_d6_shortened_golay.tmr_postselection.partitions import PartitionCertificate


REFERENCE_START = 44
REFERENCE_QUBITS = tuple(range(REFERENCE_START, REFERENCE_START + 22))


@dataclass(frozen=True)
class TeleportationNoise:
    probability: float = 0.0
    cnot: bool = True
    measurement: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("noise probability must lie in [0,1]")

    @property
    def active(self) -> bool:
        return self.probability > 0 and (self.cnot or self.measurement)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EstimatorCircuit:
    text: str
    theta: float
    partition_count: int
    logical_count: int
    selected_logicals: tuple[int, ...]
    terminal_z_measurements: tuple[int, ...]
    terminal_x_measurements: tuple[int, ...]
    postselection_mask: tuple[int, ...]
    postselection_detectors: tuple[int, ...]
    detector_groups: dict[str, tuple[int, ...]]
    observable_groups: dict[str, tuple[int, ...]]
    operation_counts: dict[str, int]
    preparation_noise: dict[str, object]
    teleportation_noise: dict[str, object]
    artifact: dict[str, str]


def _shifted_encoder_lines(code: CodeArtifact, offset: int) -> list[str]:
    """Shift the Clifford encoder, whose emitted instructions have qubit targets only."""

    shifted: list[str] = []
    for instruction in encoded_plus_circuit(code):
        targets = instruction.targets_copy()
        if any(not target.is_qubit_target for target in targets):
            raise ValueError("encoded-plus circuit unexpectedly contains a non-qubit target")
        arguments = instruction.gate_args_copy()
        gate = instruction.name
        if arguments:
            rendered = ",".join(f"{value:.17g}" for value in arguments)
            gate = f"{gate}({rendered})"
        shifted.append(
            f"{gate} {' '.join(str(target.value + offset) for target in targets)}".rstrip()
        )
    return shifted


def _pauli_product(axis: str, support: Sequence[int], offset: int = 0) -> str:
    return "*".join(f"{axis}{int(qubit) + offset}" for qubit in support)


def _observable_line(index: int, measurements: Sequence[int], total: int) -> str:
    records = " ".join(f"rec[{int(measurement) - total}]" for measurement in measurements)
    return f"OBSERVABLE_INCLUDE({index}) {records}"


def build_estimator_circuit(
    *,
    code: CodeArtifact,
    partition_certificate: PartitionCertificate,
    theta: float,
    partition_count: int = 3,
    logical_count: int = 2,
    preparation_noise: NoiseModel | None = None,
    teleportation_noise: TeleportationNoise | None = None,
) -> EstimatorCircuit:
    """Build the paper's inverse-resource teleportation estimator.

    The resource patch is control and occupies qubits 0--21.  Its syndrome
    ancillas occupy 22--43.  The ideal inverse-reference patch is target and
    occupies qubits 44--65.  The estimator tail is ideal by default, exactly as
    in Fig. 10; ``teleportation_noise`` enables the later operational study.
    """

    preparation_noise = NoiseModel() if preparation_noise is None else preparation_noise
    teleportation_noise = (
        TeleportationNoise() if teleportation_noise is None else teleportation_noise
    )
    if logical_count not in (1, 2):
        raise ValueError("logical_count must be one or two")

    preparation = build_preparation_circuit(
        code=code,
        partition_certificate=partition_certificate,
        mode="full",
        protocol="single-final-check",
        theta=theta,
        partition_count=partition_count,
        logical_count=logical_count,
        noise=preparation_noise,
    )
    selected = tuple(preparation.logicals)
    lines = preparation.text.rstrip().splitlines()
    for local, qubit in enumerate(REFERENCE_QUBITS):
        half, within = divmod(local, 11)
        lines.append(f"QUBIT_COORDS({2 + half}, {within}) {qubit}")
    lines.extend(_shifted_encoder_lines(code, REFERENCE_START))
    lines.append("TICK")

    inverse_alpha = -float(theta) / math.pi
    for logical in selected:
        support = np.flatnonzero(code.logical_z[logical]).astype(int).tolist()
        lines.append(
            f"R_PAULI({inverse_alpha:.17g}) "
            f"{_pauli_product('Z', support, REFERENCE_START)}"
        )
    lines.append("TICK")

    pairs = [
        qubit
        for control, target in zip(DATA_QUBITS, REFERENCE_QUBITS, strict=True)
        for qubit in (control, target)
    ]
    pair_text = " ".join(map(str, pairs))
    lines.append(f"CX {pair_text}")
    if teleportation_noise.active and teleportation_noise.cnot:
        lines.append(f"DEPOLARIZE2({teleportation_noise.probability:.17g}) {pair_text}")
    lines.append("TICK")

    if teleportation_noise.active and teleportation_noise.measurement:
        lines.append(
            f"X_ERROR({teleportation_noise.probability:.17g}) "
            f"{' '.join(map(str, REFERENCE_QUBITS))}"
        )
        lines.append(
            f"Z_ERROR({teleportation_noise.probability:.17g}) "
            f"{' '.join(map(str, DATA_QUBITS))}"
        )

    measurements_before = 1 + max(
        measurement
        for group in preparation.measurement_groups.values()
        for measurement in group
    )
    terminal_z = tuple(range(measurements_before, measurements_before + len(REFERENCE_QUBITS)))
    lines.append(f"M {' '.join(map(str, REFERENCE_QUBITS))}")
    terminal_x = tuple(
        range(terminal_z[-1] + 1, terminal_z[-1] + 1 + len(DATA_QUBITS))
    )
    lines.append(f"MX {' '.join(map(str, DATA_QUBITS))}")
    total_measurements = terminal_x[-1] + 1

    observable_groups: dict[str, tuple[int, ...]] = {}
    observable_index = 0
    for logical in selected:
        support = np.flatnonzero(code.logical_z[logical]).astype(int).tolist()
        record_indices = tuple(terminal_z[qubit] for qubit in support)
        lines.append(_observable_line(observable_index, record_indices, total_measurements))
        observable_groups[f"logical_{logical}_Z_raw"] = (observable_index,)
        observable_index += 1
    for logical in selected:
        support = np.flatnonzero(code.logical_x[logical]).astype(int).tolist()
        record_indices = tuple(terminal_x[qubit] for qubit in support)
        lines.append(_observable_line(observable_index, record_indices, total_measurements))
        observable_groups[f"logical_{logical}_X_raw"] = (observable_index,)
        observable_index += 1

    operation_counts = dict(preparation.operation_counts)
    operation_counts.update(
        {
            "reference_encoder_clifford_instructions": len(_shifted_encoder_lines(code, REFERENCE_START)),
            "ideal_inverse_logical_rotations": len(selected),
            "teleportation_cnots": len(DATA_QUBITS),
            "terminal_measurements": 2 * len(DATA_QUBITS),
        }
    )
    return EstimatorCircuit(
        text="\n".join(lines) + "\n",
        theta=float(theta),
        partition_count=int(partition_count),
        logical_count=int(logical_count),
        selected_logicals=selected,
        terminal_z_measurements=terminal_z,
        terminal_x_measurements=terminal_x,
        postselection_mask=tuple(preparation.postselection_mask),
        postselection_detectors=tuple(preparation.postselection_detectors),
        detector_groups=dict(preparation.detector_groups),
        observable_groups=observable_groups,
        operation_counts=operation_counts,
        preparation_noise=preparation_noise.to_json(),
        teleportation_noise=teleportation_noise.to_json(),
        artifact=dict(preparation.artifact),
    )
