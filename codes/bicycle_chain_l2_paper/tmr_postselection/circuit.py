"""Build ideal and circuit-level bicycle-chain TMR postselection experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import stim

from .model import (
    CodeArtifact,
    independent_indices,
    initialization_correction_map,
    row_relations,
    support_to_pauli,
)
from .partitions import Partition, PartitionCertificate, certify_partitions


Mode = Literal["ideal-projection", "scheduled-tmr-only", "full"]
Protocol = Literal["two-projection", "single-final-check"]

@dataclass(frozen=True)
class NoiseModel:
    probability: float = 0.0
    preparation: bool = True
    measurement: bool = True
    cnot: bool = True
    idle: bool = True
    rotation: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("noise probability must lie in [0,1]")

    @property
    def active(self) -> bool:
        return self.probability > 0 and any(
            (self.preparation, self.measurement, self.cnot, self.idle, self.rotation)
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitBundle:
    text: str
    mode: Mode
    protocol: Protocol
    theta: float
    theta_star: float
    partitions_per_logical: int
    logicals: tuple[int, ...]
    batch_order: tuple[int, ...]
    detector_groups: dict[str, tuple[int, ...]]
    postselection_detectors: tuple[int, ...]
    measurement_groups: dict[str, tuple[int, ...]]
    operation_counts: dict[str, int]
    artifact: dict[str, str]
    partition_certificate: dict[str, object]
    noise: dict[str, object]

    @property
    def postselection_mask(self) -> list[int]:
        mask = [0] * sum(len(group) for group in self.detector_groups.values())
        for detector in self.postselection_detectors:
            mask[detector] = 1
        return mask

    @property
    def ideal_acceptance(self) -> float:
        return ideal_tmr_acceptance(
            self.theta, self.partitions_per_logical
        ) ** len(self.logicals)


@dataclass
class _TextBuilder:
    lines: list[str] = field(default_factory=list)
    num_measurements: int = 0
    num_detectors: int = 0
    detector_groups: dict[str, list[int]] = field(default_factory=dict)
    measurement_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    operation_counts: dict[str, int] = field(
        default_factory=lambda: {
            "data_preparations": 0,
            "ancilla_preparations": 0,
            "physical_cnots": 0,
            "cnot_layers": 0,
            "physical_rotations": 0,
            "rotation_layers": 0,
            "idle_locations": 0,
            "measurements": 0,
            "ideal_mpp_measurements": 0,
            "virtual_feedback_x": 0,
        }
    )

    def append(self, line: str) -> None:
        if line:
            self.lines.append(line)

    def tick(self) -> None:
        self.lines.append("TICK")

    def measurement(self, gate: str, targets: Sequence[int], group: str) -> tuple[int, ...]:
        targets = tuple(map(int, targets))
        self.lines.append(f"{gate} {' '.join(map(str, targets))}")
        indices = tuple(range(self.num_measurements, self.num_measurements + len(targets)))
        self.num_measurements += len(targets)
        self.operation_counts["measurements"] += len(targets)
        self.measurement_groups[group] = indices
        return indices

    def mpp(self, pauli: str, group: str) -> int:
        self.lines.append(f"MPP {pauli}")
        measurement = self.num_measurements
        self.num_measurements += 1
        self.operation_counts["measurements"] += 1
        self.operation_counts["ideal_mpp_measurements"] += 1
        self.measurement_groups.setdefault(group, tuple())
        self.measurement_groups[group] = (*self.measurement_groups[group], measurement)
        return measurement

    def detector(self, measurements: Iterable[int], group: str) -> int:
        targets = " ".join(
            f"rec[{int(measurement) - self.num_measurements}]"
            for measurement in measurements
        )
        self.lines.append(f"DETECTOR {targets}")
        detector = self.num_detectors
        self.num_detectors += 1
        self.detector_groups.setdefault(group, []).append(detector)
        return detector


def physical_tmr_angle(theta: float, partitions: int) -> float:
    if partitions <= 0:
        raise ValueError("partition count must be positive")
    if partitions == 1:
        return theta
    if partitions % 2 == 0:
        raise ValueError("the first implementation supports odd M only")
    tangent = math.tan(theta / 2)
    signed_target = ((-1) ** (partitions // 2)) * tangent
    signed_root = math.copysign(abs(signed_target) ** (1 / partitions), signed_target)
    return 2 * math.atan(signed_root)


def ideal_tmr_acceptance(theta: float, partitions: int) -> float:
    theta_star = physical_tmr_angle(theta, partitions)
    return math.cos(theta_star / 2) ** (2 * partitions) + math.sin(
        theta_star / 2
    ) ** (2 * partitions)


def _pauli_string(
    axis: str, support: Sequence[int], num_data: int
) -> stim.PauliString:
    characters = ["I"] * num_data
    for qubit in support:
        characters[int(qubit)] = axis
    return stim.PauliString("".join(characters))


def encoded_plus_circuit(code: CodeArtifact) -> stim.Circuit:
    """Return a noiseless Clifford preparation of the fixed logical plus state."""

    x_rows = independent_indices(code.matrix_x)
    z_rows = independent_indices(code.matrix_z)
    stabilizers = [
        *(_pauli_string("X", code.checks_x[row], code.num_data) for row in x_rows),
        *(_pauli_string("Z", code.checks_z[row], code.num_data) for row in z_rows),
        *(_pauli_string("X", support, code.num_data) for support in code.logicals_x),
    ]
    if len(stabilizers) != code.num_data:
        raise ValueError(
            f"encoded plus state needs exactly {code.num_data} independent stabilizers"
        )
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    return tableau.to_circuit("graph_state")


def _append_depolarize1(
    builder: _TextBuilder, qubits: Sequence[int], noise: NoiseModel
) -> None:
    if qubits and noise.active:
        builder.append(
            f"DEPOLARIZE1({noise.probability:.17g}) {' '.join(map(str, qubits))}"
        )


def _append_cnot_layer(
    builder: _TextBuilder,
    pairs: Sequence[tuple[int, int]],
    noise: NoiseModel,
    live_qubits: Sequence[int],
) -> None:
    if not pairs:
        return
    flat = [qubit for pair in pairs for qubit in pair]
    builder.append(f"CX {' '.join(map(str, flat))}")
    builder.operation_counts["physical_cnots"] += len(pairs)
    builder.operation_counts["cnot_layers"] += 1
    if noise.active and noise.cnot:
        builder.append(f"DEPOLARIZE2({noise.probability:.17g}) {' '.join(map(str, flat))}")
    if noise.active and noise.idle:
        used = set(flat)
        idle = tuple(qubit for qubit in live_qubits if qubit not in used)
        _append_depolarize1(builder, idle, noise)
        builder.operation_counts["idle_locations"] += len(idle)
    builder.tick()


def _logical_groups(
    code: CodeArtifact,
    selected: Sequence[int],
    partitions: Mapping[int, Partition],
    count: int,
) -> list[tuple[int, ...]]:
    if count == 1:
        return [
            tuple(np.flatnonzero(code.logical_z[logical]).astype(int).tolist())
            for logical in selected
        ]
    return [tuple(part) for logical in selected for part in partitions[logical]]


def _choose_pivot(code: CodeArtifact, group: Sequence[int]) -> int:
    logical_degree = np.count_nonzero(code.logical_z, axis=0)
    return min(map(int, group), key=lambda qubit: (int(logical_degree[qubit]), qubit))


def _append_direct_tmr(
    builder: _TextBuilder,
    code: CodeArtifact,
    selected: Sequence[int],
    partitions: Mapping[int, Partition],
    partition_count: int,
    theta_star: float,
) -> None:
    alpha = theta_star / math.pi
    for group in _logical_groups(code, selected, partitions, partition_count):
        builder.append(f"R_PAULI({alpha:.17g}) {support_to_pauli('Z', group)}")
        builder.operation_counts["physical_rotations"] += 1
    builder.operation_counts["rotation_layers"] += 1
    builder.tick()


def _append_ladder_tmr(
    builder: _TextBuilder,
    code: CodeArtifact,
    selected: Sequence[int],
    partitions: Mapping[int, Partition],
    partition_count: int,
    theta_star: float,
    noise: NoiseModel,
) -> None:
    groups = _logical_groups(code, selected, partitions, partition_count)
    records = []
    occupied: set[int] = set()
    for group in groups:
        if occupied.intersection(group):
            raise ValueError("one TMR layer contains overlapping partial rotations")
        occupied.update(group)
        pivot = _choose_pivot(code, group)
        controls = tuple(qubit for qubit in group if qubit != pivot)
        records.append((controls, pivot))
    maximum = max((len(controls) for controls, _pivot in records), default=0)
    for layer in range(maximum):
        pairs = [(controls[layer], pivot) for controls, pivot in records if layer < len(controls)]
        _append_cnot_layer(builder, pairs, noise, tuple(range(code.num_data)))

    pivots = tuple(pivot for _controls, pivot in records)
    alpha = theta_star / math.pi
    builder.append(f"R_Z({alpha:.17g}) {' '.join(map(str, pivots))}")
    builder.operation_counts["physical_rotations"] += len(pivots)
    builder.operation_counts["rotation_layers"] += 1
    if noise.active and noise.rotation:
        _append_depolarize1(builder, pivots, noise)
    builder.tick()

    for layer in reversed(range(maximum)):
        pairs = [(controls[layer], pivot) for controls, pivot in records if layer < len(controls)]
        _append_cnot_layer(builder, pairs, noise, tuple(range(code.num_data)))


def _append_projection_detectors(
    builder: _TextBuilder,
    x_measurements: Sequence[int],
    z_measurements: Sequence[int],
    label: str,
) -> None:
    for measurement in x_measurements:
        builder.detector((measurement,), f"{label}_X")
    for measurement in z_measurements:
        builder.detector((measurement,), f"{label}_Z")


def _append_ideal_projection(
    builder: _TextBuilder, code: CodeArtifact, label: str
) -> None:
    x_measurements = [
        builder.mpp(support_to_pauli("X", support), f"{label}_raw_X")
        for support in code.checks_x
    ]
    z_measurements = [
        builder.mpp(support_to_pauli("Z", support), f"{label}_raw_Z")
        for support in code.checks_z
    ]
    _append_projection_detectors(builder, x_measurements, z_measurements, label)
    builder.tick()


def _append_full_syndrome_round(
    builder: _TextBuilder,
    code: CodeArtifact,
    noise: NoiseModel,
    label: str,
) -> None:
    data_qubits = tuple(range(code.num_data))
    x_ancilla_start = code.num_data
    z_ancilla_start = code.num_data + code.num_checks
    x_ancillas = tuple(range(x_ancilla_start, z_ancilla_start))
    z_ancillas = tuple(range(z_ancilla_start, z_ancilla_start + code.num_checks))
    factory_qubits = data_qubits + x_ancillas + z_ancillas
    builder.append(f"RX {' '.join(map(str, x_ancillas))}")
    builder.append(f"R {' '.join(map(str, z_ancillas))}")
    builder.operation_counts["ancilla_preparations"] += 2 * code.num_checks
    if noise.active and noise.preparation:
        _append_depolarize1(builder, x_ancillas + z_ancillas, noise)
    builder.tick()

    for layer in code.schedule:
        pairs = []
        for gate in layer:
            if gate.kind == "X":
                pairs.append((x_ancilla_start + gate.check, gate.data))
            else:
                pairs.append((gate.data, z_ancilla_start + gate.check))
        _append_cnot_layer(builder, pairs, noise, factory_qubits)

    if noise.active and noise.measurement:
        builder.append(
            f"Z_ERROR({noise.probability:.17g}) {' '.join(map(str, x_ancillas))}"
        )
        builder.append(
            f"X_ERROR({noise.probability:.17g}) {' '.join(map(str, z_ancillas))}"
        )
    x_measurements = builder.measurement("MX", x_ancillas, f"{label}_raw_X")
    z_measurements = builder.measurement("M", z_ancillas, f"{label}_raw_Z")
    _append_projection_detectors(builder, x_measurements, z_measurements, label)
    builder.tick()


def _append_initialization(
    builder: _TextBuilder, code: CodeArtifact, noise: NoiseModel
) -> None:
    data_qubits = tuple(range(code.num_data))
    z_ancilla_start = code.num_data + code.num_checks
    z_ancillas = tuple(range(z_ancilla_start, z_ancilla_start + code.num_checks))
    builder.append(f"RX {' '.join(map(str, data_qubits))}")
    builder.operation_counts["data_preparations"] += code.num_data
    if noise.active and noise.preparation:
        _append_depolarize1(builder, data_qubits, noise)
    builder.tick()

    builder.append(f"R {' '.join(map(str, z_ancillas))}")
    builder.operation_counts["ancilla_preparations"] += code.num_checks
    if noise.active and noise.preparation:
        _append_depolarize1(builder, z_ancillas, noise)
    builder.tick()
    live = data_qubits + z_ancillas
    for layer in code.schedule:
        pairs = [
            (gate.data, z_ancilla_start + gate.check)
            for gate in layer
            if gate.kind == "Z"
        ]
        _append_cnot_layer(builder, pairs, noise, live)
    if noise.active and noise.measurement:
        builder.append(
            f"X_ERROR({noise.probability:.17g}) {' '.join(map(str, z_ancillas))}"
        )
    measurements = builder.measurement("M", z_ancillas, "initialization_raw_Z")

    for relation in row_relations(code.matrix_z):
        builder.detector((measurements[check] for check in relation), "initialization_relations")

    independent_rows, _correction_qubits, right_inverse = initialization_correction_map(
        code.matrix_z
    )
    for local_syndrome, check in enumerate(independent_rows):
        targets = np.flatnonzero(right_inverse[:, local_syndrome]).astype(int).tolist()
        if not targets:
            continue
        record = f"rec[{measurements[check] - builder.num_measurements}]"
        pairs = " ".join(f"{record} {qubit}" for qubit in targets)
        builder.append(f"CX {pairs}")
        builder.operation_counts["virtual_feedback_x"] += len(targets)
    builder.tick()


def _selected_batches(
    code: CodeArtifact, logical_count: int, batch_order: tuple[int, ...]
) -> list[tuple[int, ...]]:
    if logical_count not in range(1, len(code.batches[0]) + 1):
        raise ValueError(f"logical_count must lie in 1..{len(code.batches[0])}")
    return [tuple(code.batches[batch_order[0]][:logical_count])]


def build_circuit(
    *,
    code: CodeArtifact,
    partition_certificate: PartitionCertificate,
    mode: Mode,
    protocol: Protocol,
    theta: float,
    partition_count: int = 3,
    logical_count: int = 4,
    batch_order: tuple[int, ...] = (0,),
    noise: NoiseModel | None = None,
) -> CircuitBundle:
    if mode not in ("ideal-projection", "scheduled-tmr-only", "full"):
        raise ValueError(f"unknown mode: {mode}")
    if protocol not in ("two-projection", "single-final-check"):
        raise ValueError(f"unknown protocol: {protocol}")
    if batch_order != (0,):
        raise ValueError("the single disjoint batch has batch_order=(0,)")
    if partition_count not in (1, 3):
        raise ValueError("the first implementation supports M=1 or M=3")
    if protocol == "single-final-check" and logical_count > 1:
        if not partition_certificate.single_final_check_certified:
            raise ValueError("combined TMR kernel is not certified")
    noise = NoiseModel() if noise is None else noise
    if mode == "ideal-projection" and noise.active:
        raise ValueError("ideal projection mode cannot contain physical noise")

    certificate = certify_partitions(code, partition_certificate.partitions)
    selected_batches = _selected_batches(code, logical_count, batch_order)
    selected_logicals = tuple(logical for batch in selected_batches for logical in batch)
    theta_star = physical_tmr_angle(theta, partition_count)
    builder = _TextBuilder()

    for qubit in range(code.num_data):
        half, within = divmod(qubit, code.num_data // 2)
        builder.append(
            f"QUBIT_COORDS({half}, {within}) {qubit}"
        )
    if mode == "full":
        _append_initialization(builder, code, noise)
    else:
        builder.lines.extend(str(encoded_plus_circuit(code)).strip().splitlines())
        builder.tick()

    for batch_index, selected in enumerate(selected_batches):
        if mode == "ideal-projection":
            _append_direct_tmr(
                builder,
                code,
                selected,
                certificate.partitions,
                partition_count,
                theta_star,
            )
        else:
            _append_ladder_tmr(
                builder,
                code,
                selected,
                certificate.partitions,
                partition_count,
                theta_star,
                noise,
            )
        project_now = protocol == "two-projection" or batch_index == len(selected_batches) - 1
        if project_now:
            label = (
                "post_all"
                if protocol == "single-final-check" and len(selected_batches) > 1
                else f"post_batch_{batch_order[batch_index]}"
            )
            if mode == "ideal-projection":
                _append_ideal_projection(builder, code, label)
            else:
                _append_full_syndrome_round(builder, code, noise, label)

    postselection = tuple(
        detector
        for group, detectors in builder.detector_groups.items()
        if group.startswith("post_")
        for detector in detectors
    )
    frozen_groups = {
        group: tuple(detectors) for group, detectors in builder.detector_groups.items()
    }
    return CircuitBundle(
        text="\n".join(builder.lines) + "\n",
        mode=mode,
        protocol=protocol,
        theta=float(theta),
        theta_star=float(theta_star),
        partitions_per_logical=partition_count,
        logicals=selected_logicals,
        batch_order=batch_order,
        detector_groups=frozen_groups,
        postselection_detectors=postselection,
        measurement_groups=dict(builder.measurement_groups),
        operation_counts=dict(builder.operation_counts),
        artifact={
            "basis_path": str(code.basis_path),
            "schedule_path": str(code.schedule_path),
            "basis_sha256": code.basis_sha256,
            "schedule_sha256": code.schedule_sha256,
        },
        partition_certificate=certificate.to_json(),
        noise=noise.to_json(),
    )
