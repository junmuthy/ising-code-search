"""Noiseless branch-conditioned circuits for hybrid BB64 repair validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table
from bb64_tmr_postselection.circuit import encoded_plus_circuit, physical_tmr_angle
from bb64_tmr_postselection.model import support_to_pauli

from .model import HybridModel


CONSTANT_QUBIT = 64


@dataclass(frozen=True)
class CircuitBundle:
    text: str
    theta: float
    syndrome_class_id: int | None
    alternative_mask: tuple[int, ...]
    postselection_mask: tuple[int, ...]
    initial_detector_count: int
    final_detector_count: int
    observable_count: int
    expected_branch_probability: float
    operation_counts: dict[str, int]


class _Builder:
    def __init__(self, prefix: str):
        self.lines = prefix.strip().splitlines()
        self.measurements = 0
        self.detectors = 0
        self.postselection: list[int] = []

    def append(self, instruction: str) -> None:
        self.lines.append(instruction)

    def prepare_constant_one(self) -> None:
        self.lines.extend((f"R {CONSTANT_QUBIT}", f"X {CONSTANT_QUBIT}", f"M {CONSTANT_QUBIT}"))
        self.measurements += 1

    def mpp_detector(self, pauli: str, expected: int, *, postselect: bool) -> None:
        self.lines.append(f"MPP {pauli}")
        self.measurements += 1
        references = "rec[-1]"
        if expected:
            references += f" rec[-{self.measurements}]"
        self.lines.append(f"DETECTOR {references}")
        if postselect:
            self.postselection.append(self.detectors)
        self.detectors += 1

    def mpp_observable(self, pauli: str, observable: int) -> None:
        self.lines.append(f"MPP {pauli}")
        self.measurements += 1
        self.lines.append(f"OBSERVABLE_INCLUDE({observable}) rec[-1]")

    def finish(self) -> str:
        return "\n".join(self.lines) + "\n"


def _append_rotation(builder: _Builder, angle: float, support: Sequence[int]) -> None:
    builder.append(f"R_PAULI({angle / math.pi:.17g}) {support_to_pauli('Z', support)}")


def _append_check_measurements(
    builder: _Builder,
    model: HybridModel,
    *,
    syndrome_x: Sequence[int],
    syndrome_z: Sequence[int],
    postselect: bool,
) -> None:
    for expected, support in zip(syndrome_x, model.code.checks_x, strict=True):
        builder.mpp_detector(support_to_pauli("X", support), int(expected), postselect=postselect)
    for expected, support in zip(syndrome_z, model.code.checks_z, strict=True):
        builder.mpp_detector(support_to_pauli("Z", support), int(expected), postselect=postselect)


def _append_logical_x_observables(builder: _Builder, model: HybridModel) -> None:
    for logical, support in enumerate(model.code.logicals_x):
        builder.mpp_observable(support_to_pauli("X", support), logical)


def class_id_for_labels(model: HybridModel, labels: Iterable[int]) -> int:
    labels_array = np.asarray(tuple(labels), dtype=np.uint8)
    if labels_array.shape != (8,) or np.any(labels_array > 3):
        raise ValueError("local branch labels must contain eight values in 0..3")
    base4 = int(np.dot(labels_array.astype(np.uint32), 4 ** np.arange(8, dtype=np.uint32)))
    matches = np.flatnonzero(model.recovery.base4_branch_id == base4)
    if len(matches) != 1:
        raise ValueError("local branch labels do not identify one syndrome class")
    return int(matches[0])


def build_synthetic_repair_circuit(
    model: HybridModel,
    *,
    theta: float,
    alternative_mask: Sequence[int],
) -> CircuitBundle:
    """Prepare known branch angles, repair them, and cancel the target angle."""

    mask = np.asarray(alternative_mask, dtype=np.uint8)
    if mask.shape != (8,) or np.any(mask > 1):
        raise ValueError("alternative mask must be eight binary values")
    table = angle_table(theta)
    target_angle = float(table["target"]["logical_angle"])
    alternative_angle = float(table["alternative"]["logical_angle"])
    builder = _Builder(str(encoded_plus_circuit(model.code)))
    counts = {"branch_rotations": 0, "repair_rotations": 0, "inverse_rotations": 0}
    for logical, support in enumerate(model.code.logicals_z):
        angle = alternative_angle if mask[logical] else target_angle
        _append_rotation(builder, angle, support)
        counts["branch_rotations"] += 1
    for logical in np.flatnonzero(mask):
        _append_rotation(
            builder,
            theta - alternative_angle,
            model.code.logicals_z[int(logical)],
        )
        counts["repair_rotations"] += 1
    for support in model.code.logicals_z:
        _append_rotation(builder, -theta, support)
        counts["inverse_rotations"] += 1
    _append_logical_x_observables(builder, model)
    return CircuitBundle(
        text=builder.finish(),
        theta=float(theta),
        syndrome_class_id=None,
        alternative_mask=tuple(map(int, mask)),
        postselection_mask=(),
        initial_detector_count=0,
        final_detector_count=0,
        observable_count=8,
        expected_branch_probability=1.0,
        operation_counts=counts,
    )


def build_ideal_branch_repair_circuit(
    model: HybridModel,
    *,
    theta: float,
    syndrome_class_id: int,
) -> CircuitBundle:
    """Postselect an exact TMR branch, correct, repair, and cancel its target."""

    recovery = model.recovery
    class_id = int(syndrome_class_id)
    if not 0 <= class_id < len(recovery.syndrome_coordinates):
        raise ValueError("syndrome class ID is out of range")
    mask = recovery.alternative_mask[class_id]
    table = angle_table(theta)
    target_probability = float(table["probability_target"])
    alternative_pattern_probability = float(table["alternative"]["probability_per_pattern"])
    bad_count = int(np.count_nonzero(mask))
    expected_probability = target_probability ** (8 - bad_count) * alternative_pattern_probability**bad_count

    builder = _Builder(str(encoded_plus_circuit(model.code)))
    builder.prepare_constant_one()
    counts = {
        "tmr_rotations": 0,
        "pauli_correction_weight": int(recovery.correction_weight[class_id]),
        "repair_rotations": bad_count,
        "inverse_rotations": 8,
    }
    alpha = physical_tmr_angle(theta, 3)
    for support in recovery.piece_supports:
        _append_rotation(builder, alpha, np.flatnonzero(support).astype(int).tolist())
        counts["tmr_rotations"] += 1
    builder.append("TICK")

    syndrome = recovery.displayed_syndromes[class_id]
    _append_check_measurements(
        builder,
        model,
        syndrome_x=syndrome,
        syndrome_z=np.zeros(32, dtype=np.uint8),
        postselect=True,
    )
    initial_detectors = builder.detectors
    correction = np.flatnonzero(recovery.correction_physical_supports[class_id]).astype(int)
    if len(correction):
        builder.append("Z " + " ".join(map(str, correction)))
    alternative_angle = float(table["alternative"]["logical_angle"])
    for logical in np.flatnonzero(mask):
        _append_rotation(
            builder,
            theta - alternative_angle,
            model.code.logicals_z[int(logical)],
        )
    for support in model.code.logicals_z:
        _append_rotation(builder, -theta, support)

    _append_check_measurements(
        builder,
        model,
        syndrome_x=np.zeros(32, dtype=np.uint8),
        syndrome_z=np.zeros(32, dtype=np.uint8),
        postselect=False,
    )
    final_detectors = builder.detectors - initial_detectors
    _append_logical_x_observables(builder, model)
    return CircuitBundle(
        text=builder.finish(),
        theta=float(theta),
        syndrome_class_id=class_id,
        alternative_mask=tuple(map(int, mask)),
        postselection_mask=tuple(1 if index in builder.postselection else 0 for index in range(builder.detectors)),
        initial_detector_count=initial_detectors,
        final_detector_count=final_detectors,
        observable_count=8,
        expected_branch_probability=expected_probability,
        operation_counts=counts,
    )
