"""Full scheduled BB64 hybrid circuit conditioned on one raw syndrome class."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table
from bb64_tmr_postselection.circuit import (
    NoiseModel,
    _TextBuilder,
    _append_full_syndrome_round,
    _append_ladder_tmr,
    build_circuit as build_m3_circuit,
)
from bb64_tmr_postselection.model import BATCHES, support_to_pauli
from bb64_tmr_postselection.partitions import certify_partitions

from .model import HybridModel


@dataclass(frozen=True)
class RawBranchCalibrationCircuit:
    text: str
    theta: float
    syndrome_class_id: int
    alternative_mask: tuple[int, ...]
    postselection_mask: tuple[int, ...]
    expected_detectors: tuple[int, ...]
    prefix_detector_count: int
    operation_counts: dict[str, int]
    expected_ideal_branch_probability: float


def build_raw_branch_calibration_circuit(
    model: HybridModel,
    *,
    theta: float,
    syndrome_class_id: int,
    noise: NoiseModel,
    final_syndrome_rounds: int = 1,
) -> RawBranchCalibrationCircuit:
    """Build a static continuation for one exact first-round syndrome.

    This is an exact simulation of a controller that trusts one raw syndrome
    round.  It is a stepping stone, not the final repeated-syndrome decoder.
    """

    class_id = int(syndrome_class_id)
    recovery = model.recovery
    if not 0 <= class_id < len(recovery.syndrome_coordinates):
        raise ValueError("syndrome class ID is out of range")
    if final_syndrome_rounds <= 0:
        raise ValueError("at least one final syndrome round is required")
    certificate = certify_partitions(model.code, recovery.partitions)
    prefix = build_m3_circuit(
        code=model.code,
        partition_certificate=certificate,
        mode="full",
        protocol="single-final-check",
        theta=theta,
        partition_count=3,
        logical_count=8,
        noise=noise,
    )
    builder = _TextBuilder(
        lines=prefix.text.rstrip().splitlines(),
        num_measurements=1
        + max(measurement for group in prefix.measurement_groups.values() for measurement in group),
        num_detectors=sum(len(group) for group in prefix.detector_groups.values()),
        detector_groups={key: list(value) for key, value in prefix.detector_groups.items()},
        measurement_groups=dict(prefix.measurement_groups),
        operation_counts=dict(prefix.operation_counts),
    )
    prefix_detector_count = builder.num_detectors

    correction = np.flatnonzero(recovery.correction_physical_supports[class_id]).astype(int)
    if len(correction):
        builder.append("Z " + " ".join(map(str, correction)))
    mask = recovery.alternative_mask[class_id]
    alternative_angle = float(angle_table(theta)["alternative"]["logical_angle"])
    residual = theta - alternative_angle
    for batch in BATCHES:
        selected = tuple(logical for logical in batch if mask[logical])
        if selected:
            _append_ladder_tmr(
                builder,
                model.code,
                selected,
                recovery.partitions,
                1,
                residual,
                noise,
            )
    for round_index in range(final_syndrome_rounds):
        _append_full_syndrome_round(
            builder,
            model.code,
            noise,
            f"recovery_final_{round_index}",
        )

    # Ideal diagnostic inverse and logical-X readout.
    for support in model.code.logicals_z:
        builder.append(f"R_PAULI({-theta / math.pi:.17g}) {support_to_pauli('Z', support)}")
    builder.tick()
    for logical, support in enumerate(model.code.logicals_x):
        measurement = builder.mpp(support_to_pauli("X", support), "logical_X")
        builder.append(
            f"OBSERVABLE_INCLUDE({logical}) rec[{measurement - builder.num_measurements}]"
        )

    expected = np.zeros(builder.num_detectors, dtype=np.uint8)
    expected[list(prefix.detector_groups["post_all_X"])] = recovery.displayed_syndromes[class_id]
    postselection = np.zeros(builder.num_detectors, dtype=np.uint8)
    for group, detectors in builder.detector_groups.items():
        if group.startswith("post_all_") or group.startswith("recovery_final_"):
            postselection[list(detectors)] = 1

    table = angle_table(theta)
    bad = int(recovery.alternative_count[class_id])
    probability = float(table["probability_target"]) ** (8 - bad) * float(
        table["alternative"]["probability_per_pattern"]
    ) ** bad
    counts = dict(builder.operation_counts)
    counts["virtual_branch_correction_weight"] = int(recovery.correction_weight[class_id])
    counts["ideal_inverse_rotations"] = 8
    counts["repaired_logicals"] = bad
    return RawBranchCalibrationCircuit(
        text="\n".join(builder.lines) + "\n",
        theta=float(theta),
        syndrome_class_id=class_id,
        alternative_mask=tuple(map(int, mask)),
        postselection_mask=tuple(map(int, postselection)),
        expected_detectors=tuple(map(int, expected)),
        prefix_detector_count=prefix_detector_count,
        operation_counts=counts,
        expected_ideal_branch_probability=probability,
    )
