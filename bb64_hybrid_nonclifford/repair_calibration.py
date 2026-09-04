"""Scheduled noisy M=1 repair component for the BB64 hybrid protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table
from bb64_tmr_postselection.circuit import (
    NoiseModel,
    _TextBuilder,
    _append_full_syndrome_round,
    _append_ladder_tmr,
    encoded_plus_circuit,
)
from bb64_tmr_postselection.model import BATCHES, support_to_pauli

from .model import HybridModel


@dataclass(frozen=True)
class RepairCalibrationCircuit:
    text: str
    theta: float
    alternative_mask: tuple[int, ...]
    selected_batches: tuple[tuple[int, ...], ...]
    final_syndrome_rounds: int
    postselection_mask: tuple[int, ...]
    observable_count: int
    operation_counts: dict[str, int]
    noise: dict[str, object]


def build_repair_calibration_circuit(
    model: HybridModel,
    *,
    theta: float,
    alternative_mask: Sequence[int],
    noise: NoiseModel,
    final_syndrome_rounds: int = 1,
) -> RepairCalibrationCircuit:
    """Prepare an ideal branch and simulate its scheduled noisy M=1 repair.

    The final syndrome is postselected in this component calibration.  The
    later full protocol replaces that postselection with the spacetime decoder.
    """

    mask = np.asarray(alternative_mask, dtype=np.uint8)
    if mask.shape != (8,) or np.any(mask > 1):
        raise ValueError("alternative mask must be eight binary values")
    if final_syndrome_rounds <= 0:
        raise ValueError("at least one final syndrome round is required")

    table = angle_table(theta)
    target = float(table["target"]["logical_angle"])
    alternative = float(table["alternative"]["logical_angle"])
    residual = theta - alternative
    builder = _TextBuilder()
    builder.lines.extend(str(encoded_plus_circuit(model.code)).strip().splitlines())
    builder.tick()

    # This is an ideal input fixture representing the branch after its
    # syndrome-dependent Pauli correction.  Noise is enabled only on the
    # scheduled repair and final extraction below.
    for logical, support in enumerate(model.code.logicals_z):
        angle = alternative if mask[logical] else target
        builder.append(f"R_PAULI({angle / math.pi:.17g}) {support_to_pauli('Z', support)}")
    builder.tick()

    selected_batches: list[tuple[int, ...]] = []
    for batch in BATCHES:
        selected = tuple(logical for logical in batch if mask[logical])
        if not selected:
            continue
        selected_batches.append(selected)
        _append_ladder_tmr(
            builder,
            model.code,
            selected,
            model.recovery.partitions,
            1,
            residual,
            noise,
        )

    for round_index in range(final_syndrome_rounds):
        _append_full_syndrome_round(
            builder,
            model.code,
            noise,
            f"final_round_{round_index}",
        )

    # Ideal inverse target rotations turn perfect repaired resources back into
    # logical |+> states.  These diagnostic operations are deliberately noise
    # free and are not included in physical protocol resource counts.
    for support in model.code.logicals_z:
        builder.append(f"R_PAULI({-theta / math.pi:.17g}) {support_to_pauli('Z', support)}")
    builder.tick()
    for logical, support in enumerate(model.code.logicals_x):
        measurement = builder.mpp(support_to_pauli("X", support), "logical_X")
        builder.append(
            f"OBSERVABLE_INCLUDE({logical}) rec[{measurement - builder.num_measurements}]"
        )

    postselection_detectors = tuple(
        detector
        for group, detectors in builder.detector_groups.items()
        if group.startswith("final_round_")
        for detector in detectors
    )
    postselection_mask = tuple(
        1 if detector in postselection_detectors else 0
        for detector in range(builder.num_detectors)
    )
    counts = dict(builder.operation_counts)
    counts["ideal_branch_rotations"] = 8
    counts["ideal_inverse_rotations"] = 8
    counts["repaired_logicals"] = int(np.count_nonzero(mask))
    return RepairCalibrationCircuit(
        text="\n".join(builder.lines) + "\n",
        theta=float(theta),
        alternative_mask=tuple(map(int, mask)),
        selected_batches=tuple(selected_batches),
        final_syndrome_rounds=int(final_syndrome_rounds),
        postselection_mask=postselection_mask,
        observable_count=8,
        operation_counts=counts,
        noise=noise.to_json(),
    )
