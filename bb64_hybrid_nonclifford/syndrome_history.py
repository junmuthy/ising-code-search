"""Scheduled BB64 M=3 circuits that retain repeated syndrome histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bb64_tmr_postselection.circuit import (
    NoiseModel,
    _TextBuilder,
    _append_full_syndrome_round,
    build_circuit as build_m3_circuit,
)
from bb64_tmr_postselection.partitions import certify_partitions

from .model import HybridModel


@dataclass(frozen=True)
class SyndromeHistoryCircuit:
    text: str
    theta: float
    syndrome_rounds: int
    x_detector_groups: tuple[tuple[int, ...], ...]
    z_detector_groups: tuple[tuple[int, ...], ...]
    operation_counts: dict[str, int]
    noise: dict[str, object]


def build_syndrome_history_circuit(
    model: HybridModel,
    *,
    theta: float,
    noise: NoiseModel,
    syndrome_rounds: int,
) -> SyndromeHistoryCircuit:
    """Build full M=3 preparation followed by retained syndrome rounds."""

    if syndrome_rounds <= 0:
        raise ValueError("syndrome_rounds must be positive")
    certificate = certify_partitions(model.code, model.recovery.partitions)
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
        + max(
            measurement
            for group in prefix.measurement_groups.values()
            for measurement in group
        ),
        num_detectors=sum(len(group) for group in prefix.detector_groups.values()),
        detector_groups={key: list(value) for key, value in prefix.detector_groups.items()},
        measurement_groups=dict(prefix.measurement_groups),
        operation_counts=dict(prefix.operation_counts),
    )
    labels = ["post_all"]
    for round_index in range(1, syndrome_rounds):
        label = f"history_{round_index}"
        _append_full_syndrome_round(builder, model.code, noise, label)
        labels.append(label)
    x_groups = tuple(tuple(builder.detector_groups[f"{label}_X"]) for label in labels)
    z_groups = tuple(tuple(builder.detector_groups[f"{label}_Z"]) for label in labels)
    if any(len(group) != 32 for group in (*x_groups, *z_groups)):
        raise ValueError("every retained syndrome group must contain 32 displayed checks")
    return SyndromeHistoryCircuit(
        text="\n".join(builder.lines) + "\n",
        theta=float(theta),
        syndrome_rounds=int(syndrome_rounds),
        x_detector_groups=x_groups,
        z_detector_groups=z_groups,
        operation_counts=dict(builder.operation_counts),
        noise=noise.to_json(),
    )


def extract_syndrome_histories(
    detectors: np.ndarray,
    circuit: SyndromeHistoryCircuit,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract `(shots, rounds, 32)` displayed X and Z arrays."""

    values = np.asarray(detectors, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("detectors must have shape (shots, detectors)")
    highest = max(index for group in (*circuit.x_detector_groups, *circuit.z_detector_groups) for index in group)
    if values.shape[1] <= highest:
        raise ValueError("detector array is narrower than the retained syndrome groups")
    x_history = np.stack([values[:, group] for group in circuit.x_detector_groups], axis=1)
    z_history = np.stack([values[:, group] for group in circuit.z_detector_groups], axis=1)
    return x_history, z_history
