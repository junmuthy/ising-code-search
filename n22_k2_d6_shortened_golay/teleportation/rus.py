"""Classical repeat-until-success model used after circuit-level calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def expected_parallel_levels(logical_count: int, *, maximum_levels: int | None = None) -> float:
    """Expected maximum of ``logical_count`` geometric(1/2) ladders."""

    if logical_count <= 0:
        raise ValueError("logical_count must be positive")
    limit = maximum_levels if maximum_levels is not None else 80
    return sum(1 - (1 - 2.0 ** (-round_index)) ** logical_count for round_index in range(limit))


def fold_angle(theta: float, *, tolerance: float = 1e-12) -> float:
    """Fold an RZ angle modulo the free logical Pauli RZ(pi)."""

    folded = (float(theta) + math.pi / 2) % math.pi - math.pi / 2
    return 0.0 if abs(folded) < tolerance else abs(folded)


@dataclass(frozen=True)
class InjectionPoint:
    theta: float
    logical_count: int
    partition_count: int
    acceptance: float
    error_pattern_probabilities: tuple[float, ...]
    syndrome_cycles_per_attempt: float = 2.0

    def __post_init__(self) -> None:
        if self.logical_count <= 0:
            raise ValueError("logical_count must be positive")
        if not 0 < self.acceptance <= 1:
            raise ValueError("acceptance must lie in (0,1]")
        if len(self.error_pattern_probabilities) != 1 << self.logical_count:
            raise ValueError("one probability is required for each logical error pattern")
        if any(value < 0 for value in self.error_pattern_probabilities):
            raise ValueError("error probabilities must be nonnegative")
        if not math.isclose(sum(self.error_pattern_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("error-pattern probabilities must sum to one")


class InjectionTable:
    def __init__(self, points: list[InjectionPoint], *, angle_tolerance: float = 1e-9) -> None:
        if not points:
            raise ValueError("the injection table is empty")
        self.points = list(points)
        self.angle_tolerance = float(angle_tolerance)

    @classmethod
    def from_json(cls, path: Path) -> "InjectionTable":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        points = [
            InjectionPoint(
                theta=float(point["theta"]),
                logical_count=int(point["logical_count"]),
                partition_count=int(point["partition_count"]),
                acceptance=float(point["acceptance"]),
                error_pattern_probabilities=tuple(
                    map(float, point["error_pattern_probabilities"])
                ),
                syndrome_cycles_per_attempt=float(
                    point.get("syndrome_cycles_per_attempt", 2.0)
                ),
            )
            for point in record["points"]
        ]
        return cls(points, angle_tolerance=float(record.get("angle_tolerance", 1e-9)))

    def resolve(self, theta: float, logical_count: int) -> InjectionPoint:
        candidates = [
            point for point in self.points if point.logical_count == logical_count
        ]
        if not candidates:
            raise KeyError(f"no injection point for N={logical_count}")
        distance, point = min(
            ((abs(point.theta - theta), point) for point in candidates),
            key=lambda item: item[0],
        )
        if distance > self.angle_tolerance:
            raise KeyError(
                f"no calibrated N={logical_count} point at theta={theta:.12g}; "
                f"closest is {point.theta:.12g}"
            )
        return point


def _sample_pattern(probabilities: tuple[float, ...], rng: np.random.Generator) -> int:
    return int(rng.choice(len(probabilities), p=np.asarray(probabilities)))


def simulate_rus(
    *,
    theta: float,
    logical_count: int,
    shots: int,
    table: InjectionTable,
    seed: int,
    maximum_levels: int = 20,
) -> dict[str, Any]:
    """Simulate independent sign ladders with calibrated block preparation.

    A preparation failure repeats the same block-level attempt.  Once an
    accepted resource is available, each active logical terminates with
    probability one half; failures advance to twice the angle.  Logical error
    bits accumulate modulo two across all consumed resource states.
    """

    if theta <= 0 or logical_count <= 0 or shots <= 0:
        raise ValueError("theta, logical_count, and shots must be positive")
    rng = np.random.default_rng(seed)
    round_histogram: dict[int, int] = {}
    attempt_histogram: dict[int, int] = {}
    final_error_patterns = np.zeros(1 << logical_count, dtype=np.int64)
    total_accepted_blocks = 0
    total_raw_attempts = 0
    total_syndrome_cycles = 0.0
    total_teleportations = 0
    truncated = 0

    for _ in range(shots):
        active = list(range(logical_count))
        errors = np.zeros(logical_count, dtype=np.uint8)
        raw_attempts_this_shot = 0
        accepted_blocks_this_shot = 0
        levels_used = 0
        for level in range(maximum_levels):
            effective_angle = fold_angle((2**level) * theta)
            if not active or effective_angle == 0:
                break
            point = table.resolve(effective_angle, len(active))
            raw_attempts = int(rng.geometric(point.acceptance))
            raw_attempts_this_shot += raw_attempts
            total_syndrome_cycles += raw_attempts * point.syndrome_cycles_per_attempt
            accepted_blocks_this_shot += 1
            levels_used += 1

            pattern = _sample_pattern(point.error_pattern_probabilities, rng)
            survivors: list[int] = []
            for local, logical in enumerate(active):
                errors[logical] ^= (pattern >> local) & 1
                total_teleportations += 1
                if rng.integers(2):
                    survivors.append(logical)
            active = survivors
        if active and fold_angle((2**maximum_levels) * theta) != 0:
            truncated += 1
        pattern = sum(int(bit) << logical for logical, bit in enumerate(errors))
        final_error_patterns[pattern] += 1
        round_histogram[levels_used] = round_histogram.get(levels_used, 0) + 1
        attempt_histogram[raw_attempts_this_shot] = (
            attempt_histogram.get(raw_attempts_this_shot, 0) + 1
        )
        total_accepted_blocks += accepted_blocks_this_shot
        total_raw_attempts += raw_attempts_this_shot

    per_logical_errors = [
        sum(count for pattern, count in enumerate(final_error_patterns) if pattern >> logical & 1)
        for logical in range(logical_count)
    ]
    return {
        "shots": shots,
        "theta": theta,
        "logical_count": logical_count,
        "maximum_levels": maximum_levels,
        "truncated": truncated,
        "mean_parallel_levels": total_accepted_blocks / shots,
        "mean_raw_preparation_attempts": total_raw_attempts / shots,
        "mean_syndrome_cycles": total_syndrome_cycles / shots,
        "mean_teleportations": total_teleportations / shots,
        "per_logical_error_rates": [count / shots for count in per_logical_errors],
        "any_logical_error_rate": 1 - final_error_patterns[0] / shots,
        "final_error_pattern_counts": final_error_patterns.tolist(),
        "round_histogram": {str(key): value for key, value in sorted(round_histogram.items())},
        "raw_attempt_histogram": {
            str(key): value for key, value in sorted(attempt_histogram.items())
        },
    }
