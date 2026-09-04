"""Noiseless adaptive policies for the syndrome-conditioned BB64 factory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np

from bb64_tmr_postselection.model import BATCHES

from .algebra import NUM_LOGICALS, angle_table


INITIAL_M3_CNOTS = 848
INITIAL_M3_CNOT_LAYERS = 24
INITIAL_M3_ROTATIONS = 24
INITIAL_M3_SYNDROME_CYCLES = 2  # initialization Z cycle plus final X/Z cycle
CORRECTION_SYNDROME_CNOTS = 512
CORRECTION_SYNDROME_LAYERS = 8
M1_CNOTS_PER_LOGICAL = 14
M1_LAYERS_PER_NONEMPTY_BATCH = 14


@dataclass(frozen=True)
class PolicyResult:
    theta: float
    threshold: int
    policy: str
    keep_probability_per_attempt: float
    reset_probability_per_attempt: float
    immediate_success_probability_per_attempt: float
    correction_probability_given_kept: float
    expected_initial_attempts: float
    expected_resets: float
    expected_bad_logicals_given_kept: float
    expected_nonempty_batches_given_kept: float
    expected_nonempty_batches_given_correction: float
    expected_m3_stages: float
    expected_m1_stages: float
    expected_total_tmr_stages: float
    expected_physical_cnots: float
    expected_cnot_layers: float
    expected_physical_rotations: float
    expected_syndrome_cycles: float
    residual_correction_angle: float
    final_angle_error: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _probability_of_bad_masks(theta: float) -> tuple[np.ndarray, np.ndarray]:
    table = angle_table(theta)
    good = float(table["probability_target"])
    bad = float(table["probability_any_alternative"])
    masks = np.arange(1 << NUM_LOGICALS, dtype=np.uint16)
    bits = ((masks[:, None] >> np.arange(NUM_LOGICALS)) & 1).astype(np.uint8)
    counts = np.count_nonzero(bits, axis=1)
    probabilities = good ** (NUM_LOGICALS - counts) * bad**counts
    probabilities /= np.sum(probabilities)
    return bits, probabilities


def _nonempty_batch_count(bits: np.ndarray) -> np.ndarray:
    result = np.zeros(len(bits), dtype=np.uint8)
    for batch in BATCHES:
        result += np.any(bits[:, list(batch)], axis=1)
    return result


def evaluate_policy(theta: float, threshold: int) -> PolicyResult:
    """Evaluate reset-above-threshold plus deterministic M=1 repair.

    A retained nonzero branch is repaired in one adaptive logical ``M=1``
    stage.  The M=1 rotation is an in-code logical rotation, so its ideal
    branch is deterministic.  Resource counts conservatively include one
    full final syndrome round after a repair stage.
    """

    if not 0 <= threshold <= NUM_LOGICALS:
        raise ValueError("threshold must lie from zero through eight")
    bits, probabilities = _probability_of_bad_masks(theta)
    bad_count = np.count_nonzero(bits, axis=1)
    batches = _nonempty_batch_count(bits)
    kept = bad_count <= threshold
    kept_probability = float(np.sum(probabilities[kept]))
    if kept_probability <= 0:
        raise ValueError("policy has zero keep probability")
    conditional = probabilities[kept] / kept_probability
    kept_bad = bad_count[kept]
    kept_batches = batches[kept]
    correction = kept_bad > 0

    immediate = float(probabilities[0])
    correction_probability = float(np.sum(conditional[correction]))
    expected_bad = float(np.dot(conditional, kept_bad))
    expected_batches = float(np.dot(conditional, kept_batches))
    expected_batches_correction = (
        float(np.dot(conditional[correction], kept_batches[correction]) / correction_probability)
        if correction_probability
        else 0.0
    )
    expected_attempts = 1.0 / kept_probability
    expected_resets = expected_attempts - 1.0

    table = angle_table(theta)
    alternative_angle = float(table["alternative"]["logical_angle"])
    residual_angle = float(theta - alternative_angle)
    final_error = abs((alternative_angle + residual_angle) - theta)

    expected_cnots = expected_attempts * INITIAL_M3_CNOTS
    expected_layers = expected_attempts * INITIAL_M3_CNOT_LAYERS
    expected_rotations = expected_attempts * INITIAL_M3_ROTATIONS
    expected_syndromes = expected_attempts * INITIAL_M3_SYNDROME_CYCLES
    if correction_probability:
        expected_cnots += CORRECTION_SYNDROME_CNOTS * correction_probability
        expected_cnots += M1_CNOTS_PER_LOGICAL * expected_bad
        expected_layers += CORRECTION_SYNDROME_LAYERS * correction_probability
        expected_layers += M1_LAYERS_PER_NONEMPTY_BATCH * expected_batches
        expected_rotations += expected_bad
        expected_syndromes += correction_probability

    names = {
        0: "zero-syndrome STAR post-selection",
        1: "repair at most one alternative logical",
        2: "repair at most two alternative logicals",
        3: "repair at most three alternative logicals",
        8: "fully adaptive all-syndrome recovery",
    }
    return PolicyResult(
        theta=float(theta),
        threshold=int(threshold),
        policy=names.get(threshold, f"repair at most {threshold} alternative logicals"),
        keep_probability_per_attempt=kept_probability,
        reset_probability_per_attempt=1.0 - kept_probability,
        immediate_success_probability_per_attempt=immediate,
        correction_probability_given_kept=correction_probability,
        expected_initial_attempts=expected_attempts,
        expected_resets=expected_resets,
        expected_bad_logicals_given_kept=expected_bad,
        expected_nonempty_batches_given_kept=expected_batches,
        expected_nonempty_batches_given_correction=expected_batches_correction,
        expected_m3_stages=expected_attempts,
        expected_m1_stages=correction_probability,
        expected_total_tmr_stages=expected_attempts + correction_probability,
        expected_physical_cnots=expected_cnots,
        expected_cnot_layers=expected_layers,
        expected_physical_rotations=expected_rotations,
        expected_syndrome_cycles=expected_syndromes,
        residual_correction_angle=residual_angle,
        final_angle_error=float(final_error),
    )


def compare_policies(theta: float, thresholds: Iterable[int] = (0, 1, 2, 3, 8)) -> list[PolicyResult]:
    return [evaluate_policy(theta, threshold) for threshold in thresholds]


def monte_carlo_policy(
    theta: float,
    threshold: int,
    *,
    shots: int = 1_000_000,
    seed: int = 20260904,
) -> dict[str, float | int]:
    """Cross-check analytic policy moments with independent Bernoulli shots."""

    if shots <= 0:
        raise ValueError("shots must be positive")
    table = angle_table(theta)
    bad_probability = float(table["probability_any_alternative"])
    rng = np.random.default_rng(seed)
    bad = rng.random((shots, NUM_LOGICALS)) < bad_probability
    counts = np.count_nonzero(bad, axis=1)
    kept = counts <= threshold
    kept_shots = int(np.count_nonzero(kept))
    keep = kept_shots / shots
    keep_se = math.sqrt(keep * (1 - keep) / shots)
    if kept_shots:
        retained_counts = counts[kept]
        mean_bad = float(np.mean(retained_counts))
        bad_se = float(np.std(retained_counts, ddof=1) / math.sqrt(kept_shots)) if kept_shots > 1 else 0.0
        batch_counts = np.zeros(kept_shots, dtype=np.uint8)
        retained = bad[kept]
        for batch in BATCHES:
            batch_counts += np.any(retained[:, list(batch)], axis=1)
        mean_batches = float(np.mean(batch_counts))
        batch_se = float(np.std(batch_counts, ddof=1) / math.sqrt(kept_shots)) if kept_shots > 1 else 0.0
        correction = float(np.mean(retained_counts > 0))
        correction_se = math.sqrt(correction * (1 - correction) / kept_shots)
    else:
        mean_bad = bad_se = mean_batches = batch_se = correction = correction_se = float("nan")
    return {
        "shots": int(shots),
        "seed": int(seed),
        "kept_shots": kept_shots,
        "keep_probability": keep,
        "keep_probability_se": keep_se,
        "expected_bad_logicals_given_kept": mean_bad,
        "expected_bad_logicals_given_kept_se": bad_se,
        "expected_nonempty_batches_given_kept": mean_batches,
        "expected_nonempty_batches_given_kept_se": batch_se,
        "correction_probability_given_kept": correction,
        "correction_probability_given_kept_se": correction_se,
    }
