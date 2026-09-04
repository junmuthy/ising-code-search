"""Exact table and repeated-measurement reference decoders."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from bb64_syndrome_recovery.algebra import NUM_LOGICALS, angle_table

from .model import HybridModel


@dataclass(frozen=True)
class DecoderResult:
    syndrome_class_id: int
    syndrome_coordinates: tuple[int, ...]
    local_branch_labels: tuple[int, ...]
    alternative_mask: tuple[int, ...]
    selected_piece: tuple[int, ...]
    bad_count: int
    correction_support: tuple[int, ...]
    correction_gauge_z_frame: tuple[int, ...]
    best_log_likelihood: float = 0.0
    runner_up_log_likelihood: float = -math.inf
    likelihood_gap: float = math.inf
    posterior_probability: float = 1.0
    ambiguous: bool = False


@dataclass(frozen=True)
class DecodeBatch:
    syndrome_class_id: np.ndarray
    runner_up_class_id: np.ndarray
    best_log_likelihood: np.ndarray
    runner_up_log_likelihood: np.ndarray
    likelihood_gap: np.ndarray
    posterior_probability: np.ndarray
    ambiguous: np.ndarray


def _result_from_class(
    model: HybridModel,
    class_id: int,
    *,
    best_log_likelihood: float = 0.0,
    runner_up_log_likelihood: float = -math.inf,
    posterior_probability: float = 1.0,
    ambiguity_gap: float = 0.0,
) -> DecoderResult:
    recovery = model.recovery
    class_id = int(class_id)
    if not 0 <= class_id < len(recovery.syndrome_coordinates):
        raise ValueError("syndrome class ID is out of range")
    gap = float(best_log_likelihood - runner_up_log_likelihood)
    return DecoderResult(
        syndrome_class_id=class_id,
        syndrome_coordinates=tuple(map(int, recovery.syndrome_coordinates[class_id])),
        local_branch_labels=tuple(map(int, recovery.local_branch_labels[class_id])),
        alternative_mask=tuple(map(int, recovery.alternative_mask[class_id])),
        selected_piece=tuple(map(int, recovery.selected_piece[class_id])),
        bad_count=int(recovery.alternative_count[class_id]),
        correction_support=tuple(
            np.flatnonzero(recovery.correction_physical_supports[class_id]).astype(int).tolist()
        ),
        correction_gauge_z_frame=tuple(map(int, recovery.logical_z_frame[class_id])),
        best_log_likelihood=float(best_log_likelihood),
        runner_up_log_likelihood=float(runner_up_log_likelihood),
        likelihood_gap=gap,
        posterior_probability=float(posterior_probability),
        ambiguous=bool(gap <= ambiguity_gap),
    )


class NoiselessTableDecoder:
    """Exact membership check and lookup for ideal TMR syndromes."""

    def __init__(self, model: HybridModel):
        self.model = model
        self.rows = np.asarray(model.recovery.independent_syndrome_rows, dtype=int)

    def decode(self, displayed_syndrome: np.ndarray) -> DecoderResult:
        syndrome = np.asarray(displayed_syndrome, dtype=np.uint8)
        if syndrome.shape != (32,):
            raise ValueError("displayed X syndrome must have shape (32,)")
        class_id = int(self.model.class_ids_from_coordinates(syndrome[self.rows]))
        if not np.array_equal(self.model.displayed_syndromes[class_id], syndrome):
            raise ValueError("displayed syndrome is outside the ideal TMR syndrome image")
        return _result_from_class(self.model, class_id)


class MeasurementMapDecoder:
    """Exact MAP reference decoder for a persistent syndrome plus readout flips.

    This decoder deliberately models measurement noise only.  It is the oracle
    against which the later circuit-level latent-boundary decoder will be
    checked.  All 32 displayed X-check outcomes are scored in every round.
    """

    def __init__(
        self,
        model: HybridModel,
        *,
        theta: float,
        measurement_probability: float,
        ambiguity_gap: float = 0.0,
    ):
        if not 0.0 < measurement_probability < 0.5:
            raise ValueError("measurement probability must lie strictly between zero and 1/2")
        self.model = model
        self.theta = float(theta)
        self.measurement_probability = float(measurement_probability)
        self.ambiguity_gap = float(ambiguity_gap)
        self.patterns = model.displayed_syndromes.astype(np.int16)
        self.log_prior = self._build_log_prior(theta)
        self.log_flip = math.log(measurement_probability)
        self.log_clean = math.log1p(-measurement_probability)

    def _build_log_prior(self, theta: float) -> np.ndarray:
        table = angle_table(theta)
        p_target = float(table["probability_target"])
        p_alternative_pattern = float(table["alternative"]["probability_per_pattern"])
        bad = self.model.recovery.alternative_count.astype(np.int16)
        prior = (NUM_LOGICALS - bad) * math.log(p_target) + bad * math.log(p_alternative_pattern)
        normalization = float(np.logaddexp.reduce(prior))
        return prior - normalization

    def decode_batch(
        self,
        histories: np.ndarray,
        *,
        batch_size: int = 16,
        progress: Callable[[str], None] | None = None,
    ) -> DecodeBatch:
        history = np.asarray(histories, dtype=np.uint8)
        if history.ndim == 2:
            history = history[None, :, :]
        if history.ndim != 3 or history.shape[2] != 32:
            raise ValueError("histories must have shape (shots, rounds, 32)")
        if np.any(history > 1):
            raise ValueError("syndrome histories must be binary")
        shots, rounds, _ = history.shape
        if rounds <= 0 or batch_size <= 0:
            raise ValueError("round count and batch size must be positive")

        best_ids = np.empty(shots, dtype=np.uint32)
        second_ids = np.empty(shots, dtype=np.uint32)
        best_scores = np.empty(shots, dtype=np.float64)
        second_scores = np.empty(shots, dtype=np.float64)
        posteriors = np.empty(shots, dtype=np.float64)
        coefficient = self.log_flip - self.log_clean

        for start in range(0, shots, batch_size):
            stop = min(start + batch_size, shots)
            ones = np.count_nonzero(history[start:stop], axis=1).astype(np.int16)
            # Hamming distance from each persistent ideal syndrome over every
            # repeated measurement round.
            distance = np.sum(ones, axis=1, dtype=np.int32)[:, None]
            distance = distance + (rounds - 2 * ones) @ self.patterns.T
            scores = self.log_prior[None, :] + distance * coefficient
            # Terms common to every class are omitted from the scores.  They
            # cancel in likelihood gaps and the posterior normalization.
            order = np.argpartition(scores, -2, axis=1)[:, -2:]
            pair_scores = np.take_along_axis(scores, order, axis=1)
            swap = pair_scores[:, 0] > pair_scores[:, 1]
            second = np.where(swap, order[:, 1], order[:, 0])
            best = np.where(swap, order[:, 0], order[:, 1])
            second_score = scores[np.arange(stop - start), second]
            best_score = scores[np.arange(stop - start), best]
            maximum = np.max(scores, axis=1)
            log_norm = maximum + np.log(np.sum(np.exp(scores - maximum[:, None]), axis=1))

            best_ids[start:stop] = best.astype(np.uint32)
            second_ids[start:stop] = second.astype(np.uint32)
            best_scores[start:stop] = best_score
            second_scores[start:stop] = second_score
            posteriors[start:stop] = np.exp(best_score - log_norm)
            if progress is not None:
                progress(f"decoded {stop:,}/{shots:,} histories")

        gaps = best_scores - second_scores
        return DecodeBatch(
            syndrome_class_id=best_ids,
            runner_up_class_id=second_ids,
            best_log_likelihood=best_scores,
            runner_up_log_likelihood=second_scores,
            likelihood_gap=gaps,
            posterior_probability=posteriors,
            ambiguous=gaps <= self.ambiguity_gap,
        )

    def decode(self, history: np.ndarray) -> DecoderResult:
        batch = self.decode_batch(np.asarray(history)[None, :, :])
        return _result_from_class(
            self.model,
            int(batch.syndrome_class_id[0]),
            best_log_likelihood=float(batch.best_log_likelihood[0]),
            runner_up_log_likelihood=float(batch.runner_up_log_likelihood[0]),
            posterior_probability=float(batch.posterior_probability[0]),
            ambiguity_gap=self.ambiguity_gap,
        )


def sample_measurement_histories(
    model: HybridModel,
    class_ids: np.ndarray,
    *,
    rounds: int,
    measurement_probability: float,
    seed: int,
) -> np.ndarray:
    """Sample independent readout flips around selected persistent syndromes."""

    ids = np.asarray(class_ids, dtype=np.uint32)
    if ids.ndim != 1 or np.any(ids >= len(model.displayed_syndromes)):
        raise ValueError("class IDs must be a one-dimensional in-range array")
    if rounds <= 0 or not 0 <= measurement_probability <= 1:
        raise ValueError("invalid measurement-noise configuration")
    rng = np.random.default_rng(seed)
    clean = np.repeat(model.displayed_syndromes[ids, None, :], rounds, axis=1)
    flips = rng.random(clean.shape) < measurement_probability
    return clean ^ flips.astype(np.uint8)
