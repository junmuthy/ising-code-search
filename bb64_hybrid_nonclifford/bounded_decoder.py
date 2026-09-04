"""Bounded-fault latent-boundary reference decoder for BB64 histories."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Literal, Sequence

import numpy as np

from bb64_syndrome_recovery.algebra import angle_table

from .decoder import DecoderResult, NoiselessTableDecoder
from .model import HybridModel


Axis = Literal["X", "Z"]


@dataclass(frozen=True)
class FaultEvent:
    axis: Axis
    kind: Literal["data", "measurement"]
    round_index: int
    location: int
    history_pattern: int
    log_odds: float


@dataclass(frozen=True)
class FaultExplanation:
    score_offset: float
    event_indices: tuple[int, ...]
    data_correction: tuple[int, ...]
    weight: int


@dataclass(frozen=True)
class ChannelDecode:
    in_radius: bool
    syndrome_class_id: int | None
    runner_up_class_id: int | None
    best_score: float
    runner_up_score: float
    likelihood_gap: float
    ambiguous: bool
    events: tuple[FaultEvent, ...]
    data_correction: tuple[int, ...]


@dataclass(frozen=True)
class LatentBoundaryResult:
    in_radius: bool
    ambiguous: bool
    branch: DecoderResult | None
    x_syndrome_decode: ChannelDecode
    z_syndrome_decode: ChannelDecode
    data_z_correction: tuple[int, ...]
    data_x_correction: tuple[int, ...]


def _history_to_int(history: np.ndarray, rounds: int) -> int:
    bits = np.asarray(history, dtype=np.uint8)
    if bits.shape != (rounds, 32) or np.any(bits > 1):
        raise ValueError(f"history must be a binary array of shape ({rounds}, 32)")
    packed = np.packbits(bits.reshape(-1), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def _syndrome_to_int(syndrome: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(syndrome))


def _repeat_syndrome(value: int, rounds: int) -> int:
    return sum(value << (32 * round_index) for round_index in range(rounds))


def _persistent_pattern(syndrome: int, start_round: int, rounds: int) -> int:
    return sum(syndrome << (32 * round_index) for round_index in range(start_round, rounds))


def _event_catalog(
    matrix: np.ndarray,
    *,
    axis: Axis,
    rounds: int,
    data_probability: float,
    measurement_probability: float,
) -> tuple[FaultEvent, ...]:
    if not 0 < data_probability < 0.5 or not 0 < measurement_probability < 0.5:
        raise ValueError("fault probabilities must lie strictly between zero and 1/2")
    data_log_odds = math.log(data_probability / (1.0 - data_probability))
    measurement_log_odds = math.log(
        measurement_probability / (1.0 - measurement_probability)
    )
    events: list[FaultEvent] = []
    for round_index in range(rounds):
        for qubit in range(matrix.shape[1]):
            syndrome = _syndrome_to_int(matrix[:, qubit])
            events.append(
                FaultEvent(
                    axis=axis,
                    kind="data",
                    round_index=round_index,
                    location=qubit,
                    history_pattern=_persistent_pattern(syndrome, round_index, rounds),
                    log_odds=data_log_odds,
                )
            )
        for check in range(32):
            events.append(
                FaultEvent(
                    axis=axis,
                    kind="measurement",
                    round_index=round_index,
                    location=check,
                    history_pattern=1 << (32 * round_index + check),
                    log_odds=measurement_log_odds,
                )
            )
    return tuple(events)


def _correction_for_events(events: Sequence[FaultEvent]) -> tuple[int, ...]:
    parity: set[int] = set()
    for event in events:
        if event.kind != "data":
            continue
        if event.location in parity:
            parity.remove(event.location)
        else:
            parity.add(event.location)
    return tuple(sorted(parity))


def _build_bounded_catalog(
    events: Sequence[FaultEvent], max_faults: int
) -> dict[int, FaultExplanation]:
    if max_faults not in (0, 1, 2):
        raise ValueError("the reference catalog supports maximum fault weight 0, 1, or 2")
    catalog: dict[int, FaultExplanation] = {
        0: FaultExplanation(score_offset=0.0, event_indices=(), data_correction=(), weight=0)
    }
    for weight in range(1, max_faults + 1):
        for indices in itertools.combinations(range(len(events)), weight):
            selected = tuple(events[index] for index in indices)
            pattern = 0
            score = 0.0
            for event in selected:
                pattern ^= event.history_pattern
                score += event.log_odds
            explanation = FaultExplanation(
                score_offset=score,
                event_indices=indices,
                data_correction=_correction_for_events(selected),
                weight=weight,
            )
            previous = catalog.get(pattern)
            if previous is None or (score, tuple(-index for index in indices)) > (
                previous.score_offset,
                tuple(-index for index in previous.event_indices),
            ):
                catalog[pattern] = explanation
    return catalog


class BoundedLatentBoundaryDecoder:
    """Decode a phenomenological history exactly through fault weight two.

    An X-check history is decomposed into a repeated ideal TMR syndrome plus
    bounded data-Z and X-check measurement events.  A Z-check history is
    decomposed into bounded data-X and Z-check measurement events around zero.
    This is an exact oracle for its stated event model, not yet the scheduled
    circuit-location decoder.
    """

    def __init__(
        self,
        model: HybridModel,
        *,
        theta: float,
        rounds: int,
        data_probability: float,
        measurement_probability: float,
        max_faults: int = 2,
        ambiguity_tolerance: float = 1e-12,
    ):
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        self.model = model
        self.theta = float(theta)
        self.rounds = int(rounds)
        self.max_faults = int(max_faults)
        self.ambiguity_tolerance = float(ambiguity_tolerance)
        self.x_events = _event_catalog(
            model.code.matrix_x,
            axis="Z",
            rounds=rounds,
            data_probability=data_probability,
            measurement_probability=measurement_probability,
        )
        self.z_events = _event_catalog(
            model.code.matrix_z,
            axis="X",
            rounds=rounds,
            data_probability=data_probability,
            measurement_probability=measurement_probability,
        )
        self.x_catalog = _build_bounded_catalog(self.x_events, max_faults)
        self.z_catalog = _build_bounded_catalog(self.z_events, max_faults)
        self.class_from_syndrome = {
            _syndrome_to_int(syndrome): class_id
            for class_id, syndrome in enumerate(model.displayed_syndromes)
        }
        table = angle_table(theta)
        p_target = float(table["probability_target"])
        p_alternative = float(table["alternative"]["probability_per_pattern"])
        bad = model.recovery.alternative_count
        self.log_prior = (8 - bad) * math.log(p_target) + bad * math.log(p_alternative)
        self.table_decoder = NoiselessTableDecoder(model)

    def _decode_x_channel(self, observed: int) -> ChannelDecode:
        best_by_class: dict[int, tuple[float, FaultExplanation]] = {}
        for pattern, explanation in self.x_catalog.items():
            corrected = observed ^ pattern
            first = corrected & ((1 << 32) - 1)
            if corrected != _repeat_syndrome(first, self.rounds):
                continue
            class_id = self.class_from_syndrome.get(first)
            if class_id is None:
                continue
            score = float(self.log_prior[class_id] + explanation.score_offset)
            previous = best_by_class.get(class_id)
            if previous is None or score > previous[0]:
                best_by_class[class_id] = (score, explanation)
        if not best_by_class:
            return ChannelDecode(False, None, None, -math.inf, -math.inf, 0.0, True, (), ())
        ordered = sorted(best_by_class.items(), key=lambda item: (-item[1][0], item[0]))
        class_id, (best_score, explanation) = ordered[0]
        if len(ordered) == 1:
            runner_up_id = None
            runner_up_score = -math.inf
            gap = math.inf
        else:
            runner_up_id, (runner_up_score, _) = ordered[1]
            gap = best_score - runner_up_score
        selected_events = tuple(self.x_events[index] for index in explanation.event_indices)
        return ChannelDecode(
            True,
            class_id,
            runner_up_id,
            best_score,
            runner_up_score,
            gap,
            gap <= self.ambiguity_tolerance,
            selected_events,
            explanation.data_correction,
        )

    def _decode_z_channel(self, observed: int) -> ChannelDecode:
        explanation = self.z_catalog.get(observed)
        if explanation is None:
            return ChannelDecode(False, None, None, -math.inf, -math.inf, 0.0, True, (), ())
        selected_events = tuple(self.z_events[index] for index in explanation.event_indices)
        return ChannelDecode(
            True,
            None,
            None,
            explanation.score_offset,
            -math.inf,
            math.inf,
            False,
            selected_events,
            explanation.data_correction,
        )

    def decode(self, x_history: np.ndarray, z_history: np.ndarray) -> LatentBoundaryResult:
        x_decode = self._decode_x_channel(_history_to_int(x_history, self.rounds))
        z_decode = self._decode_z_channel(_history_to_int(z_history, self.rounds))
        in_radius = x_decode.in_radius and z_decode.in_radius
        ambiguous = not in_radius or x_decode.ambiguous or z_decode.ambiguous
        branch = None
        if x_decode.syndrome_class_id is not None:
            branch = self.table_decoder.decode(
                self.model.displayed_syndromes[x_decode.syndrome_class_id]
            )
        return LatentBoundaryResult(
            in_radius=in_radius,
            ambiguous=ambiguous,
            branch=branch,
            x_syndrome_decode=x_decode,
            z_syndrome_decode=z_decode,
            data_z_correction=x_decode.data_correction,
            data_x_correction=z_decode.data_correction,
        )

    def apply_events(
        self,
        history: np.ndarray,
        *,
        channel: Literal["x", "z"],
        event_indices: Iterable[int],
    ) -> np.ndarray:
        """Return a copied history with selected catalog events injected."""

        value = _history_to_int(history, self.rounds)
        events = self.x_events if channel == "x" else self.z_events
        for index in event_indices:
            value ^= events[int(index)].history_pattern
        byte_count = (self.rounds * 32 + 7) // 8
        packed = np.frombuffer(value.to_bytes(byte_count, byteorder="little"), dtype=np.uint8)
        return np.unpackbits(packed, bitorder="little")[: self.rounds * 32].reshape(
            self.rounds, 32
        )
