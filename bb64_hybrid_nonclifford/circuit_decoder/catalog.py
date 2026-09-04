"""Compressed detector/frame signatures for the scheduled BB64 circuit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from bb64_hybrid_nonclifford.syndrome_history import SyndromeHistoryCircuit

from .frames import BoundaryFrame, BoundaryFrameDecomposer
from .ledger import FaultLedger, build_fault_ledger
from .propagate import PauliSignature, ParsedCircuit, parse_circuit, propagate_faults


@dataclass(frozen=True)
class SignatureGroup:
    signature: PauliSignature
    frame: BoundaryFrame
    relative_weight: float
    mechanism_count: int
    location_weights: tuple[tuple[int, float], ...]
    minimum_mechanism_id: int


@dataclass(frozen=True)
class PairSignatureGroup:
    signature: PauliSignature
    frame: BoundaryFrame
    relative_weight: float
    source_group_pairs: int


@dataclass(frozen=True)
class FaultCatalog:
    circuit: ParsedCircuit
    history: SyndromeHistoryCircuit
    ledger: FaultLedger
    no_fault_group: SignatureGroup
    single_groups: tuple[SignatureGroup, ...]
    pair_groups: tuple[PairSignatureGroup, ...] = ()
    selected_pair_relative_weight: float = 0.0

    @property
    def modeled_absolute_probability(self) -> float:
        relative = 1.0 + math.fsum(group.relative_weight for group in self.single_groups)
        relative += math.fsum(group.relative_weight for group in self.pair_groups)
        return self.ledger.no_fault_probability * relative

    @property
    def conservative_tail_probability(self) -> float:
        return max(0.0, 1.0 - self.modeled_absolute_probability)


def _key(signature: PauliSignature) -> tuple[int, int, int, int]:
    return (
        signature.detector_mask,
        signature.final_x_mask,
        signature.final_z_mask,
        signature.rotation_sign_mask,
    )


def _xor(left: PauliSignature, right: PauliSignature) -> PauliSignature:
    return PauliSignature(
        left.detector_mask ^ right.detector_mask,
        left.final_x_mask ^ right.final_x_mask,
        left.final_z_mask ^ right.final_z_mask,
        left.rotation_sign_mask ^ right.rotation_sign_mask,
    )


def build_fault_catalog(
    history: SyndromeHistoryCircuit,
    code,
    *,
    mechanism_ids: tuple[int, ...] | None = None,
    progress: Callable[[str], None] | None = None,
) -> FaultCatalog:
    """Propagate and merge all single-location Pauli mechanisms."""

    parsed = parse_circuit(history.text)
    ledger = build_fault_ledger(history.text, progress=progress)
    decomposer = BoundaryFrameDecomposer(code)
    empty = PauliSignature(0, 0, 0, 0)
    no_fault = SignatureGroup(empty, decomposer.decompose(0, 0), 1.0, 0, (), -1)
    grouped: dict[tuple[int, int, int, int], dict[str, object]] = {}
    selected_mechanisms = (
        ledger.mechanisms
        if mechanism_ids is None
        else tuple(ledger.mechanisms[index] for index in mechanism_ids)
    )
    total = len(selected_mechanisms)
    for count, mechanism in enumerate(selected_mechanisms, start=1):
        signature = propagate_faults(parsed, (mechanism,))
        record = grouped.setdefault(
            _key(signature),
            {"signature": signature, "weight": 0.0, "locations": {}, "count": 0, "minimum": mechanism.mechanism_id},
        )
        record["weight"] = float(record["weight"]) + mechanism.odds
        locations = record["locations"]
        assert isinstance(locations, dict)
        locations[mechanism.location_id] = locations.get(mechanism.location_id, 0.0) + mechanism.odds
        record["count"] = int(record["count"]) + 1
        if progress is not None and (count % 2000 == 0 or count == total):
            progress(f"single signatures: {count:,}/{total:,}; {len(grouped):,} unique")
    groups = []
    for record in grouped.values():
        signature = record["signature"]
        assert isinstance(signature, PauliSignature)
        locations = record["locations"]
        assert isinstance(locations, dict)
        groups.append(
            SignatureGroup(
                signature=signature,
                frame=decomposer.decompose(signature.final_x_mask, signature.final_z_mask),
                relative_weight=float(record["weight"]),
                mechanism_count=int(record["count"]),
                location_weights=tuple(sorted((int(key), float(value)) for key, value in locations.items())),
                minimum_mechanism_id=int(record["minimum"]),
            )
        )
    groups.sort(key=lambda group: (-group.relative_weight, group.minimum_mechanism_id))
    represented_single_odds = math.fsum(group.relative_weight for group in groups)
    expected_single_odds = ledger.single_fault_probability / ledger.no_fault_probability
    if mechanism_ids is None and not math.isclose(
        represented_single_odds, expected_single_odds, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise AssertionError("compressed groups lost single-fault probability mass")
    return FaultCatalog(parsed, history, ledger, no_fault, tuple(groups))


def add_selected_pairs(
    catalog: FaultCatalog,
    code,
    *,
    top_groups: int,
    progress: Callable[[str], None] | None = None,
) -> FaultCatalog:
    """Add exact distinct-location pairs among the most probable signatures."""

    if top_groups < 0:
        raise ValueError("top_groups must be nonnegative")
    selected = catalog.single_groups[:top_groups]
    decomposer = BoundaryFrameDecomposer(code)
    pair_records: dict[tuple[int, int, int, int], list[object]] = {}
    pair_count = len(selected) * (len(selected) + 1) // 2
    visited = 0
    for left_index, left in enumerate(selected):
        left_locations = dict(left.location_weights)
        for right_index in range(left_index, len(selected)):
            right = selected[right_index]
            right_locations = dict(right.location_weights)
            if left_index == right_index:
                total = left.relative_weight * left.relative_weight
                same_location = math.fsum(value * value for value in left_locations.values())
                weight = (total - same_location) / 2.0
            else:
                same_location = math.fsum(
                    value * right_locations.get(location, 0.0)
                    for location, value in left_locations.items()
                )
                weight = left.relative_weight * right.relative_weight - same_location
            visited += 1
            if weight <= 0:
                continue
            signature = _xor(left.signature, right.signature)
            key = _key(signature)
            record = pair_records.setdefault(key, [signature, 0.0, 0])
            record[1] = float(record[1]) + weight
            record[2] = int(record[2]) + 1
            if progress is not None and (visited % 1000 == 0 or visited == pair_count):
                progress(f"selected pairs: {visited:,}/{pair_count:,}; {len(pair_records):,} unique")
    pairs = []
    for signature, weight, sources in pair_records.values():
        assert isinstance(signature, PauliSignature)
        pairs.append(
            PairSignatureGroup(
                signature,
                decomposer.decompose(signature.final_x_mask, signature.final_z_mask),
                float(weight),
                int(sources),
            )
        )
    pairs.sort(key=lambda group: -group.relative_weight)
    selected_weight = math.fsum(group.relative_weight for group in pairs)
    return FaultCatalog(
        catalog.circuit,
        catalog.history,
        catalog.ledger,
        catalog.no_fault_group,
        catalog.single_groups,
        tuple(pairs),
        selected_weight,
    )
