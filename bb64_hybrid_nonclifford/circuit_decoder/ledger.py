"""Enumerate every elementary Pauli mechanism in a generated BB64 circuit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable


_INSTRUCTION = re.compile(r"^([A-Z0-9_]+)(?:\(([^)]*)\))?(?:\s+(.*))?$")
_PAULIS = ((0, 0, "I"), (1, 0, "X"), (1, 1, "Y"), (0, 1, "Z"))


@dataclass(frozen=True)
class FaultMechanism:
    """One mutually exclusive nonidentity outcome of one noise location."""

    mechanism_id: int
    location_id: int
    instruction_index: int
    channel: str
    targets: tuple[int, ...]
    pauli: str
    x_mask: int
    z_mask: int
    probability: float
    odds: float


@dataclass(frozen=True)
class FaultLocation:
    location_id: int
    instruction_index: int
    channel: str
    targets: tuple[int, ...]
    probability: float
    mechanism_ids: tuple[int, ...]


@dataclass(frozen=True)
class FaultLedger:
    mechanisms: tuple[FaultMechanism, ...]
    locations: tuple[FaultLocation, ...]
    no_fault_probability: float
    single_fault_probability: float
    exactly_two_fault_probability: float
    through_two_fault_probability: float

    @property
    def higher_fault_tail_probability(self) -> float:
        return max(0.0, 1.0 - self.through_two_fault_probability)


def parse_instruction(line: str) -> tuple[str, tuple[float, ...], tuple[str, ...]]:
    match = _INSTRUCTION.match(line.strip())
    if match is None:
        raise ValueError(f"cannot parse circuit instruction: {line!r}")
    name, argument_text, target_text = match.groups()
    arguments = () if argument_text is None else tuple(float(item.strip()) for item in argument_text.split(","))
    targets = () if target_text is None else tuple(target_text.split())
    return name, arguments, targets


def _mask(paulis: Iterable[tuple[int, int]]) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for qubit, code in paulis:
        x, z, _label = _PAULIS[code]
        x_mask ^= x << qubit
        z_mask ^= z << qubit
    return x_mask, z_mask


def build_fault_ledger(
    circuit_text: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> FaultLedger:
    """Expand Stim noise channels into elementary Pauli mechanisms.

    A multi-target noise instruction represents one statistically independent
    location per target (or target pair for ``DEPOLARIZE2``).  Mechanism odds
    are relative to the identity outcome of that same location, which makes
    exact zero-, one-, and distinct-location two-fault masses easy to compose.
    """

    mechanisms: list[FaultMechanism] = []
    locations: list[FaultLocation] = []
    supported = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}
    for instruction_index, line in enumerate(circuit_text.splitlines()):
        if not line.strip():
            continue
        name, arguments, raw_targets = parse_instruction(line)
        if name not in supported:
            continue
        if len(arguments) != 1 or not 0.0 <= arguments[0] < 1.0:
            raise ValueError(f"invalid probability on instruction {instruction_index}")
        probability = arguments[0]
        targets = tuple(map(int, raw_targets))
        width = 2 if name == "DEPOLARIZE2" else 1
        if len(targets) % width:
            raise ValueError(f"odd target count on {name} instruction {instruction_index}")
        for offset in range(0, len(targets), width):
            location_targets = targets[offset : offset + width]
            location_id = len(locations)
            first_mechanism = len(mechanisms)
            if name == "DEPOLARIZE1":
                components = ((1,), (2,), (3,))
                component_probability = probability / 3.0
            elif name == "DEPOLARIZE2":
                components = tuple(
                    (left, right)
                    for left in range(4)
                    for right in range(4)
                    if left or right
                )
                component_probability = probability / 15.0
            elif name == "X_ERROR":
                components = ((1,),)
                component_probability = probability
            else:
                components = ((3,),)
                component_probability = probability
            for component in components:
                x_mask, z_mask = _mask(zip(location_targets, component, strict=True))
                label = "".join(_PAULIS[code][2] for code in component)
                mechanisms.append(
                    FaultMechanism(
                        mechanism_id=len(mechanisms),
                        location_id=location_id,
                        instruction_index=instruction_index,
                        channel=name,
                        targets=location_targets,
                        pauli=label,
                        x_mask=x_mask,
                        z_mask=z_mask,
                        probability=component_probability,
                        odds=component_probability / (1.0 - probability),
                    )
                )
            locations.append(
                FaultLocation(
                    location_id=location_id,
                    instruction_index=instruction_index,
                    channel=name,
                    targets=location_targets,
                    probability=probability,
                    mechanism_ids=tuple(range(first_mechanism, len(mechanisms))),
                )
            )
        if progress is not None and len(locations) and len(locations) % 500 == 0:
            progress(f"ledger: {len(locations):,} locations, {len(mechanisms):,} mechanisms")

    log_p0 = math.fsum(math.log1p(-location.probability) for location in locations)
    p0 = math.exp(log_p0)
    location_odds = [location.probability / (1.0 - location.probability) for location in locations]
    first = math.fsum(location_odds)
    second = (first * first - math.fsum(value * value for value in location_odds)) / 2.0
    p1 = p0 * first
    p2 = p0 * second
    return FaultLedger(
        mechanisms=tuple(mechanisms),
        locations=tuple(locations),
        no_fault_probability=p0,
        single_fault_probability=p1,
        exactly_two_fault_probability=p2,
        through_two_fault_probability=p0 + p1 + p2,
    )
