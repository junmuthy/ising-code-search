"""Categorical factor graph for arbitrary-weight BB64 circuit faults."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from bb64_hybrid_nonclifford.circuit import class_id_for_labels
from bb64_hybrid_nonclifford.model import HybridModel
from bb64_hybrid_nonclifford.syndrome_history import SyndromeHistoryCircuit
from bb64_syndrome_recovery.algebra import angle_table

from .ledger import FaultLedger, FaultLocation, build_fault_ledger
from .propagate import PauliSignature, ParsedCircuit, parse_circuit, propagate_faults


@dataclass(frozen=True)
class CategoricalVariable:
    kind: str
    source_id: int
    bit_width: int
    state_codes: tuple[int, ...]
    state_semantics: tuple[int, ...]
    log_priors: tuple[float, ...]
    signatures: tuple[PauliSignature, ...]

    def __post_init__(self) -> None:
        size = 1 << self.bit_width
        if self.state_codes != tuple(range(size)):
            raise ValueError("categorical state codes must exhaust their binary representation")
        if not (
            len(self.state_semantics)
            == len(self.log_priors)
            == len(self.signatures)
            == size
        ):
            raise ValueError("categorical state arrays have inconsistent lengths")
        probability = math.fsum(math.exp(value) for value in self.log_priors)
        if not math.isclose(probability, 1.0, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"variable prior is not normalized: {probability}")


@dataclass(frozen=True)
class D3FactorGraph:
    variables: tuple[CategoricalVariable, ...]
    physical_variable_count: int
    parsed_circuit: ParsedCircuit
    history: SyndromeHistoryCircuit
    ledger: FaultLedger
    component_offsets: tuple[int, ...]
    component_detector_masks: tuple[int, ...]

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_components(self) -> int:
        return self.component_offsets[-1]

    @property
    def branch_variable_indices(self) -> tuple[int, ...]:
        return tuple(range(self.physical_variable_count, len(self.variables)))

    def validate_codes(self, codes: Iterable[int]) -> tuple[int, ...]:
        values = tuple(map(int, codes))
        if len(values) != len(self.variables):
            raise ValueError("state-code vector has the wrong length")
        for code, variable in zip(values, self.variables, strict=True):
            if not 0 <= code < 1 << variable.bit_width:
                raise ValueError("state code is outside its variable alphabet")
        return values

    def detector_mask(self, codes: Iterable[int]) -> int:
        values = self.validate_codes(codes)
        result = 0
        for variable, code in zip(self.variables, values, strict=True):
            result ^= variable.signatures[code].detector_mask
        return result

    def physical_signature(self, codes: Iterable[int]) -> PauliSignature:
        values = self.validate_codes(codes)
        detector = final_x = final_z = rotation = 0
        for variable, code in zip(
            self.variables[: self.physical_variable_count],
            values[: self.physical_variable_count],
            strict=True,
        ):
            signature = variable.signatures[code]
            detector ^= signature.detector_mask
            final_x ^= signature.final_x_mask
            final_z ^= signature.final_z_mask
            rotation ^= signature.rotation_sign_mask
        return PauliSignature(detector, final_x, final_z, rotation)

    def log_probability(self, codes: Iterable[int]) -> float:
        values = self.validate_codes(codes)
        return math.fsum(
            variable.log_priors[code]
            for variable, code in zip(self.variables, values, strict=True)
        )

    def fault_count(self, codes: Iterable[int]) -> int:
        values = self.validate_codes(codes)
        return sum(code != 0 for code in values[: self.physical_variable_count])

    def branch_labels(self, codes: Iterable[int]) -> tuple[int, ...]:
        values = self.validate_codes(codes)
        return tuple(
            variable.state_semantics[code]
            for variable, code in zip(
                self.variables[self.physical_variable_count :],
                values[self.physical_variable_count :],
                strict=True,
            )
        )

    def branch_class_id(self, model: HybridModel, codes: Iterable[int]) -> int:
        return class_id_for_labels(model, self.branch_labels(codes))


def _xor_signatures(signatures: Iterable[PauliSignature]) -> PauliSignature:
    detector = final_x = final_z = rotation = 0
    for signature in signatures:
        detector ^= signature.detector_mask
        final_x ^= signature.final_x_mask
        final_z ^= signature.final_z_mask
        rotation ^= signature.rotation_sign_mask
    return PauliSignature(detector, final_x, final_z, rotation)


def _pauli_bits(label: str) -> int:
    code = 0
    for target, pauli in enumerate(label):
        if pauli in "XY":
            code |= 1 << (2 * target)
        if pauli in "YZ":
            code |= 1 << (2 * target + 1)
    return code


def _location_width(location: FaultLocation) -> int:
    if location.channel == "DEPOLARIZE1":
        return 2
    if location.channel == "DEPOLARIZE2":
        return 4
    if location.channel in ("X_ERROR", "Z_ERROR"):
        return 1
    raise ValueError(f"unsupported location channel: {location.channel}")


def _physical_variable(
    parsed: ParsedCircuit,
    ledger: FaultLedger,
    location: FaultLocation,
) -> CategoricalVariable:
    width = _location_width(location)
    states: dict[int, tuple[int, float, PauliSignature]] = {
        0: (-1, math.log1p(-location.probability), PauliSignature(0, 0, 0, 0))
    }
    for mechanism_id in location.mechanism_ids:
        mechanism = ledger.mechanisms[mechanism_id]
        code = 1 if width == 1 else _pauli_bits(mechanism.pauli)
        if code in states:
            raise ValueError("two physical outcomes have the same binary state code")
        states[code] = (
            mechanism_id,
            math.log(mechanism.probability),
            propagate_faults(parsed, (mechanism,)),
        )
    if set(states) != set(range(1 << width)):
        raise ValueError("physical location does not exhaust its Pauli state space")
    ordered = tuple(states[code] for code in range(1 << width))
    variable = CategoricalVariable(
        kind=location.channel,
        source_id=location.location_id,
        bit_width=width,
        state_codes=tuple(range(1 << width)),
        state_semantics=tuple(item[0] for item in ordered),
        log_priors=tuple(item[1] for item in ordered),
        signatures=tuple(item[2] for item in ordered),
    )
    basis = variable.signatures
    for code, signature in enumerate(basis):
        predicted = _xor_signatures(
            basis[1 << bit] for bit in range(width) if (code >> bit) & 1
        )
        if predicted != signature:
            raise AssertionError("Pauli location signature is not linear in symplectic bits")
    return variable


def _ideal_detector_mask(history: SyndromeHistoryCircuit, syndrome: np.ndarray) -> int:
    result = 0
    for group in history.x_detector_groups:
        for check, detector in enumerate(group):
            result |= int(syndrome[check]) << detector
    return result


def _branch_variable(
    model: HybridModel,
    history: SyndromeHistoryCircuit,
    logical: int,
    theta: float,
) -> CategoricalVariable:
    table = angle_table(theta)
    p0 = float(table["probability_target"])
    p1 = float(table["alternative"]["probability_per_pattern"])
    states: dict[int, tuple[int, float, PauliSignature]] = {}
    for label in range(4):
        labels = [0] * 8
        labels[logical] = label
        class_id = class_id_for_labels(model, labels)
        quotient = model.quotient_coordinates[class_id, logical]
        code = int(quotient[0]) | (int(quotient[1]) << 1)
        detector = _ideal_detector_mask(history, model.displayed_syndromes[class_id])
        states[code] = (
            label,
            math.log(p0 if label == 0 else p1),
            PauliSignature(detector, 0, 0, 0),
        )
    ordered = tuple(states[code] for code in range(4))
    variable = CategoricalVariable(
        kind="TMR_BRANCH",
        source_id=logical,
        bit_width=2,
        state_codes=(0, 1, 2, 3),
        state_semantics=tuple(item[0] for item in ordered),
        log_priors=tuple(item[1] for item in ordered),
        signatures=tuple(item[2] for item in ordered),
    )
    for code, signature in enumerate(variable.signatures):
        predicted = _xor_signatures(
            variable.signatures[1 << bit] for bit in range(2) if (code >> bit) & 1
        )
        if predicted != signature:
            raise AssertionError("TMR syndrome is not linear in quotient coordinates")
    return variable


def _component_data(
    variables: tuple[CategoricalVariable, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [0]
    detectors: list[int] = []
    for variable in variables:
        detectors.extend(
            variable.signatures[1 << bit].detector_mask
            for bit in range(variable.bit_width)
        )
        offsets.append(len(detectors))
    return tuple(offsets), tuple(detectors)


def build_d3_factor_graph(
    model: HybridModel,
    history: SyndromeHistoryCircuit,
    *,
    theta: float,
    progress: Callable[[str], None] | None = None,
) -> D3FactorGraph:
    parsed = parse_circuit(history.text)
    ledger = build_fault_ledger(history.text)
    physical: list[CategoricalVariable] = []
    for count, location in enumerate(ledger.locations, start=1):
        physical.append(_physical_variable(parsed, ledger, location))
        if progress is not None and (count % 250 == 0 or count == len(ledger.locations)):
            progress(f"factor graph: physical variables {count:,}/{len(ledger.locations):,}")
    branches = [
        _branch_variable(model, history, logical, theta) for logical in range(8)
    ]
    variables = tuple((*physical, *branches))
    offsets, detectors = _component_data(variables)
    return D3FactorGraph(
        variables=variables,
        physical_variable_count=len(physical),
        parsed_circuit=parsed,
        history=history,
        ledger=ledger,
        component_offsets=offsets,
        component_detector_masks=detectors,
    )


def _words(value: int, count: int) -> tuple[int, ...]:
    return tuple((value >> (64 * index)) & ((1 << 64) - 1) for index in range(count))


def _integer(words: np.ndarray) -> int:
    return sum(int(value) << (64 * index) for index, value in enumerate(words))


def save_d3_factor_graph(path: Path, graph: D3FactorGraph) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_offsets = [0]
    state_codes: list[int] = []
    semantics: list[int] = []
    log_priors: list[float] = []
    detector_words: list[tuple[int, ...]] = []
    final_x: list[int] = []
    final_z: list[int] = []
    rotation: list[int] = []
    for variable in graph.variables:
        for code, semantic, log_prior, signature in zip(
            variable.state_codes,
            variable.state_semantics,
            variable.log_priors,
            variable.signatures,
            strict=True,
        ):
            state_codes.append(code)
            semantics.append(semantic)
            log_priors.append(log_prior)
            detector_words.append(_words(signature.detector_mask, 4))
            final_x.append(signature.final_x_mask)
            final_z.append(signature.final_z_mask)
            rotation.append(signature.rotation_sign_mask)
        state_offsets.append(len(state_codes))
    payload = {
        "schema": np.asarray("bb64-d3-factor-graph-v1"),
        "circuit_sha256": np.asarray(hashlib.sha256(graph.history.text.encode()).hexdigest()),
        "physical_variable_count": np.asarray(graph.physical_variable_count, dtype=np.uint32),
        "variable_kind": np.asarray([variable.kind for variable in graph.variables]),
        "variable_source": np.asarray([variable.source_id for variable in graph.variables], dtype=np.int32),
        "variable_width": np.asarray([variable.bit_width for variable in graph.variables], dtype=np.uint8),
        "state_offsets": np.asarray(state_offsets, dtype=np.uint32),
        "state_codes": np.asarray(state_codes, dtype=np.uint8),
        "state_semantics": np.asarray(semantics, dtype=np.int32),
        "state_log_priors": np.asarray(log_priors, dtype=np.float64),
        "state_detector": np.asarray(detector_words, dtype=np.uint64),
        "state_final_x": np.asarray(final_x, dtype=np.uint64),
        "state_final_z": np.asarray(final_z, dtype=np.uint64),
        "state_rotation": np.asarray(rotation, dtype=np.uint32),
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_d3_factor_graph(path: Path, history: SyndromeHistoryCircuit) -> D3FactorGraph:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if str(arrays["schema"]) != "bb64-d3-factor-graph-v1":
        raise ValueError("unknown D3 factor-graph schema")
    digest = hashlib.sha256(history.text.encode()).hexdigest()
    if str(arrays["circuit_sha256"]) != digest:
        raise ValueError("factor graph was generated from a different circuit")
    offsets = arrays["state_offsets"]
    variables = []
    for index in range(len(arrays["variable_width"])):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        signatures = tuple(
            PauliSignature(
                _integer(arrays["state_detector"][state]),
                int(arrays["state_final_x"][state]),
                int(arrays["state_final_z"][state]),
                int(arrays["state_rotation"][state]),
            )
            for state in range(start, stop)
        )
        variables.append(
            CategoricalVariable(
                str(arrays["variable_kind"][index]),
                int(arrays["variable_source"][index]),
                int(arrays["variable_width"][index]),
                tuple(map(int, arrays["state_codes"][start:stop])),
                tuple(map(int, arrays["state_semantics"][start:stop])),
                tuple(map(float, arrays["state_log_priors"][start:stop])),
                signatures,
            )
        )
    parsed = parse_circuit(history.text)
    ledger = build_fault_ledger(history.text)
    variable_tuple = tuple(variables)
    component_offsets, component_detectors = _component_data(variable_tuple)
    return D3FactorGraph(
        variable_tuple,
        int(arrays["physical_variable_count"]),
        parsed,
        history,
        ledger,
        component_offsets,
        component_detectors,
    )
