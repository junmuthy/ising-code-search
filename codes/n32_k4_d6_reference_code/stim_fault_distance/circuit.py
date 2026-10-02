"""Build Clifford memory and bulk-cycle circuits for Stim."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Collection, Literal

import stim

from .model import CodeData, NUM_CHECKS_PER_TYPE, NUM_DATA_QUBITS, load_code_data


Basis = Literal["X", "Z"]
X_ANCILLA_START = NUM_DATA_QUBITS
Z_ANCILLA_START = NUM_DATA_QUBITS + NUM_CHECKS_PER_TYPE
NUM_QUBITS = NUM_DATA_QUBITS + 2 * NUM_CHECKS_PER_TYPE
DATA_QUBITS = tuple(range(NUM_DATA_QUBITS))
X_ANCILLAS = tuple(range(X_ANCILLA_START, Z_ANCILLA_START))
Z_ANCILLAS = tuple(range(Z_ANCILLA_START, NUM_QUBITS))
ALL_QUBITS = tuple(range(NUM_QUBITS))


@dataclass(frozen=True)
class NoiseModel:
    """Independent Pauli channels used in the Clifford circuit."""

    probability: float = 1e-3
    preparation: bool = True
    measurement: bool = True
    cnot: bool = True
    idle: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("noise probability must lie in [0, 1]")

    @classmethod
    def noiseless(cls) -> "NoiseModel":
        return cls(0.0, False, False, False, False)

    @classmethod
    def cnot_only(cls, probability: float = 1e-3) -> "NoiseModel":
        return cls(probability, False, False, True, False)

    @classmethod
    def without_idles(cls, probability: float = 1e-3) -> "NoiseModel":
        return cls(probability, True, True, True, False)

    @property
    def active(self) -> bool:
        return self.probability > 0 and any(
            (self.preparation, self.measurement, self.cnot, self.idle)
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _record_targets(circuit: stim.Circuit, measurements: Collection[int]) -> list[stim.GateTarget]:
    now = circuit.num_measurements
    return [stim.target_rec(int(index) - now) for index in measurements]


def _append_detector(
    circuit: stim.Circuit,
    measurements: Collection[int],
    coordinates: tuple[float, ...],
) -> None:
    circuit.append("DETECTOR", _record_targets(circuit, measurements), coordinates)


def _append_observable(
    circuit: stim.Circuit,
    measurements: Collection[int],
    observable: int,
) -> None:
    circuit.append(
        "OBSERVABLE_INCLUDE",
        _record_targets(circuit, measurements),
        observable,
    )


def _append_depolarize1(circuit: stim.Circuit, qubits: Collection[int], p: float) -> None:
    targets = list(qubits)
    if targets and p > 0:
        circuit.append("DEPOLARIZE1", targets, p)


def _append_measurement_error(
    circuit: stim.Circuit,
    basis: Basis,
    qubits: Collection[int],
    p: float,
) -> None:
    targets = list(qubits)
    if not targets or p <= 0:
        return
    circuit.append("Z_ERROR" if basis == "X" else "X_ERROR", targets, p)


def _prepare_data(
    circuit: stim.Circuit,
    basis: Basis,
    noise: NoiseModel,
    noisy: bool,
) -> None:
    circuit.append("RX" if basis == "X" else "R", DATA_QUBITS)
    if noisy and noise.preparation:
        _append_depolarize1(circuit, DATA_QUBITS, noise.probability)
    circuit.append("TICK")


def _measure_syndrome_round(
    circuit: stim.Circuit,
    code: CodeData,
    noise: NoiseModel,
    noisy: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    circuit.append("RX", X_ANCILLAS)
    circuit.append("R", Z_ANCILLAS)
    if noisy and noise.preparation:
        _append_depolarize1(circuit, X_ANCILLAS + Z_ANCILLAS, noise.probability)
    circuit.append("TICK")

    for layer in code.schedule:
        pairs: list[int] = []
        used: set[int] = set()
        for gate in layer:
            if gate.kind == "X":
                control = X_ANCILLA_START + gate.check
                target = gate.data
            elif gate.kind == "Z":
                control = gate.data
                target = Z_ANCILLA_START + gate.check
            else:
                raise ValueError(f"unknown check type: {gate.kind}")
            pairs.extend((control, target))
            used.update((control, target))
        circuit.append("CX", pairs)
        if noisy and noise.cnot:
            circuit.append("DEPOLARIZE2", pairs, noise.probability)
        if noisy and noise.idle:
            idle = [qubit for qubit in ALL_QUBITS if qubit not in used]
            _append_depolarize1(circuit, idle, noise.probability)
        circuit.append("TICK")

    if noisy and noise.measurement:
        _append_measurement_error(circuit, "X", X_ANCILLAS, noise.probability)
        _append_measurement_error(circuit, "Z", Z_ANCILLAS, noise.probability)
    start_x = circuit.num_measurements
    circuit.append("MX", X_ANCILLAS)
    measurements_x = tuple(range(start_x, start_x + NUM_CHECKS_PER_TYPE))
    start_z = circuit.num_measurements
    circuit.append("M", Z_ANCILLAS)
    measurements_z = tuple(range(start_z, start_z + NUM_CHECKS_PER_TYPE))
    circuit.append("TICK")
    return measurements_x, measurements_z


def _append_round_detectors(
    circuit: stim.Circuit,
    code: CodeData,
    basis: Basis,
    round_index: int,
    current_x: tuple[int, ...],
    current_z: tuple[int, ...],
    previous_x: tuple[int, ...] | None,
    previous_z: tuple[int, ...] | None,
) -> None:
    current = {"X": current_x, "Z": current_z}
    previous = {"X": previous_x, "Z": previous_z}
    if round_index == 0:
        for check, measurement in enumerate(current[basis]):
            _append_detector(
                circuit,
                (measurement,),
                (float(round_index), 0.0 if basis == "X" else 1.0, float(check)),
            )
    else:
        for kind_index, kind in enumerate(("X", "Z")):
            assert previous[kind] is not None
            for check in range(NUM_CHECKS_PER_TYPE):
                _append_detector(
                    circuit,
                    (current[kind][check], previous[kind][check]),
                    (float(round_index), float(kind_index), float(check)),
                )

    # These within-round constraints expose both independent redundant-check
    # relations, including the parity relation used by STAR postselection.
    for kind_index, kind in enumerate(("X", "Z")):
        for relation_index, relation in enumerate(code.redundant_relations):
            _append_detector(
                circuit,
                tuple(current[kind][check] for check in relation),
                (
                    float(round_index),
                    float(kind_index),
                    float(NUM_CHECKS_PER_TYPE + relation_index),
                ),
            )


def _measure_data_and_close(
    circuit: stim.Circuit,
    code: CodeData,
    basis: Basis,
    last_x: tuple[int, ...],
    last_z: tuple[int, ...],
    noise: NoiseModel,
    noisy: bool,
    rounds: int,
) -> tuple[int, ...]:
    if noisy and noise.measurement:
        _append_measurement_error(circuit, basis, DATA_QUBITS, noise.probability)
    start = circuit.num_measurements
    circuit.append("MX" if basis == "X" else "M", DATA_QUBITS)
    data_measurements = tuple(range(start, start + NUM_DATA_QUBITS))

    checks = code.checks_x if basis == "X" else code.checks_z
    last = last_x if basis == "X" else last_z
    for check, support in enumerate(checks):
        _append_detector(
            circuit,
            (last[check], *(data_measurements[data] for data in support)),
            (float(rounds), 0.0 if basis == "X" else 1.0, float(check)),
        )

    logicals = code.logicals_x if basis == "X" else code.logicals_z
    for logical, support in enumerate(logicals):
        _append_observable(
            circuit,
            tuple(data_measurements[data] for data in support),
            logical,
        )
    return data_measurements


def build_memory_circuit(
    basis: Basis,
    rounds: int,
    *,
    noise: NoiseModel | None = None,
    noisy_rounds: Collection[int] | None = None,
    noisy_boundaries: bool = True,
    code: CodeData | None = None,
) -> stim.Circuit:
    """Construct an X- or Z-basis Clifford memory experiment.

    Args:
        basis: Logical memory basis, ``"X"`` or ``"Z"``.
        rounds: Number of complete simultaneous syndrome rounds.
        noise: Enabled circuit-level Pauli channels.
        noisy_rounds: Syndrome round indices on which noise is enabled.  By
            default every round is noisy when ``noise`` is active.
        noisy_boundaries: Whether data preparation/readout are noisy.
        code: Optional preloaded immutable input data.
    """

    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if rounds < 1:
        raise ValueError("at least one syndrome round is required")
    code = load_code_data() if code is None else code
    noise = NoiseModel.noiseless() if noise is None else noise
    selected_rounds = (
        set(range(rounds))
        if noisy_rounds is None and noise.active
        else set(noisy_rounds or ())
    )
    if not selected_rounds.issubset(range(rounds)):
        raise ValueError("noisy_rounds contains an out-of-range index")

    circuit = stim.Circuit()
    for qubit in DATA_QUBITS:
        circuit.append("QUBIT_COORDS", [qubit], [float(qubit // 8), float(qubit % 8)])
    _prepare_data(circuit, basis, noise, noisy_boundaries and noise.active)

    previous_x: tuple[int, ...] | None = None
    previous_z: tuple[int, ...] | None = None
    for round_index in range(rounds):
        current_x, current_z = _measure_syndrome_round(
            circuit,
            code,
            noise,
            round_index in selected_rounds,
        )
        _append_round_detectors(
            circuit,
            code,
            basis,
            round_index,
            current_x,
            current_z,
            previous_x,
            previous_z,
        )
        previous_x, previous_z = current_x, current_z

    assert previous_x is not None and previous_z is not None
    _measure_data_and_close(
        circuit,
        code,
        basis,
        previous_x,
        previous_z,
        noise,
        noisy_boundaries and noise.active,
        rounds,
    )
    return circuit


def build_bulk_circuit(
    basis: Basis,
    *,
    noise: NoiseModel | None = None,
    code: CodeData | None = None,
) -> stim.Circuit:
    """Isolate faults in one extraction cycle between two ideal guard rounds."""

    noise = NoiseModel() if noise is None else noise
    return build_memory_circuit(
        basis,
        3,
        noise=noise,
        noisy_rounds={1},
        noisy_boundaries=False,
        code=code,
    )
