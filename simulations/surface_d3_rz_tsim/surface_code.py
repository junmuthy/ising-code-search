"""Algebra and canonical layout for a [[9, 1, 3]] rotated surface code."""

from __future__ import annotations

from itertools import product

import numpy as np
import stim


# These are Stim's canonical distance-three rotated-memory coordinates and IDs.
DATA_QUBITS = (1, 3, 5, 8, 10, 12, 15, 17, 19)
ANCILLA_QUBITS = (2, 9, 11, 13, 14, 16, 18, 25)
X_ANCILLAS = (2, 11, 16, 25)
Z_ANCILLAS = (9, 13, 14, 18)

QUBIT_COORDS = {
    1: (1, 1),
    2: (2, 0),
    3: (3, 1),
    5: (5, 1),
    8: (1, 3),
    9: (2, 2),
    10: (3, 3),
    11: (4, 2),
    12: (5, 3),
    13: (6, 2),
    14: (0, 4),
    15: (1, 5),
    16: (2, 4),
    17: (3, 5),
    18: (4, 4),
    19: (5, 5),
    25: (4, 6),
}

# Supports inferred from Stim's canonical four-layer extraction schedule.
X_CHECKS = (
    (1, 3),
    (3, 5, 10, 12),
    (8, 10, 15, 17),
    (17, 19),
)
Z_CHECKS = (
    (1, 3, 8, 10),
    (5, 12),
    (8, 15),
    (10, 12, 17, 19),
)

LOGICAL_X = (1, 8, 15)
LOGICAL_Z = (1, 3, 5)

# i X_L Z_L, with the phase chosen so that the operator is Hermitian.
LOGICAL_Y_AXES = {1: "Y", 3: "Z", 5: "Z", 8: "X", 15: "X"}

_DATA_TO_DENSE = {qubit: index for index, qubit in enumerate(DATA_QUBITS)}


def _binary_row(support: tuple[int, ...]) -> np.ndarray:
    row = np.zeros(len(DATA_QUBITS), dtype=np.uint8)
    for qubit in support:
        row[_DATA_TO_DENSE[qubit]] = 1
    return row


HX = np.array([_binary_row(support) for support in X_CHECKS], dtype=np.uint8)
HZ = np.array([_binary_row(support) for support in Z_CHECKS], dtype=np.uint8)
LX = _binary_row(LOGICAL_X)
LZ = _binary_row(LOGICAL_Z)


def gf2_rank(matrix: np.ndarray) -> int:
    """Return the row rank of a binary matrix."""

    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivot = next(
            (row for row in range(rank, work.shape[0]) if work[row, column]),
            None,
        )
        if pivot is None:
            continue
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def _in_row_span(vector: np.ndarray, generators: np.ndarray) -> bool:
    return gf2_rank(np.vstack([generators, vector])) == gf2_rank(generators)


def css_distance() -> tuple[int, int, int]:
    """Compute exact X, Z, and overall distances by enumerating 2^9 vectors."""

    best_x = len(DATA_QUBITS) + 1
    best_z = len(DATA_QUBITS) + 1
    for bits in product((0, 1), repeat=len(DATA_QUBITS)):
        vector = np.asarray(bits, dtype=np.uint8)
        weight = int(vector.sum())
        if weight == 0:
            continue
        if not np.any((HZ @ vector) % 2) and not _in_row_span(vector, HX):
            best_x = min(best_x, weight)
        if not np.any((HX @ vector) % 2) and not _in_row_span(vector, HZ):
            best_z = min(best_z, weight)
    return best_x, best_z, min(best_x, best_z)


def dense_pauli(
    *, axis: str | None = None, support: tuple[int, ...] = (), axes: dict[int, str] | None = None
) -> str:
    """Return a dense nine-character Stim Pauli string."""

    paulis = ["_"] * len(DATA_QUBITS)
    if axes is not None:
        for qubit, local_axis in axes.items():
            paulis[_DATA_TO_DENSE[qubit]] = local_axis
    elif axis is not None:
        for qubit in support:
            paulis[_DATA_TO_DENSE[qubit]] = axis
    else:
        raise ValueError("Specify either axis/support or axes")
    return "".join(paulis)


def physical_pauli_text(
    *, axis: str | None = None, support: tuple[int, ...] = (), axes: dict[int, str] | None = None
) -> str:
    """Return a Stim MPP product using the canonical sparse physical IDs."""

    if axes is not None:
        terms = [f"{axes[qubit]}{qubit}" for qubit in sorted(axes)]
    elif axis is not None:
        terms = [f"{axis}{qubit}" for qubit in support]
    else:
        raise ValueError("Specify either axis/support or axes")
    return "*".join(terms)


def logical_pauli_text(basis: str) -> str:
    basis = basis.upper()
    if basis == "X":
        return physical_pauli_text(axis="X", support=LOGICAL_X)
    if basis == "Y":
        return physical_pauli_text(axes=LOGICAL_Y_AXES)
    if basis == "Z":
        return physical_pauli_text(axis="Z", support=LOGICAL_Z)
    raise ValueError(f"Unknown logical basis {basis!r}")


def encoded_plus_dense_circuit() -> stim.Circuit:
    """Synthesize an exact Clifford preparation of the all-+1 |+>_L state."""

    generators = [
        *[stim.PauliString(dense_pauli(axis="X", support=s)) for s in X_CHECKS],
        *[stim.PauliString(dense_pauli(axis="Z", support=s)) for s in Z_CHECKS],
        stim.PauliString(dense_pauli(axis="X", support=LOGICAL_X)),
    ]
    tableau = stim.Tableau.from_stabilizers(
        generators,
        allow_redundant=False,
        allow_underconstrained=False,
    )
    return tableau.to_circuit(method="elimination")


def encoded_plus_physical_circuit() -> stim.Circuit:
    """Return the exact |+>_L encoder remapped to canonical physical IDs."""

    physical = stim.Circuit()
    for instruction in encoded_plus_dense_circuit().flattened():
        targets = instruction.targets_copy()
        if not all(target.is_qubit_target for target in targets):
            raise ValueError(f"Unexpected non-qubit encoder target in {instruction}")
        mapped_targets = [DATA_QUBITS[target.value] for target in targets]
        physical.append(
            instruction.name,
            mapped_targets,
            instruction.gate_args_copy(),
        )
    return physical


def validate_code() -> dict[str, object]:
    """Return an exact algebraic and encoder validation report."""

    rank_x = gf2_rank(HX)
    rank_z = gf2_rank(HZ)
    distance_x, distance_z, distance = css_distance()
    checks_commute = not np.any((HX @ HZ.T) % 2)
    logicals_commute_with_checks = bool(
        not np.any((HZ @ LX) % 2) and not np.any((HX @ LZ) % 2)
    )
    logicals_anticommute = bool(int(LX @ LZ) % 2 == 1)

    simulator = stim.TableauSimulator()
    simulator.do_circuit(encoded_plus_dense_circuit())
    expected_generators = [
        *[stim.PauliString(dense_pauli(axis="X", support=s)) for s in X_CHECKS],
        *[stim.PauliString(dense_pauli(axis="Z", support=s)) for s in Z_CHECKS],
        stim.PauliString(dense_pauli(axis="X", support=LOGICAL_X)),
    ]
    encoder_expectations = [
        simulator.peek_observable_expectation(generator)
        for generator in expected_generators
    ]
    encoder_valid = all(value == 1 for value in encoder_expectations)

    report = {
        "n": len(DATA_QUBITS),
        "k": len(DATA_QUBITS) - rank_x - rank_z,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "checks_commute": checks_commute,
        "logicals_commute_with_checks": logicals_commute_with_checks,
        "logicals_anticommute": logicals_anticommute,
        "distance_x": distance_x,
        "distance_z": distance_z,
        "distance": distance,
        "encoder_expectations": encoder_expectations,
        "encoder_valid": encoder_valid,
    }
    required_truths = (
        checks_commute,
        logicals_commute_with_checks,
        logicals_anticommute,
        encoder_valid,
        report["k"] == 1,
        distance == 3,
    )
    if not all(required_truths):
        raise AssertionError(f"Rotated surface-code validation failed: {report}")
    return report
