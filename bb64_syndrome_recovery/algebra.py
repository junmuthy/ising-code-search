"""Exact algebra for syndrome-conditioned BB64 TMR recovery.

The 24 partial logical-Z pieces generate a 16-dimensional syndrome image.
The eight-dimensional kernel is generated exactly by the eight complete
logical-Z operators.  Consequently, every syndrome branch is a product of
eight one-logical-qubit diagonal operators after a canonical partial-product
correction; this module constructs and certifies that statement exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from bb64_tmr_postselection.circuit import physical_tmr_angle
from bb64_tmr_postselection.model import (
    BATCHES,
    CodeArtifact,
    gf2_inverse,
    gf2_rank,
    independent_indices,
    load_code_artifact,
    vector_from_support,
)
from bb64_tmr_postselection.partitions import (
    Partition,
    certify_partitions,
    load_partitions,
    syndrome_matrix,
)


NUM_LOGICALS = 8
PIECES_PER_LOGICAL = 3
NUM_PIECES = NUM_LOGICALS * PIECES_PER_LOGICAL
SYNDROME_RANK = NUM_PIECES - NUM_LOGICALS
NUM_SYNDROME_CLASSES = 1 << SYNDROME_RANK
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARTITIONS = PACKAGE_DIR.parent / "bb64_tmr_postselection" / "partitions_m3.json"


@dataclass(frozen=True)
class RecoveryModel:
    """Frozen code artifacts and the complete canonical syndrome table."""

    code: CodeArtifact
    partitions: Mapping[int, Partition]
    piece_supports: np.ndarray
    syndrome_matrix: np.ndarray
    triple_generators: np.ndarray
    independent_piece_columns: tuple[int, ...]
    independent_syndrome_rows: tuple[int, ...]
    syndrome_coordinates: np.ndarray
    displayed_syndromes: np.ndarray
    initial_piece_masks: np.ndarray
    canonical_piece_masks: np.ndarray
    correction_piece_masks: np.ndarray
    correction_physical_supports: np.ndarray
    logical_z_frame: np.ndarray
    alternative_mask: np.ndarray
    selected_piece: np.ndarray
    local_branch_labels: np.ndarray
    base4_branch_id: np.ndarray

    @property
    def syndrome_ids(self) -> np.ndarray:
        return np.arange(NUM_SYNDROME_CLASSES, dtype=np.uint32)

    @property
    def alternative_count(self) -> np.ndarray:
        return np.count_nonzero(self.alternative_mask, axis=1).astype(np.uint8)

    @property
    def correction_weight(self) -> np.ndarray:
        return np.count_nonzero(self.correction_physical_supports, axis=1).astype(np.uint8)

    @property
    def displayed_syndrome_weight(self) -> np.ndarray:
        return np.count_nonzero(self.displayed_syndromes, axis=1).astype(np.uint8)


def _all_binary_rows(width: int) -> np.ndarray:
    values = np.arange(1 << width, dtype=np.uint32)[:, None]
    shifts = np.arange(width, dtype=np.uint32)[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def _piece_support_matrix(partitions: Mapping[int, Partition]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for logical in range(NUM_LOGICALS):
        for part in partitions[logical]:
            rows.append(vector_from_support(part))
    return np.stack(rows, axis=0).astype(np.uint8)


def _triple_generators() -> np.ndarray:
    generators = np.zeros((NUM_PIECES, NUM_LOGICALS), dtype=np.uint8)
    for logical in range(NUM_LOGICALS):
        generators[3 * logical : 3 * logical + 3, logical] = 1
    return generators


def _canonicalize(initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = initial.reshape(-1, NUM_LOGICALS, PIECES_PER_LOGICAL).copy()
    frame = (np.count_nonzero(local, axis=2) > 1).astype(np.uint8)
    local ^= frame[:, :, None]
    if np.any(np.count_nonzero(local, axis=2) > 1):
        raise AssertionError("canonical TMR representative has local weight greater than one")
    return local.reshape(-1, NUM_PIECES), frame


def build_recovery_model(
    *,
    partitions_path: Path = DEFAULT_PARTITIONS,
    code: CodeArtifact | None = None,
) -> RecoveryModel:
    """Build and validate all ``2^16`` ideal syndrome classes."""

    code = load_code_artifact() if code is None else code
    partitions = load_partitions(partitions_path)
    certificate = certify_partitions(code, partitions)
    if certificate.combined_rank != SYNDROME_RANK:
        raise ValueError("the frozen TMR syndrome image does not have rank 16")

    pieces = _piece_support_matrix(partitions)
    syndromes = syndrome_matrix(code, partitions, range(NUM_LOGICALS)).astype(np.uint8)
    triples = _triple_generators()
    if gf2_rank(syndromes) != SYNDROME_RANK:
        raise ValueError("unexpected TMR syndrome rank")
    if gf2_rank(pieces) != NUM_PIECES:
        raise ValueError("partial-product operators are not physically independent")
    if np.any((syndromes @ triples) % 2):
        raise ValueError("complete logical-Z triples are not in the syndrome kernel")
    if not np.array_equal((triples.T @ pieces) % 2, code.logical_z):
        raise ValueError("TMR triples do not reproduce the frozen logical-Z basis")
    if gf2_rank(triples.T) != NUM_LOGICALS:
        raise ValueError("logical-Z triple generators are dependent")

    piece_columns = independent_indices(syndromes.T)
    if len(piece_columns) != SYNDROME_RANK:
        raise ValueError("failed to select 16 independent partial products")
    column_basis = syndromes[:, list(piece_columns)]
    syndrome_rows = independent_indices(column_basis)
    square = column_basis[list(syndrome_rows)]
    inverse = gf2_inverse(square)

    coordinates = _all_binary_rows(SYNDROME_RANK)
    selected_coefficients = (coordinates @ inverse.T) % 2
    initial = np.zeros((NUM_SYNDROME_CLASSES, NUM_PIECES), dtype=np.uint8)
    initial[:, list(piece_columns)] = selected_coefficients
    displayed = (initial @ syndromes.T) % 2
    if not np.array_equal(displayed[:, list(syndrome_rows)], coordinates):
        raise AssertionError("syndrome coordinate reconstruction failed")

    canonical, frame = _canonicalize(initial)
    if not np.array_equal((canonical @ syndromes.T) % 2, displayed):
        raise AssertionError("canonical corrections changed the syndrome")
    local = canonical.reshape(-1, NUM_LOGICALS, PIECES_PER_LOGICAL)
    alternative = np.any(local, axis=2).astype(np.uint8)
    selected = np.full((NUM_SYNDROME_CLASSES, NUM_LOGICALS), -1, dtype=np.int8)
    for piece in range(PIECES_PER_LOGICAL):
        selected[local[:, :, piece].astype(bool)] = piece
    labels = (selected + 1).astype(np.uint8)
    powers = (4 ** np.arange(NUM_LOGICALS, dtype=np.uint32))[None, :]
    base4 = np.sum(labels.astype(np.uint32) * powers, axis=1, dtype=np.uint32)
    if len(np.unique(base4)) != NUM_SYNDROME_CLASSES:
        raise AssertionError("canonical local branch labels are not a bijection")

    physical = (canonical @ pieces) % 2
    if not np.array_equal((physical @ code.matrix_x.T) % 2, displayed):
        raise AssertionError("physical correction support has the wrong syndrome")

    return RecoveryModel(
        code=code,
        partitions=partitions,
        piece_supports=pieces,
        syndrome_matrix=syndromes,
        triple_generators=triples,
        independent_piece_columns=tuple(map(int, piece_columns)),
        independent_syndrome_rows=tuple(map(int, syndrome_rows)),
        syndrome_coordinates=coordinates,
        displayed_syndromes=displayed,
        initial_piece_masks=initial,
        canonical_piece_masks=canonical,
        correction_piece_masks=canonical.copy(),
        correction_physical_supports=physical,
        logical_z_frame=frame,
        alternative_mask=alternative,
        selected_piece=selected,
        local_branch_labels=labels,
        base4_branch_id=base4,
    )


def branch_amplitudes(theta: float, local_weight: int, *, partitions: int = 3) -> tuple[complex, complex]:
    """Return the ``I`` and logical-``Z`` amplitudes of one corrected branch."""

    if not 0 <= local_weight <= partitions:
        raise ValueError("local weight is outside the TMR partition count")
    alpha = physical_tmr_angle(theta, partitions)
    cosine = math.cos(alpha / 2)
    sine = math.sin(alpha / 2)
    a = cosine ** (partitions - local_weight) * (-1j * sine) ** local_weight
    b = cosine ** local_weight * (-1j * sine) ** (partitions - local_weight)
    return a, b


def _fold_rotation_angle(angle: float) -> tuple[float, int]:
    """Fold an RZ angle into ``[-pi/2, pi/2)`` and return its Z-frame bit."""

    frame = 0
    while angle >= math.pi / 2:
        angle -= math.pi
        frame ^= 1
    while angle < -math.pi / 2:
        angle += math.pi
        frame ^= 1
    return angle, frame


def branch_data(theta: float, local_weight: int, *, partitions: int = 3) -> dict[str, float | int]:
    """Return probability, unitary angle, and frame for one local branch."""

    a, b = branch_amplitudes(theta, local_weight, partitions=partitions)
    probability = float(abs(a) ** 2 + abs(b) ** 2)
    plus = a + b
    minus = a - b
    if probability == 0 or abs(plus) == 0:
        raise ValueError("degenerate TMR branch")
    angle = float(np.angle(minus / plus))
    angle, frame = _fold_rotation_angle(angle)
    normalized = np.array([a, b], dtype=np.complex128) / math.sqrt(probability)
    unitarity_error = float(abs(np.real(normalized[0] * normalized[1].conjugate())))
    return {
        "local_weight": int(local_weight),
        "probability_per_pattern": probability,
        "logical_angle": angle,
        "logical_z_frame": int(frame),
        "unitarity_error": unitarity_error,
    }


def angle_table(theta: float) -> dict[str, object]:
    """Return the two canonical local branch types for ``M=3``."""

    target = branch_data(theta, 0)
    alternative = branch_data(theta, 1)
    return {
        "theta": float(theta),
        "physical_angle": float(physical_tmr_angle(theta, 3)),
        "target": target,
        "alternative": alternative,
        "probability_target": float(target["probability_per_pattern"]),
        "probability_any_alternative": float(3 * alternative["probability_per_pattern"]),
    }


def add_angle_columns(model: RecoveryModel, theta: float) -> dict[str, np.ndarray]:
    """Create per-class logical-angle and angle-frame arrays."""

    table = angle_table(theta)
    target = table["target"]
    alternative = table["alternative"]
    angles = np.where(
        model.alternative_mask,
        float(alternative["logical_angle"]),
        float(target["logical_angle"]),
    )
    frames = np.where(
        model.alternative_mask,
        int(alternative["logical_z_frame"]),
        int(target["logical_z_frame"]),
    ).astype(np.uint8)
    return {"logical_angles": angles.astype(np.float64), "angle_z_frames": frames}


def verify_factorization(
    model: RecoveryModel,
    theta: float,
    *,
    exhaustive: bool = True,
    batch_size: int = 8192,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Certify that all syndrome branches contain no correlated logical phase.

    The exhaustive check compares every one of the ``65536 * 256`` expansion
    amplitudes with the tensor product of its eight local amplitudes.
    """

    logical_labels = _all_binary_rows(NUM_LOGICALS)
    physical_logicals = (logical_labels @ model.triple_generators.T @ model.piece_supports) % 2
    expected_logicals = (logical_labels @ model.code.logical_z) % 2
    logical_support_error = int(np.count_nonzero(physical_logicals ^ expected_logicals))
    pairing = (physical_logicals @ model.code.logical_x.T) % 2
    pairing_error = int(np.count_nonzero(pairing ^ logical_labels))

    max_amplitude_error = 0.0
    amplitudes = {
        local_weight: branch_amplitudes(theta, local_weight)
        for local_weight in (0, 1)
    }
    checked = 0
    if exhaustive:
        local_weights = model.alternative_mask.astype(np.uint8)
        for start in range(0, NUM_SYNDROME_CLASSES, batch_size):
            stop = min(start + batch_size, NUM_SYNDROME_CLASSES)
            weights = local_weights[start:stop, :, None]
            lambdas = logical_labels.T[None, :, :]
            expanded_weights = np.where(lambdas, 3 - weights, weights)

            alpha = physical_tmr_angle(theta, 3)
            c = math.cos(alpha / 2)
            s = math.sin(alpha / 2)
            direct = np.prod(c ** (3 - expanded_weights) * (-1j * s) ** expanded_weights, axis=1)

            predicted = np.ones_like(direct, dtype=np.complex128)
            for logical in range(NUM_LOGICALS):
                is_alternative = local_weights[start:stop, logical].astype(bool)
                a = np.where(is_alternative, amplitudes[1][0], amplitudes[0][0])[:, None]
                b = np.where(is_alternative, amplitudes[1][1], amplitudes[0][1])[:, None]
                predicted *= np.where(logical_labels[:, logical][None, :], b, a)
            max_amplitude_error = max(max_amplitude_error, float(np.max(np.abs(direct - predicted))))
            checked += (stop - start) * len(logical_labels)
            if progress is not None:
                progress(f"factorization {stop:,}/{NUM_SYNDROME_CLASSES:,} syndromes; {checked:,} amplitudes")

    table = angle_table(theta)
    p0 = float(table["probability_target"])
    p1 = float(table["probability_any_alternative"])
    counts = np.bincount(model.alternative_count, minlength=NUM_LOGICALS + 1)
    probabilities = [math.comb(NUM_LOGICALS, bad) * p0 ** (NUM_LOGICALS - bad) * p1**bad for bad in range(NUM_LOGICALS + 1)]
    return {
        "num_syndrome_classes": NUM_SYNDROME_CLASSES,
        "num_logical_labels_per_syndrome": len(logical_labels),
        "num_amplitudes_checked": int(checked),
        "syndrome_rank_gf2": gf2_rank(model.syndrome_matrix),
        "kernel_dimension": NUM_PIECES - gf2_rank(model.syndrome_matrix),
        "kernel_generated_exactly_by_logical_triples": True,
        "logical_support_error_count": logical_support_error,
        "logical_pairing_error_count": pairing_error,
        "max_factorization_amplitude_error": max_amplitude_error,
        "corrected_branches_are_product_rotations": bool(
            logical_support_error == 0 and pairing_error == 0 and max_amplitude_error < 1e-13
        ),
        "local_branch_data": table,
        "class_count_by_alternative_logicals": counts.astype(int).tolist(),
        "probability_by_alternative_logicals": probabilities,
        "total_probability": float(sum(probabilities)),
        "batches": [list(batch) for batch in BATCHES],
    }
