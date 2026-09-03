"""Exact low-weight joint Pauli decoder for terminal teleportation readout.

The Fig. 10 estimator measures the target patch in Z and the resource patch in
X.  An X component on the resource propagates through the ideal transversal
CNOT and flips target-Z outcomes; a Z component flips resource-X outcomes.
Decoding both strings together therefore amounts to decoding one Pauli error
against the combined CSS syndrome.

For the small ``[[22,2,6]]`` code we do not need an external MILP solver.  The
decoder enumerates Pauli corrections in increasing symplectic weight, but only
until every syndrome actually observed in a sample batch has been resolved.
This gives the exact minimum-weight class for those syndromes.  Degenerate
minimum-weight classes are resolved by maximum multiplicity and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np

from n22_k2_d6_shortened_golay.tmr_postselection.model import (
    CodeArtifact,
    independent_indices,
)


@dataclass(frozen=True)
class CorrectionClass:
    syndrome: int
    weight: int
    logical_signature: int
    degeneracy: int
    ambiguous: bool


@dataclass(frozen=True)
class DecodedBatch:
    logical_z: np.ndarray
    logical_x: np.ndarray
    syndrome: np.ndarray
    correction_weight: np.ndarray
    ambiguous: np.ndarray


def _row_masks(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(
        sum(1 << int(qubit) for qubit in np.flatnonzero(row))
        for row in np.asarray(matrix, dtype=np.uint8)
    )


def _parity(value: int) -> int:
    return value.bit_count() & 1


class JointPauliDecoder:
    """Lazy exact minimum-symplectic-weight decoder for observed syndromes."""

    def __init__(self, code: CodeArtifact, *, max_search_weight: int = 5) -> None:
        self.code = code
        self.num_data = int(code.matrix_x.shape[1])
        self.num_logicals = int(code.logical_x.shape[0])
        self.max_search_weight = int(max_search_weight)

        self.z_check_rows = independent_indices(code.matrix_z)
        self.x_check_rows = independent_indices(code.matrix_x)
        self.z_checks = np.asarray(code.matrix_z[list(self.z_check_rows)], dtype=np.uint8)
        self.x_checks = np.asarray(code.matrix_x[list(self.x_check_rows)], dtype=np.uint8)
        self.num_z_syndromes = len(self.z_check_rows)
        self.num_x_syndromes = len(self.x_check_rows)

        z_check_masks = _row_masks(self.z_checks)
        x_check_masks = _row_masks(self.x_checks)
        logical_z_masks = _row_masks(code.logical_z)
        logical_x_masks = _row_masks(code.logical_x)

        self._x_syndrome_by_qubit: list[int] = []
        self._z_syndrome_by_qubit: list[int] = []
        self._x_signature_by_qubit: list[int] = []
        self._z_signature_by_qubit: list[int] = []
        for qubit in range(self.num_data):
            bit = 1 << qubit
            x_syndrome = sum(
                _parity(bit & mask) << row for row, mask in enumerate(z_check_masks)
            )
            z_syndrome = sum(
                _parity(bit & mask) << (self.num_z_syndromes + row)
                for row, mask in enumerate(x_check_masks)
            )
            x_signature = sum(
                _parity(bit & mask) << logical for logical, mask in enumerate(logical_z_masks)
            )
            z_signature = sum(
                _parity(bit & mask) << (self.num_logicals + logical)
                for logical, mask in enumerate(logical_x_masks)
            )
            self._x_syndrome_by_qubit.append(x_syndrome)
            self._z_syndrome_by_qubit.append(z_syndrome)
            self._x_signature_by_qubit.append(x_signature)
            self._z_signature_by_qubit.append(z_signature)

        self._cache: dict[int, CorrectionClass] = {
            0: CorrectionClass(0, 0, 0, 1, False)
        }

    @property
    def cached_syndromes(self) -> int:
        return len(self._cache)

    def syndrome_and_signature(self, x_mask: int, z_mask: int) -> tuple[int, int]:
        syndrome = 0
        signature = 0
        for qubit in range(self.num_data):
            bit = 1 << qubit
            if x_mask & bit:
                syndrome ^= self._x_syndrome_by_qubit[qubit]
                signature ^= self._x_signature_by_qubit[qubit]
            if z_mask & bit:
                syndrome ^= self._z_syndrome_by_qubit[qubit]
                signature ^= self._z_signature_by_qubit[qubit]
        return syndrome, signature

    def _enumerate_weight(
        self, weight: int, unresolved: set[int]
    ) -> dict[int, dict[int, int]]:
        counts: dict[int, dict[int, int]] = {}
        if not unresolved:
            return counts
        for support in combinations(range(self.num_data), weight):
            for paulis in product((1, 2, 3), repeat=weight):
                syndrome = 0
                signature = 0
                for qubit, pauli in zip(support, paulis, strict=True):
                    if pauli & 1:  # X or Y
                        syndrome ^= self._x_syndrome_by_qubit[qubit]
                        signature ^= self._x_signature_by_qubit[qubit]
                    if pauli & 2:  # Z or Y
                        syndrome ^= self._z_syndrome_by_qubit[qubit]
                        signature ^= self._z_signature_by_qubit[qubit]
                if syndrome not in unresolved:
                    continue
                classes = counts.setdefault(syndrome, {})
                classes[signature] = classes.get(signature, 0) + 1
        return counts

    def ensure(self, syndromes: Iterable[int]) -> None:
        unresolved = {int(value) for value in syndromes if int(value) not in self._cache}
        if not unresolved:
            return
        # Start at weight one for every newly observed set.  Earlier searches
        # deliberately enumerate only requested syndromes, so skipping an old
        # weight could miss a newly encountered low-weight syndrome.
        for weight in range(1, self.max_search_weight + 1):
            found = self._enumerate_weight(weight, unresolved)
            for syndrome, classes in found.items():
                best_degeneracy = max(classes.values())
                best_signatures = sorted(
                    signature
                    for signature, multiplicity in classes.items()
                    if multiplicity == best_degeneracy
                )
                self._cache[syndrome] = CorrectionClass(
                    syndrome=syndrome,
                    weight=weight,
                    logical_signature=best_signatures[0],
                    degeneracy=best_degeneracy,
                    ambiguous=len(best_signatures) > 1,
                )
            unresolved.difference_update(found)
            if not unresolved:
                return
        preview = ", ".join(hex(value) for value in sorted(unresolved)[:8])
        raise ValueError(
            f"{len(unresolved)} observed syndromes need correction weight above "
            f"{self.max_search_weight}; first values: {preview}"
        )

    def _syndrome_integers(self, terminal_z: np.ndarray, terminal_x: np.ndarray) -> np.ndarray:
        z_bits = np.asarray(terminal_z, dtype=np.uint8)
        x_bits = np.asarray(terminal_x, dtype=np.uint8)
        z_syndrome = (z_bits @ self.z_checks.T) & 1
        x_syndrome = (x_bits @ self.x_checks.T) & 1
        z_powers = (1 << np.arange(self.num_z_syndromes, dtype=np.uint32))
        x_powers = (
            1 << (self.num_z_syndromes + np.arange(self.num_x_syndromes, dtype=np.uint32))
        )
        return (
            z_syndrome.astype(np.uint32) @ z_powers
            ^ x_syndrome.astype(np.uint32) @ x_powers
        ).astype(np.uint32)

    def decode(self, terminal_z: np.ndarray, terminal_x: np.ndarray) -> DecodedBatch:
        terminal_z = np.asarray(terminal_z, dtype=np.uint8)
        terminal_x = np.asarray(terminal_x, dtype=np.uint8)
        if terminal_z.ndim != 2 or terminal_z.shape[1] != self.num_data:
            raise ValueError("terminal Z records must have shape (shots, 22)")
        if terminal_x.shape != terminal_z.shape:
            raise ValueError("terminal X records must have the same shape as terminal Z records")

        syndrome = self._syndrome_integers(terminal_z, terminal_x)
        self.ensure(np.unique(syndrome).tolist())
        signatures = np.fromiter(
            (self._cache[int(value)].logical_signature for value in syndrome),
            dtype=np.uint8,
            count=len(syndrome),
        )
        weights = np.fromiter(
            (self._cache[int(value)].weight for value in syndrome),
            dtype=np.uint8,
            count=len(syndrome),
        )
        ambiguous = np.fromiter(
            (self._cache[int(value)].ambiguous for value in syndrome),
            dtype=np.bool_,
            count=len(syndrome),
        )

        raw_z = (terminal_z @ self.code.logical_z.T) & 1
        raw_x = (terminal_x @ self.code.logical_x.T) & 1
        z_correction = (
            signatures[:, None] >> np.arange(self.num_logicals, dtype=np.uint8)
        ) & 1
        x_correction = (
            signatures[:, None]
            >> (self.num_logicals + np.arange(self.num_logicals, dtype=np.uint8))
        ) & 1
        return DecodedBatch(
            logical_z=(raw_z ^ z_correction).astype(np.uint8),
            logical_x=(raw_x ^ x_correction).astype(np.uint8),
            syndrome=syndrome,
            correction_weight=weights,
            ambiguous=ambiguous,
        )
