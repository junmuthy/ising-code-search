"""Exact algebra for direct one-block codes on ``C28 x C4``.

The decomposition ``C28 = C7 x C4`` gives a trivial binary sector and two
reciprocal ``GF(8)[C4 x C4]`` sectors.  An ideal in one reciprocal sector,
together with its annihilator in the other, embeds as a rank-48 binary
self-orthogonal ideal.  The untouched trivial sector contains the canonical
16 weight-seven logical fibres.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

GF8_MODULUS = 0b1011
GF8_SIZE = 8
QUOTIENT_X_ORDER = 4
QUOTIENT_Y_ORDER = 4
THICKNESS = 7
LONG_ORDER = QUOTIENT_X_ORDER * THICKNESS
GROUP_ORDER = LONG_ORDER * QUOTIENT_Y_ORDER
QUOTIENT_ORDER = QUOTIENT_X_ORDER * QUOTIENT_Y_ORDER
CHECK_RANK = 3 * QUOTIENT_ORDER


def _gf8_multiply(left: int, right: int) -> int:
    product = 0
    aa, bb = int(left), int(right)
    while bb:
        if bb & 1:
            product ^= aa
        bb >>= 1
        aa <<= 1
        if aa & 0b1000:
            aa ^= GF8_MODULUS
    return product & 0b111


GF8_MUL = np.asarray(
    [[_gf8_multiply(left, right) for right in range(8)] for left in range(8)],
    dtype=np.uint8,
)
GF8_INV = np.zeros(8, dtype=np.uint8)
for _value in range(1, 8):
    for _candidate in range(1, 8):
        if GF8_MUL[_value, _candidate] == 1:
            GF8_INV[_value] = _candidate
            break


def gf8_scale(values: np.ndarray[Any, Any], scalar: int) -> np.ndarray[Any, Any]:
    return GF8_MUL[int(scalar), np.asarray(values, dtype=np.uint8)]


def gf8_rref(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    work = np.asarray(matrix, dtype=np.uint8).copy()
    row = 0
    pivots: list[int] = []
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        work[[row, pivot]] = work[[pivot, row]]
        work[row] = gf8_scale(work[row], int(GF8_INV[work[row, column]]))
        for other in range(work.shape[0]):
            if other != row and work[other, column]:
                work[other] ^= gf8_scale(work[row], int(work[other, column]))
        pivots.append(column)
        row += 1
        if row == work.shape[0]:
            break
    return work[:row], tuple(pivots)


def gf8_nullspace(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    rref, pivots = gf8_rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for row, free_column in enumerate(free):
        basis[row, free_column] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            basis[row, pivot_column] = rref[pivot_row, free_column]
    return basis


def gf8_contains(rref_basis: np.ndarray[Any, Any], vector: np.ndarray[Any, Any]) -> bool:
    work = np.asarray(vector, dtype=np.uint8).copy()
    for row in rref_basis:
        pivot = int(np.flatnonzero(row)[0])
        if work[pivot]:
            work ^= gf8_scale(row, int(work[pivot]))
    return not np.any(work)


def gf8_subspace_key(matrix: np.ndarray[Any, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in gf8_rref(matrix)[0])


def quotient_index(power_x: int, power_y: int) -> int:
    return (power_x % QUOTIENT_X_ORDER) * QUOTIENT_Y_ORDER + power_y % QUOTIENT_Y_ORDER


def quotient_coordinates(index: int) -> tuple[int, int]:
    return divmod(index, QUOTIENT_Y_ORDER)


def quotient_translate(polynomial: np.ndarray[Any, Any], shift_x: int, shift_y: int) -> np.ndarray[Any, Any]:
    source = np.asarray(polynomial, dtype=np.uint8).reshape(
        QUOTIENT_X_ORDER, QUOTIENT_Y_ORDER
    )
    return np.roll(source, (shift_x, shift_y), axis=(0, 1)).reshape(-1)


def quotient_multiplication_matrix(polynomial: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [
            quotient_translate(polynomial, shift_x, shift_y)
            for shift_x in range(QUOTIENT_X_ORDER)
            for shift_y in range(QUOTIENT_Y_ORDER)
        ],
        dtype=np.uint8,
    ).T


def quotient_ideal_basis(generators: Sequence[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    if not generators:
        return np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    rows = [
        quotient_translate(generator, shift_x, shift_y)
        for generator in generators
        for shift_x in range(QUOTIENT_X_ORDER)
        for shift_y in range(QUOTIENT_Y_ORDER)
    ]
    return gf8_rref(np.asarray(rows, dtype=np.uint8))[0]


def quotient_annihilator_basis(generators: Sequence[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    if not generators:
        return np.eye(QUOTIENT_ORDER, dtype=np.uint8)
    equations = np.vstack([quotient_multiplication_matrix(row) for row in generators])
    return gf8_rref(gf8_nullspace(equations))[0]


def _binomial_support(exponent: int) -> tuple[int, ...]:
    return tuple(power for power in range(exponent + 1) if power & ~exponent == 0)


def nilpotent_monomial(power_u: int, power_v: int) -> np.ndarray[Any, Any]:
    polynomial = np.zeros(QUOTIENT_ORDER, dtype=np.uint8)
    for power_x in _binomial_support(power_u):
        for power_y in _binomial_support(power_v):
            polynomial[quotient_index(power_x, power_y)] ^= 1
    return polynomial


def iter_monomial_ideal_thresholds() -> Iterator[tuple[int, int, int, int]]:
    """Enumerate all 70 monomial ideals of ``GF(8)[U,V]/(U^4,V^4)``."""
    for nondecreasing in itertools.combinations_with_replacement(range(5), 4):
        yield tuple(reversed(nondecreasing))


def monomial_ideal_basis(thresholds: Sequence[int]) -> np.ndarray[Any, Any]:
    rows = [
        nilpotent_monomial(power_u, power_v)
        for power_v, threshold in enumerate(thresholds)
        for power_u in range(int(threshold), QUOTIENT_X_ORDER)
    ]
    if not rows:
        return np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    return gf8_rref(np.asarray(rows, dtype=np.uint8))[0]


def canonical_sparse_polynomial(polynomial: np.ndarray[Any, Any]) -> tuple[int, ...]:
    values = np.asarray(polynomial, dtype=np.uint8)
    normalized = []
    for index in np.flatnonzero(values):
        power_x, power_y = quotient_coordinates(int(index))
        shifted = quotient_translate(values, -power_x, -power_y)
        scaled = gf8_scale(shifted, int(GF8_INV[shifted[0]]))
        normalized.append(tuple(map(int, scaled)))
    return min(normalized) if normalized else tuple([0] * QUOTIENT_ORDER)


@dataclass(frozen=True)
class SparsePolynomial:
    family: str
    coefficients: tuple[int, ...]

    @property
    def vector(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.coefficients, dtype=np.uint8)

    @property
    def support_size(self) -> int:
        return sum(value != 0 for value in self.coefficients)

    @property
    def support(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (*quotient_coordinates(index), coefficient)
            for index, coefficient in enumerate(self.coefficients)
            if coefficient
        )

    @property
    def candidate_id(self) -> str:
        terms = "-".join(f"{xx}.{yy}.{cc}" for xx, yy, cc in self.support)
        return f"{self.family}-{terms}"


def iter_sparse_polynomials() -> Iterator[SparsePolynomial]:
    """All normalized nonunit monomials/binomials/trinomials through weight 3."""
    seen: set[tuple[int, ...]] = set()
    unit = np.zeros(QUOTIENT_ORDER, dtype=np.uint8)
    unit[0] = 1
    key = canonical_sparse_polynomial(unit)
    seen.add(key)
    yield SparsePolynomial("monomial", key)
    for position in range(1, QUOTIENT_ORDER):
        polynomial = unit.copy()
        polynomial[position] = 1
        key = canonical_sparse_polynomial(polynomial)
        if key not in seen:
            seen.add(key)
            yield SparsePolynomial("binomial", key)
    for position_one, position_two in itertools.combinations(range(1, QUOTIENT_ORDER), 2):
        for coefficient_one in range(2, GF8_SIZE):
            polynomial = unit.copy()
            polynomial[position_one] = coefficient_one
            polynomial[position_two] = 1 ^ coefficient_one
            key = canonical_sparse_polynomial(polynomial)
            if key not in seen:
                seen.add(key)
                yield SparsePolynomial("trinomial", key)


def _cyclic_binary_convolve(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    output = np.zeros(THICKNESS, dtype=np.uint8)
    for left_index in np.flatnonzero(left):
        for right_index in np.flatnonzero(right):
            output[(int(left_index) + int(right_index)) % THICKNESS] ^= 1
    return output


C_PLUS = np.zeros(THICKNESS, dtype=np.uint8)
C_PLUS[[0, 1, 2, 4]] = 1
COEFFICIENT_WORDS = np.zeros((GF8_SIZE, THICKNESS), dtype=np.uint8)
for _coefficient in range(GF8_SIZE):
    bits = np.asarray([(_coefficient >> power) & 1 for power in range(THICKNESS)], dtype=np.uint8)
    COEFFICIENT_WORDS[_coefficient] = _cyclic_binary_convolve(bits, C_PLUS)


def embed_plus(polynomial: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    output = np.zeros(GROUP_ORDER, dtype=np.uint8)
    for index in np.flatnonzero(polynomial):
        quotient_x, quotient_y = quotient_coordinates(int(index))
        coefficient = int(polynomial[index])
        for thickness_power in np.flatnonzero(COEFFICIENT_WORDS[coefficient]):
            physical_x = (21 * quotient_x + 4 * int(thickness_power)) % LONG_ORDER
            output[physical_x * QUOTIENT_Y_ORDER + quotient_y] ^= 1
    return output


def physical_dagger(vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    source = np.asarray(vector, dtype=np.uint8).reshape(LONG_ORDER, QUOTIENT_Y_ORDER)
    output = np.zeros_like(source)
    for power_x in range(LONG_ORDER):
        for power_y in range(QUOTIENT_Y_ORDER):
            output[-power_x % LONG_ORDER, -power_y % QUOTIENT_Y_ORDER] = source[power_x, power_y]
    return output.reshape(-1)


def gf2_rref(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row = 0
    pivots: list[int] = []
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        work[[row, pivot]] = work[[pivot, row]]
        for other in range(work.shape[0]):
            if other != row and work[other, column]:
                work[other] ^= work[row]
        pivots.append(column)
        row += 1
        if row == work.shape[0]:
            break
    return work[:row], tuple(pivots)


def physical_sector_basis(quotient_basis: np.ndarray[Any, Any], *, dagger: bool) -> np.ndarray[Any, Any]:
    rows = []
    for quotient_row in quotient_basis:
        for scalar in (1, 2, 4):
            physical = embed_plus(gf8_scale(quotient_row, scalar))
            rows.append(physical_dagger(physical) if dagger else physical)
    if not rows:
        return np.zeros((0, GROUP_ORDER), dtype=np.uint8)
    return gf2_rref(np.asarray(rows, dtype=np.uint8))[0]


def build_stabilizer_basis(ideal_basis: np.ndarray[Any, Any], annihilator_basis: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    plus = physical_sector_basis(ideal_basis, dagger=False)
    minus = physical_sector_basis(annihilator_basis, dagger=True)
    return gf2_rref(np.vstack([plus, minus]))[0]


def fibre_logicals() -> np.ndarray[Any, Any]:
    logicals = np.zeros((QUOTIENT_ORDER, GROUP_ORDER), dtype=np.uint8)
    for logical_x in range(QUOTIENT_X_ORDER):
        for logical_y in range(QUOTIENT_Y_ORDER):
            row = quotient_index(logical_x, logical_y)
            for thickness in range(THICKNESS):
                physical_x = logical_x + QUOTIENT_X_ORDER * thickness
                logicals[row, physical_x * QUOTIENT_Y_ORDER + logical_y] = 1
    return logicals


def validate_stabilizer(stabilizer: np.ndarray[Any, Any]) -> dict[str, bool | int]:
    logicals = fibre_logicals()
    combined_rank = len(gf2_rref(np.vstack([stabilizer, logicals]))[0])
    return {
        "rank": int(len(stabilizer)),
        "self_orthogonal": bool(not np.any(stabilizer @ stabilizer.T % 2)),
        "fibres_commute": bool(not np.any(stabilizer @ logicals.T % 2)),
        "fibres_orthonormal": bool(
            np.array_equal(logicals @ logicals.T % 2, np.eye(QUOTIENT_ORDER, dtype=np.uint8))
        ),
        "fibres_complete": combined_rank - len(stabilizer) == QUOTIENT_ORDER,
    }


def physical_seed_from_sparse(polynomial: SparsePolynomial, *, dagger: bool) -> np.ndarray[Any, Any]:
    seed = embed_plus(polynomial.vector)
    return physical_dagger(seed) if dagger else seed


def displacement_subgroup_size(seeds: Sequence[np.ndarray[Any, Any]]) -> int:
    generators: set[tuple[int, int]] = set()
    for seed in seeds:
        support = [divmod(int(index), QUOTIENT_Y_ORDER) for index in np.flatnonzero(seed)]
        if not support:
            continue
        anchor_x, anchor_y = support[0]
        generators.update(
            ((power_x - anchor_x) % LONG_ORDER, (power_y - anchor_y) % QUOTIENT_Y_ORDER)
            for power_x, power_y in support[1:]
        )
    signed = generators | {
        (-power_x % LONG_ORDER, -power_y % QUOTIENT_Y_ORDER)
        for power_x, power_y in generators
    }
    reached = {(0, 0)}
    frontier = [(0, 0)]
    while frontier:
        current_x, current_y = frontier.pop()
        for delta_x, delta_y in signed:
            target = (
                (current_x + delta_x) % LONG_ORDER,
                (current_y + delta_y) % QUOTIENT_Y_ORDER,
            )
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return len(reached)
