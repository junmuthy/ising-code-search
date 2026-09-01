"""Complete systematic parameterization of balanced `[[n,2,d]]` CSS codes."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import z3

from distance_first_c2.search import CSSState, canonical_basis


@dataclass(frozen=True)
class CSector:
    """One `C`-matrix orbit under row permutations and column exchange."""

    counts: tuple[int, int, int, int]
    raw_multiplicity: int

    @property
    def name(self) -> str:
        return "c-" + "-".join(map(str, self.counts))

    @property
    def matrix(self) -> tuple[tuple[int, int], ...]:
        patterns = ((0, 0), (0, 1), (1, 0), (1, 1))
        return tuple(
            pattern
            for pattern, count in zip(patterns, self.counts)
            for _ in range(count)
        )


def canonical_c_sectors(rank: int) -> tuple[CSector, ...]:
    """Enumerate complete canonical `C` sectors for `k=2`."""
    sectors = []
    for count_00 in range(rank + 1):
        for count_01 in range(rank - count_00 + 1):
            for count_10 in range(rank - count_00 - count_01 + 1):
                count_11 = rank - count_00 - count_01 - count_10
                counts = (count_00, count_01, count_10, count_11)
                swapped = (count_00, count_10, count_01, count_11)
                if counts > swapped:
                    continue
                row_permutations = math.factorial(rank)
                for count in counts:
                    row_permutations //= math.factorial(count)
                column_orbit = 1 if count_01 == count_10 else 2
                sectors.append(CSector(counts, row_permutations * column_orbit))
    return tuple(sectors)


def masks_of_weight(n: int, weight: int) -> Iterator[int]:
    for support in itertools.combinations(range(n), weight):
        yield sum(1 << bit for bit in support)


def systematic_state(
    a: Sequence[Sequence[int]], c: Sequence[Sequence[int]]
) -> CSSState:
    """Construct `H_X=[I|A]` and the orthogonal `H_Z` from `B=[I|C]`."""
    rank = len(a)
    if not rank or len(c) != rank:
        raise ValueError("A and C must have the same positive row count")
    logicals = len(c[0])
    free = rank + logicals
    if any(len(row) != free for row in a):
        raise ValueError("A must have rank+k columns")
    if any(len(row) != logicals for row in c):
        raise ValueError("C must have k columns")
    n = 2 * rank + logicals
    rows_x = []
    for index, row in enumerate(a):
        mask = 1 << index
        for column, bit in enumerate(row):
            if int(bit) & 1:
                mask |= 1 << (rank + column)
        rows_x.append(mask)
    rows_z = []
    for index in range(rank):
        b = [int(column == index) for column in range(rank)] + [
            int(bit) & 1 for bit in c[index]
        ]
        p = [
            sum((int(a[row][column]) & b[column]) for column in range(free)) & 1
            for row in range(rank)
        ]
        mask = sum(bit << row for row, bit in enumerate(p))
        mask |= 1 << (rank + index)
        for logical, bit in enumerate(c[index]):
            if int(bit) & 1:
                mask |= 1 << (2 * rank + logical)
        rows_z.append(mask)
    return CSSState(
        n,
        canonical_basis(rows_x, n),
        canonical_basis(rows_z, n),
    )


def xor_expression(terms: Iterable[z3.BoolRef], constant: bool = False) -> z3.BoolRef:
    output: z3.BoolRef = z3.BoolVal(constant)
    for term in terms:
        output = z3.Xor(output, term)
    return output


class SystematicCSSSolver:
    """Exact distance constraints for all CSS codes, up to qubit permutation."""

    def __init__(
        self,
        n: int,
        *,
        timeout_ms: int = 0,
        fixed_c: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if n % 2 or n < 4:
            raise ValueError("balanced k=2 CSS codes require even n")
        self.n = n
        self.k = 2
        self.rank = (n - self.k) // 2
        self.free = self.rank + self.k
        self.a = [
            [z3.Bool(f"a_{row}_{column}") for column in range(self.free)]
            for row in range(self.rank)
        ]
        if fixed_c is not None:
            if len(fixed_c) != self.rank or any(
                len(row) != self.k for row in fixed_c
            ):
                raise ValueError("fixed C must have shape rank by k")
            self.c = [
                [z3.BoolVal(bool(int(bit) & 1)) for bit in row] for row in fixed_c
            ]
        else:
            self.c = [
                [z3.Bool(f"c_{row}_{logical}") for logical in range(self.k)]
                for row in range(self.rank)
            ]
        self.solver = z3.Solver()
        if timeout_ms:
            self.solver.set(timeout=timeout_ms)
        # The P part of H_Z is B A^T, where B=[I|C].
        self.z_p = [
            [
                xor_expression(
                    [self.a[pivot][check]]
                    + [
                        z3.And(
                            self.a[pivot][self.rank + logical],
                            self.c[check][logical],
                        )
                        for logical in range(self.k)
                    ]
                )
                for pivot in range(self.rank)
            ]
            for check in range(self.rank)
        ]
        self.constraints_by_weight: dict[int, int] = {}

    def _syndrome_x(self, operator: int) -> list[z3.BoolRef]:
        p = [(operator >> bit) & 1 for bit in range(self.rank)]
        q = [
            (operator >> (self.rank + column)) & 1
            for column in range(self.free)
        ]
        return [
            xor_expression(
                [self.a[row][column] for column in range(self.free) if q[column]],
                bool(p[row]),
            )
            for row in range(self.rank)
        ]

    def _syndrome_z(self, operator: int) -> list[z3.BoolRef]:
        p = [(operator >> bit) & 1 for bit in range(self.rank)]
        q = [
            (operator >> (self.rank + column)) & 1
            for column in range(self.free)
        ]
        output = []
        for check in range(self.rank):
            terms = [self.z_p[check][pivot] for pivot in range(self.rank) if p[pivot]]
            terms.extend(
                self.c[check][logical]
                for logical in range(self.k)
                if q[self.rank + logical]
            )
            output.append(xor_expression(terms, bool(q[check])))
        return output

    def _in_stabilizer_x(self, operator: int) -> z3.BoolRef:
        p = [(operator >> bit) & 1 for bit in range(self.rank)]
        q = [
            (operator >> (self.rank + column)) & 1
            for column in range(self.free)
        ]
        return z3.And(
            *[
                xor_expression(
                    [self.a[row][column] for row in range(self.rank) if p[row]]
                )
                == z3.BoolVal(bool(q[column]))
                for column in range(self.free)
            ]
        )

    def _in_stabilizer_z_when_commuting(self, operator: int) -> z3.BoolRef:
        q = [
            (operator >> (self.rank + column)) & 1
            for column in range(self.free)
        ]
        # Within ker(H_X), membership in S_Z is equivalent to q lying in
        # row([I|C]); the P coordinates are then forced by orthogonality.
        return z3.And(
            *[
                xor_expression(
                    [self.c[row][logical] for row in range(self.rank) if q[row]]
                )
                == z3.BoolVal(bool(q[self.rank + logical]))
                for logical in range(self.k)
            ]
        )

    def add_operator_constraint(self, operator: int) -> None:
        self.solver.add(
            z3.Or(*self._syndrome_z(operator), self._in_stabilizer_x(operator))
        )
        self.solver.add(
            z3.Or(
                *self._syndrome_x(operator),
                self._in_stabilizer_z_when_commuting(operator),
            )
        )

    def add_weight(self, weight: int) -> int:
        count = 0
        for operator in masks_of_weight(self.n, weight):
            self.add_operator_constraint(operator)
            count += 1
        self.constraints_by_weight[weight] = count
        return count

    def check(self) -> z3.CheckSatResult:
        return self.solver.check()

    def set_timeout(self, timeout_ms: int) -> None:
        self.solver.set(timeout=max(1, int(timeout_ms)))

    def state(self) -> CSSState:
        model = self.solver.model()
        a = [
            [int(z3.is_true(model.eval(bit, model_completion=True))) for bit in row]
            for row in self.a
        ]
        c = [
            [int(z3.is_true(model.eval(bit, model_completion=True))) for bit in row]
            for row in self.c
        ]
        return systematic_state(a, c)
