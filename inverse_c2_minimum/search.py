"""Direct C2-invariant folded-CSS synthesis at n=12 through n=20."""

from __future__ import annotations

import itertools
from collections import Counter, deque
from dataclasses import dataclass
from functools import reduce
from operator import xor
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import z3


TARGET_K = 2
TARGET_DISTANCE = 6
SUPPORTED_LENGTHS = (12, 14, 16, 18, 20)


def compose_permutations(
    left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    """Compose source-to-target permutations as ``left after right``."""
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def permute_mask(mask: int, permutation: Sequence[int]) -> int:
    output = 0
    for source, target in enumerate(permutation):
        if mask >> source & 1:
            output |= 1 << int(target)
    return output


def mask_to_support(mask: int, n: int) -> list[int]:
    return [index for index in range(n) if mask >> index & 1]


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in np.flatnonzero(work[:, column]):
            if row != rank:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def is_permutation_matrix(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (TARGET_K, TARGET_K)
        and np.all(np.count_nonzero(matrix, axis=0) == 1)
        and np.all(np.count_nonzero(matrix, axis=1) == 1)
    )


@dataclass(frozen=True)
class LogicalGeometry:
    """Canonical disjoint logical supports and physical C2 action."""

    n: int
    logical_weight: int
    slack_transpositions: int
    slack_fixed_points: int

    def __post_init__(self) -> None:
        if self.n not in SUPPORTED_LENGTHS:
            raise ValueError(f"unsupported length {self.n}")
        if self.logical_weight < TARGET_DISTANCE:
            raise ValueError("logical support is shorter than the target distance")
        expected = (
            2 * self.logical_weight
            + 2 * self.slack_transpositions
            + self.slack_fixed_points
        )
        if expected != self.n:
            raise ValueError(f"geometry accounts for {expected} qubits, not {self.n}")

    @property
    def name(self) -> str:
        return (
            f"n{self.n}-w{self.logical_weight}-"
            f"s2-{self.slack_transpositions}-s1-{self.slack_fixed_points}"
        )

    @property
    def transposition_orbits(self) -> int:
        return self.logical_weight + self.slack_transpositions

    @property
    def translation(self) -> tuple[int, ...]:
        permutation = list(range(self.n))
        for orbit in range(self.transposition_orbits):
            left = 2 * orbit
            right = left + 1
            permutation[left], permutation[right] = right, left
        return tuple(permutation)

    @property
    def logical_z(self) -> tuple[int, int]:
        return (
            sum(1 << (2 * orbit) for orbit in range(self.logical_weight)),
            sum(1 << (2 * orbit + 1) for orbit in range(self.logical_weight)),
        )

    @property
    def active_mask(self) -> int:
        return self.logical_z[0] ^ self.logical_z[1]

    def analyze(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "logical_weight": self.logical_weight,
            "slack_transpositions": self.slack_transpositions,
            "slack_fixed_points": self.slack_fixed_points,
            "translation": list(self.translation),
            "translation_is_involution": compose_permutations(
                self.translation, self.translation
            )
            == tuple(range(self.n)),
            "logical_z": [mask_to_support(mask, self.n) for mask in self.logical_z],
            "translation_swaps_logical_z": tuple(
                permute_mask(mask, self.translation) for mask in self.logical_z
            )
            == self.logical_z[::-1],
        }


def geometry_catalog(n: int) -> tuple[LogicalGeometry, ...]:
    """Enumerate all canonical slack cycle types for equal disjoint supports."""
    output = []
    for logical_weight in range(TARGET_DISTANCE, n // 2 + 1):
        slack = n - 2 * logical_weight
        for slack_transpositions in range(slack // 2 + 1):
            slack_fixed_points = slack - 2 * slack_transpositions
            output.append(
                LogicalGeometry(
                    n=n,
                    logical_weight=logical_weight,
                    slack_transpositions=slack_transpositions,
                    slack_fixed_points=slack_fixed_points,
                )
            )
    return tuple(output)


def n12_no_go_certificate() -> dict[str, Any]:
    """Pairing-augmentation obstruction for two even supports filling n=12."""
    return {
        "n": 12,
        "k": 2,
        "distance_lower_bound": 6,
        "disjoint_support_bound": (
            "two supports of weight >=6 fill all 12 qubits and both have weight 6"
        ),
        "pairing_equation": "M_ij = z_i dot P(z_j)",
        "row_sum_equation": "sum_j M_ij = z_i dot P(1) = wt(z_i) = 0 mod 2",
        "kernel_vector": [1, 1],
        "conclusion": "every physical permutation P gives singular logical ZX pairing",
        "scope": (
            "two disjoint logicals exchanged with X logicals by transversal H plus "
            "a physical permutation"
        ),
    }


def _signed_involutions(
    num_orbits: int,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield involutions of C2 orbits, including endpoint orientations."""
    images = [-1] * num_orbits
    flips = [-1] * num_orbits

    def visit(
        remaining: tuple[int, ...],
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
        if not remaining:
            yield tuple(images), tuple(flips)
            return
        first = remaining[0]
        tail = remaining[1:]
        for orientation in (0, 1):
            images[first] = first
            flips[first] = orientation
            yield from visit(tail)
        for offset, second in enumerate(tail):
            rest = tail[:offset] + tail[offset + 1 :]
            for orientation in (0, 1):
                images[first] = second
                images[second] = first
                flips[first] = orientation
                flips[second] = orientation
                yield from visit(rest)
        images[first] = -1
        flips[first] = -1

    yield from visit(tuple(range(num_orbits)))


def _ordinary_involutions(size: int) -> Iterator[tuple[int, ...]]:
    images = [-1] * size

    def visit(remaining: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if not remaining:
            yield tuple(images)
            return
        first = remaining[0]
        tail = remaining[1:]
        images[first] = first
        yield from visit(tail)
        for offset, second in enumerate(tail):
            images[first] = second
            images[second] = first
            yield from visit(tail[:offset] + tail[offset + 1 :])
        images[first] = -1

    yield from visit(tuple(range(size)))


def _physical_fold(
    orbit_images: Sequence[int],
    flips: Sequence[int],
    fixed_images: Sequence[int],
) -> tuple[int, ...]:
    num_orbits = len(orbit_images)
    output = [0] * (2 * num_orbits + len(fixed_images))
    for orbit, target_orbit in enumerate(orbit_images):
        for endpoint in (0, 1):
            output[2 * orbit + endpoint] = (
                2 * int(target_orbit) + (endpoint ^ int(flips[orbit]))
            )
    fixed_offset = 2 * num_orbits
    for index, target in enumerate(fixed_images):
        output[fixed_offset + index] = fixed_offset + int(target)
    return tuple(output)


def pairing_matrix(
    logical_z: Sequence[int], permutation: Sequence[int]
) -> np.ndarray:
    logical_x = tuple(permute_mask(mask, permutation) for mask in logical_z)
    return np.asarray(
        [
            [(left & right).bit_count() & 1 for right in logical_x]
            for left in logical_z
        ],
        dtype=np.uint8,
    )


def _fold_signature(
    geometry: LogicalGeometry,
    orbit_images: Sequence[int],
    flips: Sequence[int],
    fixed_images: Sequence[int],
) -> tuple[int, ...]:
    """Complete conjugacy signature under relabelings of the logical geometry."""
    counts: Counter[str] = Counter()
    seen: set[int] = set()
    for orbit, target in enumerate(orbit_images):
        if orbit in seen:
            continue
        seen.add(orbit)
        active = orbit < geometry.logical_weight
        target_active = target < geometry.logical_weight
        if target == orbit:
            prefix = "active-fixed" if active else "slack-fixed"
            counts[f"{prefix}-flip-{flips[orbit]}"] += 1
            continue
        seen.add(int(target))
        if active and target_active:
            counts[f"active-pair-flip-{flips[orbit]}"] += 1
        elif not active and not target_active:
            # Independent endpoint relabelings on slack orbits remove the sign.
            counts["slack-pair"] += 1
        else:
            counts["active-slack-pair"] += 1
    fixed_transpositions = sum(
        1 for index, target in enumerate(fixed_images) if index < target
    )
    keys = (
        "active-fixed-flip-0",
        "active-fixed-flip-1",
        "slack-fixed-flip-0",
        "slack-fixed-flip-1",
        "active-pair-flip-0",
        "active-pair-flip-1",
        "slack-pair",
        "active-slack-pair",
    )
    return tuple(counts[key] for key in keys) + (fixed_transpositions,)


@dataclass(frozen=True)
class FoldGeometry:
    geometry: LogicalGeometry
    index: int
    permutation: tuple[int, ...]
    raw_multiplicity: int

    @property
    def name(self) -> str:
        return f"{self.geometry.name}-fold-{self.index:04d}"

    def analyze(self) -> dict[str, Any]:
        permutation = self.permutation
        pairing = pairing_matrix(self.geometry.logical_z, permutation)
        return {
            "name": self.name,
            "geometry": self.geometry.analyze(),
            "permutation": list(permutation),
            "raw_multiplicity": self.raw_multiplicity,
            "involution": compose_permutations(permutation, permutation)
            == tuple(range(self.geometry.n)),
            "commutes_with_translation": compose_permutations(
                permutation, self.geometry.translation
            )
            == compose_permutations(self.geometry.translation, permutation),
            "pairing": pairing.astype(int).tolist(),
            "pairing_rank": gf2_rank(pairing),
            "pairing_is_permutation": is_permutation_matrix(pairing),
            "logical_x": [
                mask_to_support(permute_mask(mask, permutation), self.geometry.n)
                for mask in self.geometry.logical_z
            ],
        }


def canonical_fold_catalog(
    geometry: LogicalGeometry,
) -> tuple[FoldGeometry, ...]:
    """Enumerate ZX folds up to relabelings preserving C2 logical geometry."""
    representatives: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    for orbit_images, flips in _signed_involutions(geometry.transposition_orbits):
        for fixed_images in _ordinary_involutions(geometry.slack_fixed_points):
            permutation = _physical_fold(orbit_images, flips, fixed_images)
            pairing = pairing_matrix(geometry.logical_z, permutation)
            if not is_permutation_matrix(pairing):
                continue
            signature = _fold_signature(
                geometry, orbit_images, flips, fixed_images
            )
            known = representatives.get(signature)
            if known is None:
                representatives[signature] = (permutation, 1)
            else:
                known_permutation, multiplicity = known
                representatives[signature] = (
                    min(known_permutation, permutation),
                    multiplicity + 1,
                )
    raw = list(representatives.values())
    raw.sort(key=lambda item: item[0])
    return tuple(
        FoldGeometry(geometry, index, permutation, multiplicity)
        for index, (permutation, multiplicity) in enumerate(raw)
    )


def module_types(rank: int) -> tuple[tuple[int, ...], ...]:
    """All C2 module decompositions into indecomposables of rank one or two."""
    return tuple(
        (2,) * free_modules + (1,) * (rank - 2 * free_modules)
        for free_modules in range(rank // 2, -1, -1)
    )


def masks_through_weight(n: int, maximum_weight: int) -> Iterator[int]:
    for weight in range(1, maximum_weight + 1):
        for support in itertools.combinations(range(n), weight):
            yield sum(1 << qubit for qubit in support)


def _z3_bit(vector: z3.BitVecRef, index: int) -> z3.BoolRef:
    return z3.Extract(index, index, vector) == z3.BitVecVal(1, 1)


def _z3_parity(terms: Iterable[z3.BoolRef]) -> z3.BoolRef:
    terms = tuple(terms)
    if not terms:
        return z3.BoolVal(False)
    if len(terms) == 1:
        return terms[0]
    return reduce(z3.Xor, terms)


def _z3_permute(
    vector: z3.BitVecRef, permutation: Sequence[int], n: int
) -> z3.BitVecRef:
    terms = [
        z3.If(
            _z3_bit(vector, source),
            z3.BitVecVal(1 << int(target), n),
            z3.BitVecVal(0, n),
        )
        for source, target in enumerate(permutation)
    ]
    return reduce(lambda left, right: left | right, terms)


def _z3_dot(left: z3.BitVecRef, right: z3.BitVecRef, n: int) -> z3.BoolRef:
    return _z3_parity(
        z3.And(_z3_bit(left, index), _z3_bit(right, index))
        for index in range(n)
    )


def _z3_dot_mask(vector: z3.BitVecRef, mask: int, n: int) -> z3.BoolRef:
    return _z3_parity(
        _z3_bit(vector, index) for index in range(n) if mask >> index & 1
    )


def _z3_weight(vector: z3.BitVecRef, n: int) -> z3.ArithRef:
    return z3.Sum([z3.If(_z3_bit(vector, index), 1, 0) for index in range(n)])


def _xor_bitvectors(vectors: Sequence[z3.BitVecRef], n: int) -> z3.BitVecRef:
    if not vectors:
        return z3.BitVecVal(0, n)
    return reduce(xor, vectors)


class FoldedCSSSolver:
    """CEGIS solver for one C2 geometry, fold, and module type."""

    def __init__(
        self,
        *,
        fold: FoldGeometry,
        module_type: Sequence[int],
        maximum_check_weight: int = 8,
        timeout_ms: int = 10_000,
    ) -> None:
        self.fold = fold
        self.geometry = fold.geometry
        self.n = self.geometry.n
        self.target_rank = (self.n - TARGET_K) // 2
        self.module_type = tuple(module_type)
        if sum(self.module_type) != self.target_rank:
            raise ValueError("module ranks do not sum to the target check rank")
        if any(rank not in (1, 2) for rank in self.module_type):
            raise ValueError("C2 indecomposable ranks must be one or two")
        analysis = fold.analyze()
        if not all(
            analysis[key]
            for key in (
                "involution",
                "commutes_with_translation",
                "pairing_is_permutation",
            )
        ):
            raise ValueError(f"invalid fold geometry: {analysis}")
        self.maximum_check_weight = maximum_check_weight
        self.solver = z3.SolverFor("QF_BV")
        self.solver.set(timeout=timeout_ms)
        self.seed_variables = tuple(
            z3.BitVec(f"seed_{index}", self.n)
            for index in range(len(self.module_type))
        )
        self.basis_rows: tuple[z3.BitVecRef, ...] = ()
        self.distance_cuts = 0
        self.eager_distance_constraints = 0
        self._build_structural_constraints()

    def _build_structural_constraints(self) -> None:
        translation = self.geometry.translation
        basis_rows: list[z3.BitVecRef] = []
        for seed, rank in zip(self.seed_variables, self.module_type):
            translated = _z3_permute(seed, translation, self.n)
            if rank == 1:
                self.solver.add(translated == seed)
                basis_rows.append(seed)
            else:
                self.solver.add(translated != seed)
                self.solver.add(z3.ULT(seed, translated))
                basis_rows.extend((seed, translated))
            self.solver.add(_z3_weight(seed, self.n) >= 2)
            self.solver.add(_z3_weight(seed, self.n) <= self.maximum_check_weight)
        if len(basis_rows) != self.target_rank:
            raise AssertionError("wrong symbolic basis length")
        self.basis_rows = tuple(basis_rows)

        for combination in range(1, 1 << self.target_rank):
            selected = [
                row
                for index, row in enumerate(self.basis_rows)
                if combination >> index & 1
            ]
            self.solver.add(_xor_bitvectors(selected, self.n) != 0)

        for index in range(len(self.module_type) - 1):
            if self.module_type[index] == self.module_type[index + 1]:
                self.solver.add(
                    z3.ULT(self.seed_variables[index], self.seed_variables[index + 1])
                )

        for row in self.basis_rows:
            for logical in self.geometry.logical_z:
                self.solver.add(z3.Not(_z3_dot_mask(row, logical, self.n)))

        folded_rows = tuple(
            _z3_permute(row, self.fold.permutation, self.n)
            for row in self.basis_rows
        )
        for row_x in self.basis_rows:
            for row_z in folded_rows:
                self.solver.add(z3.Not(_z3_dot(row_x, row_z, self.n)))

        for qubit in range(self.n):
            self.solver.add(
                z3.Or(
                    *[_z3_bit(row, qubit) for row in self.basis_rows],
                    *[_z3_bit(row, qubit) for row in folded_rows],
                )
            )

    def check(self) -> z3.CheckSatResult:
        return self.solver.check()

    def set_timeout(self, timeout_ms: int) -> None:
        self.solver.set(timeout=timeout_ms)

    def concrete_basis(self) -> tuple[int, ...]:
        model = self.solver.model()
        return tuple(
            int(model.eval(row, model_completion=True).as_long())
            for row in self.basis_rows
        )

    def add_distance_cut(self, sector: str, operator: int) -> None:
        if sector == "z":
            rows = self.basis_rows
        elif sector == "x":
            rows = tuple(
                _z3_permute(row, self.fold.permutation, self.n)
                for row in self.basis_rows
            )
        else:
            raise ValueError("sector must be x or z")
        self.solver.add(
            z3.Or(*[_z3_dot_mask(row, operator, self.n) for row in rows])
        )
        self.distance_cuts += 1

    def add_full_distance_constraints(
        self, maximum_weight: int = TARGET_DISTANCE - 1
    ) -> int:
        """Exclude every low-weight X/Z logical in one exact eager pass."""
        logical_z = self.geometry.logical_z
        logical_x = tuple(
            permute_mask(mask, self.fold.permutation) for mask in logical_z
        )
        folded_rows = tuple(
            _z3_permute(row, self.fold.permutation, self.n)
            for row in self.basis_rows
        )
        added = 0
        for operator in masks_through_weight(self.n, maximum_weight):
            # A Z operator with a nonzero pairing against the complete logical-X
            # basis cannot be a stabilizer. It must therefore be detected by H_X.
            if any((operator & logical).bit_count() & 1 for logical in logical_x):
                self.solver.add(
                    z3.Or(
                        *[
                            _z3_dot_mask(row, operator, self.n)
                            for row in self.basis_rows
                        ]
                    )
                )
                added += 1
            # The dual statement excludes low-weight X logicals using H_Z.
            if any((operator & logical).bit_count() & 1 for logical in logical_z):
                self.solver.add(
                    z3.Or(
                        *[
                            _z3_dot_mask(row, operator, self.n)
                            for row in folded_rows
                        ]
                    )
                )
                added += 1
        self.eager_distance_constraints += added
        return added

    def block_current_seeds(self) -> None:
        model = self.solver.model()
        self.solver.add(
            z3.Or(
                *[
                    seed != model.eval(seed, model_completion=True)
                    for seed in self.seed_variables
                ]
            )
        )


def all_sums(basis: Sequence[int]) -> tuple[int, ...]:
    output = [0]
    for row in basis:
        output.extend(value ^ row for value in tuple(output))
    return tuple(output)


def minimum_logical_coset(
    logicals: Sequence[int], stabilizers: Sequence[int], n: int
) -> dict[str, Any]:
    sums = all_sums(stabilizers)
    best_weight = n + 1
    best_operator = 0
    best_label = 0
    counts: Counter[int] = Counter()
    low_operators: set[int] = set()
    for label in range(1, 1 << TARGET_K):
        logical = 0
        for index, mask in enumerate(logicals):
            if label >> index & 1:
                logical ^= mask
        for stabilizer in sums:
            operator = logical ^ stabilizer
            weight = operator.bit_count()
            if weight <= 7:
                counts[weight] += 1
            if weight < TARGET_DISTANCE:
                low_operators.add(operator)
            if weight < best_weight:
                best_weight = weight
                best_operator = operator
                best_label = label
    return {
        "distance": best_weight,
        "operator": best_operator,
        "operator_support": mask_to_support(best_operator, n),
        "logical_label": best_label,
        "low_logical_operators_total": len(low_operators),
        "low_logical_operators": sorted(
            low_operators, key=lambda value: (value.bit_count(), value)
        ),
        "weight_counts_through_seven": {
            str(weight): counts[weight] for weight in range(1, 8)
        },
    }


def tanner_components(
    basis_x: Sequence[int], basis_z: Sequence[int], n: int
) -> list[list[int]]:
    checks = tuple(basis_x) + tuple(basis_z)
    adjacency = [set() for _ in range(len(checks) + n)]
    for check_index, row in enumerate(checks):
        for qubit in mask_to_support(row, n):
            qnode = len(checks) + qubit
            adjacency[check_index].add(qnode)
            adjacency[qnode].add(check_index)
    components = []
    unseen = set(range(len(adjacency)))
    while unseen:
        start = next(iter(unseen))
        reached = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                queue.append(neighbor)
        unseen -= reached
        components.append(sorted(reached))
    return components


def _rank_masks(rows: Sequence[int], n: int) -> int:
    return gf2_rank(
        np.asarray([[row >> bit & 1 for bit in range(n)] for row in rows], dtype=np.uint8)
    )


def _column_degrees(rows: Sequence[int], n: int) -> list[int]:
    return [sum(row >> qubit & 1 for row in rows) for qubit in range(n)]


def analyze_basis(basis_x: Sequence[int], fold: FoldGeometry) -> dict[str, Any]:
    geometry = fold.geometry
    n = geometry.n
    basis_x = tuple(basis_x)
    basis_z = tuple(permute_mask(row, fold.permutation) for row in basis_x)
    logical_z = geometry.logical_z
    logical_x = tuple(permute_mask(mask, fold.permutation) for mask in logical_z)
    z_distance = minimum_logical_coset(logical_z, basis_z, n)
    x_distance = minimum_logical_coset(logical_x, basis_x, n)
    components = tanner_components(basis_x, basis_z, n)
    pairing = pairing_matrix(logical_z, fold.permutation)
    rank_x = _rank_masks(basis_x, n)
    rank_z = _rank_masks(basis_z, n)
    css = all(
        (left & right).bit_count() % 2 == 0
        for left in basis_x
        for right in basis_z
    )
    return {
        "n": n,
        "k": n - rank_x - rank_z,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "css_orthogonal": css,
        "tanner_connected": len(components) == 1,
        "tanner_component_sizes": [len(component) for component in components],
        "distance": min(z_distance["distance"], x_distance["distance"]),
        "z_distance": z_distance,
        "x_distance": x_distance,
        "basis_x_masks": list(basis_x),
        "basis_z_masks": list(basis_z),
        "basis_x": [mask_to_support(row, n) for row in basis_x],
        "basis_z": [mask_to_support(row, n) for row in basis_z],
        "row_weights_x": [row.bit_count() for row in basis_x],
        "row_weights_z": [row.bit_count() for row in basis_z],
        "column_degrees_x": _column_degrees(basis_x, n),
        "column_degrees_z": _column_degrees(basis_z, n),
        "logical_z": [mask_to_support(row, n) for row in logical_z],
        "logical_x": [mask_to_support(row, n) for row in logical_x],
        "logical_z_disjoint": not bool(logical_z[0] & logical_z[1]),
        "logical_x_disjoint": not bool(logical_x[0] & logical_x[1]),
        "logical_pairing": pairing.astype(int).tolist(),
        "logical_pairing_is_permutation": is_permutation_matrix(pairing),
        "translation": list(geometry.translation),
        "translation_swaps_logical_z": tuple(
            permute_mask(mask, geometry.translation) for mask in logical_z
        )
        == logical_z[::-1],
        "translation_swaps_logical_x": tuple(
            permute_mask(mask, geometry.translation) for mask in logical_x
        )
        == logical_x[::-1],
        "fold": fold.analyze(),
    }
