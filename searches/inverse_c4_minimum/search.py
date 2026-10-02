"""Direct folded-CSS synthesis for a single four-spin row at n=28."""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from functools import reduce
from operator import xor
from typing import Any, Iterable, Sequence

import numpy as np
import z3


LOGICAL_ORDER = 4
NUM_FIBRES = 7
NUM_QUBITS = LOGICAL_ORDER * NUM_FIBRES
TARGET_K = 4
TARGET_RANK = (NUM_QUBITS - TARGET_K) // 2
MAXIMUM_CHECK_WEIGHT = 12


def coordinate(fibre: int, position: int) -> int:
    return (fibre % NUM_FIBRES) * LOGICAL_ORDER + position % LOGICAL_ORDER


def translation_permutation(shift: int = 1) -> tuple[int, ...]:
    return tuple(
        coordinate(fibre, position + shift)
        for fibre in range(NUM_FIBRES)
        for position in range(LOGICAL_ORDER)
    )


TRANSLATION = translation_permutation()


def fibre_fold(
    fibre_images: Sequence[int],
    *,
    epsilon: int = 1,
    shifts: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return ``(f,p) -> (sigma(f), epsilon*p+delta_f)``."""
    if sorted(fibre_images) != list(range(NUM_FIBRES)):
        raise ValueError("fibre_images must be a permutation")
    if epsilon not in (1, -1):
        raise ValueError("epsilon must be +1 or -1")
    shifts = tuple(shifts or (0,) * NUM_FIBRES)
    if len(shifts) != NUM_FIBRES:
        raise ValueError("one shift is required for every fibre")
    return tuple(
        coordinate(fibre_images[fibre], epsilon * position + shifts[fibre])
        for fibre in range(NUM_FIBRES)
        for position in range(LOGICAL_ORDER)
    )


def compose_permutations(
    left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    """Compose source-to-target permutations as ``left after right``."""
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def invert_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = source
    return tuple(inverse)


def permute_mask(mask: int, permutation: Sequence[int]) -> int:
    output = 0
    for source, target in enumerate(permutation):
        if mask >> source & 1:
            output |= 1 << int(target)
    return output


def mask_to_support(mask: int) -> list[int]:
    return [index for index in range(NUM_QUBITS) if mask >> index & 1]


def logical_masks(weight: int = 6) -> tuple[int, ...]:
    if not 1 <= weight <= NUM_FIBRES:
        raise ValueError("logical weight must lie between one and seven")
    seed = sum(1 << coordinate(fibre, 0) for fibre in range(weight))
    return tuple(
        permute_mask(seed, translation_permutation(shift))
        for shift in range(LOGICAL_ORDER)
    )


def pairing_matrix(
    z_masks: Sequence[int], fold: Sequence[int]
) -> np.ndarray:
    x_masks = tuple(permute_mask(mask, fold) for mask in z_masks)
    return np.asarray(
        [[(left & right).bit_count() & 1 for right in x_masks] for left in z_masks],
        dtype=np.uint8,
    )


def gf2_rank(matrix: np.ndarray) -> int:
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    rank = 0
    for column in range(reduced.shape[1]):
        choices = np.flatnonzero(reduced[rank:, column])
        if not len(choices):
            continue
        pivot = rank + int(choices[0])
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in np.flatnonzero(reduced[:, column]):
            if row != rank:
                reduced[row] ^= reduced[rank]
        rank += 1
        if rank == reduced.shape[0]:
            break
    return rank


def is_permutation_matrix(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (LOGICAL_ORDER, LOGICAL_ORDER)
        and np.all(np.count_nonzero(matrix, axis=0) == 1)
        and np.all(np.count_nonzero(matrix, axis=1) == 1)
    )


@dataclass(frozen=True)
class FoldGeometry:
    name: str
    fibre_images: tuple[int, ...]
    epsilon: int = 1
    shifts: tuple[int, ...] = (0,) * NUM_FIBRES
    logical_weight: int = 6

    @property
    def permutation(self) -> tuple[int, ...]:
        return fibre_fold(
            self.fibre_images, epsilon=self.epsilon, shifts=self.shifts
        )

    def analyze(self) -> dict[str, Any]:
        permutation = self.permutation
        z_masks = logical_masks(self.logical_weight)
        pairing = pairing_matrix(z_masks, permutation)
        inverse = invert_permutation(permutation)
        conjugated_translation = compose_permutations(
            compose_permutations(permutation, TRANSLATION), inverse
        )
        return {
            "name": self.name,
            "fibre_images": list(self.fibre_images),
            "epsilon": self.epsilon,
            "shifts": list(self.shifts),
            "logical_weight": self.logical_weight,
            "involution": compose_permutations(permutation, permutation)
            == tuple(range(NUM_QUBITS)),
            "normalizes_translation": conjugated_translation
            in {TRANSLATION, translation_permutation(-1)},
            "pairing": pairing.astype(int).tolist(),
            "pairing_rank": gf2_rank(pairing),
            "pairing_is_permutation": is_permutation_matrix(pairing),
        }


def canonical_fold_catalog() -> tuple[FoldGeometry, ...]:
    """Simple inequivalent zero-shift involutions moving the slack fibre."""
    folds = []
    for internal_transpositions in range(3):
        images = list(range(NUM_FIBRES))
        images[5], images[6] = 6, 5
        for offset in range(internal_transpositions):
            left = 2 * offset
            right = left + 1
            images[left], images[right] = right, left
        folds.append(
            FoldGeometry(
                name=f"swap-slack-internal-{internal_transpositions}",
                fibre_images=tuple(images),
            )
        )
        folds.append(
            FoldGeometry(
                name=f"reflect-swap-slack-internal-{internal_transpositions}",
                fibre_images=tuple(images),
                epsilon=-1,
            )
        )
    return tuple(folds)


def module_partitions(
    total: int = TARGET_RANK, maximum_part: int = LOGICAL_ORDER
) -> tuple[tuple[int, ...], ...]:
    output: list[tuple[int, ...]] = []

    def visit(remaining: int, largest: int, prefix: tuple[int, ...]) -> None:
        if remaining == 0:
            output.append(prefix)
            return
        for part in range(min(remaining, largest, maximum_part), 0, -1):
            visit(remaining - part, part, prefix + (part,))

    visit(total, maximum_part, ())
    return tuple(output)


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
    vector: z3.BitVecRef, permutation: Sequence[int]
) -> z3.BitVecRef:
    terms = [
        z3.If(
            _z3_bit(vector, source),
            z3.BitVecVal(1 << int(target), NUM_QUBITS),
            z3.BitVecVal(0, NUM_QUBITS),
        )
        for source, target in enumerate(permutation)
    ]
    return reduce(lambda left, right: left | right, terms)


def _z3_dot(left: z3.BitVecRef, right: z3.BitVecRef) -> z3.BoolRef:
    return _z3_parity(
        z3.And(_z3_bit(left, index), _z3_bit(right, index))
        for index in range(NUM_QUBITS)
    )


def _z3_dot_mask(vector: z3.BitVecRef, mask: int) -> z3.BoolRef:
    return _z3_parity(
        _z3_bit(vector, index)
        for index in range(NUM_QUBITS)
        if mask >> index & 1
    )


def _z3_weight(vector: z3.BitVecRef) -> z3.ArithRef:
    return z3.Sum(
        [z3.If(_z3_bit(vector, index), 1, 0) for index in range(NUM_QUBITS)]
    )


def _nilpotent_power(vector: z3.BitVecRef, power: int) -> z3.BitVecRef:
    output = vector
    for _ in range(power):
        output = _z3_permute(output, TRANSLATION) ^ output
    return output


def _xor_bitvectors(vectors: Sequence[z3.BitVecRef]) -> z3.BitVecRef:
    if not vectors:
        return z3.BitVecVal(0, NUM_QUBITS)
    return reduce(xor, vectors)


class FoldedCSSSolver:
    """CEGIS solver for one fold and one cyclic C4 module type."""

    def __init__(
        self,
        *,
        geometry: FoldGeometry,
        module_type: Sequence[int],
        maximum_check_weight: int = MAXIMUM_CHECK_WEIGHT,
        timeout_ms: int = 10_000,
    ) -> None:
        self.geometry = geometry
        self.module_type = tuple(module_type)
        if sum(self.module_type) != TARGET_RANK:
            raise ValueError("module ranks must sum to twelve")
        analysis = geometry.analyze()
        if not all(
            analysis[key]
            for key in (
                "involution",
                "normalizes_translation",
                "pairing_is_permutation",
            )
        ):
            raise ValueError(f"invalid fold geometry: {analysis}")
        self.maximum_check_weight = maximum_check_weight
        self.timeout_ms = timeout_ms
        self.solver = z3.SolverFor("QF_BV")
        self.solver.set(timeout=timeout_ms)
        self.seed_variables = tuple(
            z3.BitVec(f"seed_{index}", NUM_QUBITS)
            for index in range(len(self.module_type))
        )
        self.z_masks = logical_masks(geometry.logical_weight)
        self.x_masks = tuple(
            permute_mask(mask, geometry.permutation) for mask in self.z_masks
        )
        self.basis_rows: tuple[z3.BitVecRef, ...] = ()
        self.distance_cuts = 0
        self._build_structural_constraints()

    def _build_structural_constraints(self) -> None:
        basis_rows: list[z3.BitVecRef] = []
        for seed, rank in zip(self.seed_variables, self.module_type):
            self.solver.add(_nilpotent_power(seed, rank) == 0)
            self.solver.add(_nilpotent_power(seed, rank - 1) != 0)
            weighted_bits = [
                (_z3_bit(seed, index), 1) for index in range(NUM_QUBITS)
            ]
            self.solver.add(z3.PbGe(weighted_bits, 2))
            self.solver.add(
                z3.PbLe(weighted_bits, self.maximum_check_weight)
            )
            basis_rows.extend(
                _z3_permute(seed, translation_permutation(shift))
                for shift in range(rank)
            )
        if len(basis_rows) != TARGET_RANK:
            raise AssertionError("wrong symbolic basis length")
        self.basis_rows = tuple(basis_rows)

        # Exact rank 12.  With only 4095 nonzero coefficient vectors this is
        # substantially smaller than a symbolic Gaussian elimination circuit.
        for combination in range(1, 1 << TARGET_RANK):
            selected = [
                row
                for index, row in enumerate(self.basis_rows)
                if combination >> index & 1
            ]
            self.solver.add(_xor_bitvectors(selected) != 0)

        # Equal-rank module seeds are interchangeable.
        for index in range(len(self.module_type) - 1):
            if self.module_type[index] == self.module_type[index + 1]:
                self.solver.add(
                    z3.ULT(self.seed_variables[index], self.seed_variables[index + 1])
                )

        # The logical Z grid lies in ker(H_X).
        for row in self.basis_rows:
            for logical in self.z_masks:
                self.solver.add(z3.Not(_z3_dot_mask(row, logical)))

        # Define H_Z as the folded image of H_X and impose CSS orthogonality.
        folded_rows = tuple(
            _z3_permute(row, self.geometry.permutation) for row in self.basis_rows
        )
        for row_x in self.basis_rows:
            for row_z in folded_rows:
                self.solver.add(z3.Not(_z3_dot(row_x, row_z)))

        # Avoid isolated physical qubits.  Full Tanner connectedness is checked
        # concretely and counterexamples are blocked after a model is returned.
        for qubit in range(NUM_QUBITS):
            self.solver.add(
                z3.Or(
                    *[_z3_bit(row, qubit) for row in self.basis_rows],
                    *[_z3_bit(row, qubit) for row in folded_rows],
                )
            )

    def check(self) -> z3.CheckSatResult:
        return self.solver.check()

    def set_timeout(self, timeout_ms: int) -> None:
        self.timeout_ms = timeout_ms
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
                _z3_permute(row, self.geometry.permutation)
                for row in self.basis_rows
            )
        else:
            raise ValueError("sector must be x or z")
        self.solver.add(
            z3.Or(*[_z3_dot_mask(row, operator) for row in rows])
        )
        self.distance_cuts += 1

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
    logicals: Sequence[int], stabilizers: Sequence[int]
) -> dict[str, Any]:
    sums = all_sums(stabilizers)
    best_weight = NUM_QUBITS + 1
    best_operator = 0
    best_label = 0
    counts: Counter[int] = Counter()
    low_operators: set[int] = set()
    for label in range(1, 1 << LOGICAL_ORDER):
        logical = 0
        for index, mask in enumerate(logicals):
            if label >> index & 1:
                logical ^= mask
        for stabilizer in sums:
            operator = logical ^ stabilizer
            weight = operator.bit_count()
            if weight <= 7:
                counts[weight] += 1
            if weight <= 5:
                low_operators.add(operator)
            if weight < best_weight:
                best_weight = weight
                best_operator = operator
                best_label = label
    return {
        "distance": best_weight,
        "operator": best_operator,
        "operator_support": mask_to_support(best_operator),
        "logical_label": best_label,
        "low_logical_operators_total": len(low_operators),
        "low_logical_operators": sorted(
            low_operators, key=lambda value: (value.bit_count(), value)
        ),
        "weight_counts_through_seven": {
            str(weight): counts[weight] for weight in range(1, 8)
        },
    }


def tanner_components(hx: Sequence[int], hz: Sequence[int]) -> list[list[int]]:
    checks = tuple(hx) + tuple(hz)
    adjacency = [set() for _ in range(len(checks) + NUM_QUBITS)]
    for check_index, row in enumerate(checks):
        for qubit in mask_to_support(row):
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


def analyze_basis(
    basis_x: Sequence[int], geometry: FoldGeometry
) -> dict[str, Any]:
    basis_x = tuple(basis_x)
    basis_z = tuple(permute_mask(row, geometry.permutation) for row in basis_x)
    z_distance = minimum_logical_coset(logical_masks(geometry.logical_weight), basis_z)
    x_masks = tuple(
        permute_mask(mask, geometry.permutation)
        for mask in logical_masks(geometry.logical_weight)
    )
    x_distance = minimum_logical_coset(x_masks, basis_x)
    components = tanner_components(basis_x, basis_z)
    css = all(
        (left & right).bit_count() % 2 == 0
        for left in basis_x
        for right in basis_z
    )
    return {
        "n": NUM_QUBITS,
        "k": TARGET_K,
        "rank_x": gf2_rank(
            np.asarray(
                [[row >> bit & 1 for bit in range(NUM_QUBITS)] for row in basis_x],
                dtype=np.uint8,
            )
        ),
        "rank_z": gf2_rank(
            np.asarray(
                [[row >> bit & 1 for bit in range(NUM_QUBITS)] for row in basis_z],
                dtype=np.uint8,
            )
        ),
        "css_orthogonal": css,
        "tanner_connected": len(components) == 1,
        "tanner_component_sizes": [len(component) for component in components],
        "distance": min(z_distance["distance"], x_distance["distance"]),
        "z_distance": z_distance,
        "x_distance": x_distance,
        "basis_x": [mask_to_support(row) for row in basis_x],
        "basis_z": [mask_to_support(row) for row in basis_z],
        "row_weights_x": [row.bit_count() for row in basis_x],
        "row_weights_z": [row.bit_count() for row in basis_z],
        "logical_z": [mask_to_support(row) for row in logical_masks(geometry.logical_weight)],
        "logical_x": [
            mask_to_support(permute_mask(row, geometry.permutation))
            for row in logical_masks(geometry.logical_weight)
        ],
        "fold": geometry.analyze(),
    }


def n24_no_go_certificate() -> dict[str, Any]:
    """Machine-readable statement of the exact pairing-augmentation proof."""
    return {
        "n": 24,
        "k": 4,
        "distance_lower_bound": 6,
        "disjoint_support_bound": "four supports of weight >=6 fill all 24 qubits",
        "pairing_equation": "M_ij = z_i dot P(z_j)",
        "row_sum_equation": "sum_j M_ij = z_i dot P(1) = wt(z_i) = 0 mod 2",
        "kernel_vector": [1, 1, 1, 1],
        "conclusion": "every physical permutation P gives singular logical ZX pairing",
        "scope": "four disjoint logicals exchanged with X logicals by transversal H plus a physical permutation",
    }


def supports_through_weight(maximum_weight: int) -> int:
    return sum(math.comb(NUM_QUBITS, weight) for weight in range(1, maximum_weight + 1))
