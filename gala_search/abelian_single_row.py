"""Paper-style ZX-dual abelian GALA search for ``[[32,4,6]]`` codes.

The construction uses ``L=4``, ``J=2`` and an abelian lift group of degree
eight.  The two supported groups are ``C8`` and ``C2 x C4``.  Polynomial
generators ``F0,F1`` determine ``G0,G1`` through the transpose-and-translate
condition from Proposition 6 of the GALA paper.
"""

from __future__ import annotations

import functools
import itertools
import math
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qldpc import codes

from gala_search.single_row import (
    _pack,
    _packed_in_span,
    _packed_row_basis,
    _tanner_connected,
    gf2_rank,
)

SCHEMA_VERSION = 1
L = 4
J = 2
GROUP_ORDER = 8
NUM_QUBITS = L * GROUP_ORDER
NUM_CHECKS = J * GROUP_ORDER
TARGET_DIMENSION = 4
TARGET_CHECK_RANK = (NUM_QUBITS - TARGET_DIMENSION) // 2
C4_ORDER = 4
LOGICAL_WEIGHT = 7


@dataclass(frozen=True)
class AbelianGroupSpec:
    """A degree-eight abelian lift group with a distinguished ``C4`` action."""

    name: str
    moduli: tuple[int, ...]
    logical_translation: tuple[int, ...]

    @property
    def elements(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product(*(range(modulus) for modulus in self.moduli)))

    def normalize(self, element: Sequence[int]) -> tuple[int, ...]:
        return tuple(value % modulus for value, modulus in zip(element, self.moduli))

    def add(
        self, left: Sequence[int], right: Sequence[int]
    ) -> tuple[int, ...]:
        return self.normalize(tuple(a + b for a, b in zip(left, right)))

    def inverse(self, element: Sequence[int]) -> tuple[int, ...]:
        return self.normalize(tuple(-value for value in element))

    def index(self, element: Sequence[int]) -> int:
        return self.elements.index(self.normalize(element))

    @property
    def identity(self) -> tuple[int, ...]:
        return (0,) * len(self.moduli)

    @property
    def involutions(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            element
            for element in self.elements
            if self.add(element, element) == self.identity
        )


GROUP_SPECS = {
    "c8": AbelianGroupSpec("C8", (8,), (2,)),
    "c2xc4": AbelianGroupSpec("C2 x C4", (2, 4), (0, 1)),
}


@dataclass(frozen=True)
class DualitySector:
    """One of the four Proposition-6 sector labels specialized to ``L=4``."""

    name: str
    relation: str
    outer_shift: int


DUALITY_SECTORS = (
    DualitySector("r0", "identity", 1),
    DualitySector("r1", "swap", 0),
    DualitySector("r2", "identity", 0),
    DualitySector("r3", "swap", 1),
)


def group_permutation(
    spec: AbelianGroupSpec, element: Sequence[int]
) -> np.ndarray:
    """Return the source-to-target permutation for translation by ``element``."""
    return np.asarray(
        [spec.index(spec.add(source, element)) for source in spec.elements],
        dtype=int,
    )


def permutation_matrix(
    spec: AbelianGroupSpec, element: Sequence[int]
) -> np.ndarray:
    permutation = group_permutation(spec, element)
    matrix = np.zeros((GROUP_ORDER, GROUP_ORDER), dtype=np.uint8)
    matrix[permutation, np.arange(GROUP_ORDER)] = 1
    return matrix


@functools.lru_cache(maxsize=None)
def basis_lifts(group_key: str) -> tuple[np.ndarray, ...]:
    spec = GROUP_SPECS[group_key]
    return tuple(permutation_matrix(spec, element) for element in spec.elements)


def polynomial_matrix(group_key: str, support: Sequence[int]) -> np.ndarray:
    if not support:
        return np.zeros((GROUP_ORDER, GROUP_ORDER), dtype=np.uint8)
    return np.bitwise_xor.reduce(
        np.asarray([basis_lifts(group_key)[index] for index in support]), axis=0
    )


def translated_dagger_support(
    spec: AbelianGroupSpec, support: Sequence[int], translation: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        sorted(
            spec.index(spec.add(translation, spec.inverse(spec.elements[index])))
            for index in support
        )
    )


def derive_g_supports(
    spec: AbelianGroupSpec,
    f_supports: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    relation: str,
    translation: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Enforce ``G_r(i) = t F_i^dagger`` for an identity or swap relation."""
    if relation == "identity":
        sources = (0, 1)
    elif relation == "swap":
        sources = (1, 0)
    else:
        raise ValueError(f"unsupported sector relation: {relation}")
    return tuple(
        translated_dagger_support(spec, f_supports[source], translation)
        for source in sources
    )  # type: ignore[return-value]


def build_checks(
    group_key: str,
    f_supports: tuple[tuple[int, ...], tuple[int, ...]],
    g_supports: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the first two rows of ``[F|G]`` and ``[G^T|F^T]``."""
    lifts = tuple(
        polynomial_matrix(group_key, support)
        for support in (*f_supports, *g_supports)
    )
    matrix_x = np.zeros((NUM_CHECKS, NUM_QUBITS), dtype=np.uint8)
    matrix_z = np.zeros_like(matrix_x)
    for entry, lift in enumerate(lifts):
        side, offset = divmod(entry, 2)
        for row in range(J):
            column = side * 2 + (offset + row) % 2
            matrix_x[
                row * GROUP_ORDER : (row + 1) * GROUP_ORDER,
                column * GROUP_ORDER : (column + 1) * GROUP_ORDER,
            ] = lift
            dual_column = (1 - side) * 2 + (offset + row) % 2
            matrix_z[
                row * GROUP_ORDER : (row + 1) * GROUP_ORDER,
                dual_column * GROUP_ORDER : (dual_column + 1) * GROUP_ORDER,
            ] = lift.T
    return matrix_x, matrix_z


def physical_fold(
    spec: AbelianGroupSpec,
    *,
    outer_shift: int,
    translation: Sequence[int],
) -> np.ndarray:
    """Construct ``tau = iota_r tensor t`` as a physical permutation."""
    internal = group_permutation(spec, translation)
    output = np.empty(NUM_QUBITS, dtype=int)
    for block in range(L):
        side, sector = divmod(block, 2)
        target_block = side * 2 + (sector + outer_shift) % 2
        for source_internal, target_internal in enumerate(internal):
            output[block * GROUP_ORDER + source_internal] = (
                target_block * GROUP_ORDER + int(target_internal)
            )
    return output


def permute_columns(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, np.asarray(permutation, dtype=int)] = matrix
    return output


def permute_check_blocks(matrix: np.ndarray, shift: int) -> np.ndarray:
    output = np.zeros_like(matrix)
    for source in range(J):
        target = (source + shift) % J
        output[
            target * GROUP_ORDER : (target + 1) * GROUP_ORDER
        ] = matrix[source * GROUP_ORDER : (source + 1) * GROUP_ORDER]
    return output


def verify_paper_fold(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, Any]:
    """Verify involution and exact check exchange, allowing paper row reindexing."""
    folded_x = permute_columns(matrix_x, permutation)
    folded_z = permute_columns(matrix_z, permutation)
    matching_row_shifts = [
        shift
        for shift in range(J)
        if np.array_equal(permute_check_blocks(folded_x, shift), matrix_z)
        and np.array_equal(permute_check_blocks(folded_z, shift), matrix_x)
    ]
    return {
        "involution": bool(
            np.array_equal(permutation[permutation], np.arange(NUM_QUBITS))
        ),
        "exact_without_row_reindexing": 0 in matching_row_shifts,
        "exact_with_paper_row_reindexing": bool(matching_row_shifts),
        "matching_row_shifts": matching_row_shifts,
    }


def translation_permutation(spec: AbelianGroupSpec) -> np.ndarray:
    """Return the diagonal physical ``C4`` translation on all four blocks."""
    internal = group_permutation(spec, spec.logical_translation)
    return np.concatenate(
        [block * GROUP_ORDER + internal for block in range(L)]
    ).astype(int)


def permutation_orbit(seed: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    rows = []
    current = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(C4_ORDER):
        rows.append(current)
        current = permute_columns(current[None, :], permutation)[0]
    return np.asarray(rows, dtype=np.uint8)


def c4_fibres(spec: AbelianGroupSpec) -> tuple[tuple[int, ...], ...]:
    """Partition the 32 qubits into the eight physical ``C4`` orbits."""
    permutation = translation_permutation(spec)
    seen: set[int] = set()
    fibres: list[tuple[int, ...]] = []
    for start in range(NUM_QUBITS):
        if start in seen:
            continue
        orbit = []
        current = start
        for _ in range(C4_ORDER):
            orbit.append(current)
            seen.add(current)
            current = int(permutation[current])
        fibres.append(tuple(orbit))
    return tuple(fibres)


def iter_graph_seeds(
    spec: AbelianGroupSpec, weight: int = LOGICAL_WEIGHT
) -> Iterable[tuple[int, ...]]:
    """Yield graph-supported seeds of fixed weight modulo ``C4`` shift."""
    fibres = c4_fibres(spec)
    for selected in itertools.combinations(range(len(fibres)), weight):
        first = selected[0]
        for phases_tail in itertools.product(range(C4_ORDER), repeat=weight - 1):
            phases = (0, *phases_tail)
            yield tuple(
                sorted(
                    fibres[fibre][phase]
                    for fibre, phase in zip((first, *selected[1:]), phases)
                )
            )


@functools.lru_cache(maxsize=None)
def paired_graph_seeds(
    group_key: str, translation_index: int, outer_shift: int
) -> tuple[tuple[int, ...], ...]:
    """Return graph seeds with full-rank ZX pairing for one formal fold."""
    spec = GROUP_SPECS[group_key]
    fold = physical_fold(
        spec,
        outer_shift=outer_shift,
        translation=spec.elements[translation_index],
    )
    logical_shift = translation_permutation(spec)
    permutation_pairing: list[tuple[int, ...]] = []
    general_pairing: list[tuple[int, ...]] = []
    for support in iter_graph_seeds(spec):
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbit = permutation_orbit(seed, logical_shift)
        x_orbit = permute_columns(z_orbit, fold)
        pairing = (z_orbit @ x_orbit.T) % 2
        if gf2_rank(pairing) != TARGET_DIMENSION:
            continue
        if np.all(np.count_nonzero(pairing, axis=0) == 1) and np.all(
            np.count_nonzero(pairing, axis=1) == 1
        ):
            permutation_pairing.append(support)
        else:
            general_pairing.append(support)
    return tuple((*permutation_pairing, *general_pairing))


def find_disjoint_logical_seed(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    *,
    group_key: str,
    translation_index: int,
    outer_shift: int,
) -> dict[str, Any] | None:
    """Find four disjoint odd-weight ``C4`` logicals paired by the fold."""
    spec = GROUP_SPECS[group_key]
    fold = physical_fold(
        spec,
        outer_shift=outer_shift,
        translation=spec.elements[translation_index],
    )
    logical_shift = translation_permutation(spec)
    column_syndromes = [_pack(matrix_x[:, qubit]) for qubit in range(NUM_QUBITS)]
    for support in paired_graph_seeds(group_key, translation_index, outer_shift):
        syndrome = 0
        for qubit in support:
            syndrome ^= column_syndromes[qubit]
        if syndrome:
            continue
        seed = np.zeros(NUM_QUBITS, dtype=np.uint8)
        seed[list(support)] = 1
        z_orbit = permutation_orbit(seed, logical_shift)
        x_orbit = permute_columns(z_orbit, fold)
        if np.any((matrix_x @ z_orbit.T) % 2) or np.any(
            (matrix_z @ x_orbit.T) % 2
        ):
            continue
        pairing = (z_orbit @ x_orbit.T) % 2
        rank_gain = gf2_rank(np.vstack([matrix_z, z_orbit])) - gf2_rank(matrix_z)
        if rank_gain != TARGET_DIMENSION:
            continue
        return {
            "seed_support": list(support),
            "logical_supports": [
                np.flatnonzero(row).astype(int).tolist() for row in z_orbit
            ],
            "logical_weights": np.count_nonzero(z_orbit, axis=1).astype(int).tolist(),
            "pairwise_disjoint": bool(np.all(np.sum(z_orbit, axis=0) <= 1)),
            "rank_mod_stabilizers": rank_gain,
            "zx_pairing": pairing.astype(int).tolist(),
            "zx_pairing_rank": gf2_rank(pairing),
            "zx_pairing_is_permutation": bool(
                np.all(np.count_nonzero(pairing, axis=0) == 1)
                and np.all(np.count_nonzero(pairing, axis=1) == 1)
            ),
        }
    return None


def find_css_logical_below_six(
    check: np.ndarray, stabilizer: np.ndarray
) -> dict[str, Any] | None:
    column_syndromes = [_pack(check[:, qubit]) for qubit in range(NUM_QUBITS)]
    stabilizer_basis = _packed_row_basis(stabilizer)
    for weight in range(1, 6):
        for support in itertools.combinations(range(NUM_QUBITS), weight):
            syndrome = 0
            vector = 0
            for qubit in support:
                syndrome ^= column_syndromes[qubit]
                vector |= 1 << qubit
            if syndrome == 0 and not _packed_in_span(vector, stabilizer_basis):
                return {"weight": weight, "support": list(support)}
    return None


def certify_distance_six(matrix_x: np.ndarray, matrix_z: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    z_logical = find_css_logical_below_six(matrix_x, matrix_z)
    x_logical = find_css_logical_below_six(matrix_z, matrix_x)
    weight_six_z = None
    weight_six_x = None
    if z_logical is None and x_logical is None:
        weight_six_z = find_css_logical_of_weight(matrix_x, matrix_z, 6)
        weight_six_x = find_css_logical_of_weight(matrix_z, matrix_x, 6)
    certified_distance = None
    if z_logical is None and x_logical is None:
        certified_distance = 6 if weight_six_z or weight_six_x else LOGICAL_WEIGHT
    return {
        "certified_distance": certified_distance,
        "certified_distance_at_least_six": z_logical is None and x_logical is None,
        "z_logical_below_six": z_logical,
        "x_logical_below_six": x_logical,
        "z_logical_at_six": weight_six_z,
        "x_logical_at_six": weight_six_x,
        "supports_checked_per_sector": sum(
            math.comb(NUM_QUBITS, weight) for weight in range(1, 6)
        ),
        "seconds": round(time.perf_counter() - started, 6),
    }


def find_css_logical_of_weight(
    check: np.ndarray, stabilizer: np.ndarray, weight: int
) -> dict[str, Any] | None:
    """Return a nontrivial CSS logical of exactly ``weight``, if one exists."""
    column_syndromes = [_pack(check[:, qubit]) for qubit in range(NUM_QUBITS)]
    stabilizer_basis = _packed_row_basis(stabilizer)
    for support in itertools.combinations(range(NUM_QUBITS), weight):
        syndrome = 0
        vector = 0
        for qubit in support:
            syndrome ^= column_syndromes[qubit]
            vector |= 1 << qubit
        if syndrome == 0 and not _packed_in_span(vector, stabilizer_basis):
            return {"weight": weight, "support": list(support)}
    return None


def iter_combined_f_supports(maximum_f_weight: int) -> Iterable[
    tuple[tuple[int, ...], tuple[int, ...]]
]:
    """Enumerate all two-polynomial supports of bounded combined weight."""
    for weight in range(1, maximum_f_weight + 1):
        for combined in itertools.combinations(range(2 * GROUP_ORDER), weight):
            yield (
                tuple(index for index in combined if index < GROUP_ORDER),
                tuple(index - GROUP_ORDER for index in combined if index >= GROUP_ORDER),
            )


def serialize_support(
    spec: AbelianGroupSpec, support: Sequence[int]
) -> list[list[int]]:
    return [list(spec.elements[index]) for index in support]


def _code_key(matrix_x: np.ndarray, matrix_z: np.ndarray) -> bytes:
    return np.packbits(np.concatenate([matrix_x.ravel(), matrix_z.ravel()])).tobytes()


def search_l4_group(
    group_key: str,
    *,
    maximum_check_weight: int = 12,
    certify_distance: bool = True,
    maximum_saved_near_misses: int = 20,
) -> dict[str, Any]:
    """Exhaust the paper-style ``L=4,J=2`` family for one abelian group."""
    spec = GROUP_SPECS[group_key]
    if maximum_check_weight % 2:
        raise ValueError("the check-weight ceiling must be even")
    maximum_f_weight = maximum_check_weight // 2
    counters: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    seen_code_folds: set[tuple[bytes, tuple[int, ...]]] = set()
    started = time.perf_counter()

    relation_sectors = {
        relation: tuple(sector for sector in DUALITY_SECTORS if sector.relation == relation)
        for relation in ("identity", "swap")
    }
    for relation, sectors in relation_sectors.items():
        for translation in spec.involutions:
            translation_index = spec.index(translation)
            for f_supports in iter_combined_f_supports(maximum_f_weight):
                counters["generator_supports"] += 1
                g_supports = derive_g_supports(
                    spec,
                    f_supports,
                    relation=relation,
                    translation=translation,
                )
                matrix_x, matrix_z = build_checks(group_key, f_supports, g_supports)
                if np.any((matrix_x @ matrix_z.T) % 2):
                    counters["non_css"] += 1
                    continue
                counters["css"] += 1
                rank_x = gf2_rank(matrix_x)
                rank_z = gf2_rank(matrix_z)
                if rank_x != TARGET_CHECK_RANK or rank_z != TARGET_CHECK_RANK:
                    counters["wrong_rank"] += 1
                    continue
                counters["rank_14_14"] += 1
                if not _tanner_connected(matrix_x, matrix_z):
                    counters["disconnected"] += 1
                    continue
                counters["connected"] += 1
                key = _code_key(matrix_x, matrix_z)

                for sector in sectors:
                    fold = physical_fold(
                        spec,
                        outer_shift=sector.outer_shift,
                        translation=translation,
                    )
                    fold_analysis = verify_paper_fold(matrix_x, matrix_z, fold)
                    if not (
                        fold_analysis["involution"]
                        and fold_analysis["exact_with_paper_row_reindexing"]
                    ):
                        counters["invalid_paper_fold"] += 1
                        continue
                    counters["paper_fold"] += 1
                    code_fold_key = (key, tuple(map(int, fold)))
                    if code_fold_key in seen_code_folds:
                        counters["duplicate_code_fold"] += 1
                        continue
                    seen_code_folds.add(code_fold_key)
                    seed = find_disjoint_logical_seed(
                        matrix_x,
                        matrix_z,
                        group_key=group_key,
                        translation_index=translation_index,
                        outer_shift=sector.outer_shift,
                    )
                    if seed is None:
                        counters["no_disjoint_logical_seed"] += 1
                        continue
                    counters["disjoint_logical_seed"] += 1
                    construction = {
                        "group": spec.name,
                        "L": L,
                        "J": J,
                        "duality_sector": sector.name,
                        "sector_relation": relation,
                        "fold_translation": list(translation),
                        "F": [
                            serialize_support(spec, support) for support in f_supports
                        ],
                        "G": [
                            serialize_support(spec, support) for support in g_supports
                        ],
                        "coefficient_weight_F": sum(map(len, f_supports)),
                        "maximum_check_weight": int(
                            max(
                                np.count_nonzero(matrix_x, axis=1).max(initial=0),
                                np.count_nonzero(matrix_z, axis=1).max(initial=0),
                            )
                        ),
                        "rank_x": rank_x,
                        "rank_z": rank_z,
                        "k": NUM_QUBITS - rank_x - rank_z,
                        "fold": {
                            **fold_analysis,
                            "permutation": fold.astype(int).tolist(),
                        },
                        "logical_grid": seed,
                    }
                    distance = (
                        certify_distance_six(matrix_x, matrix_z)
                        if certify_distance
                        else None
                    )
                    construction["distance"] = distance
                    if distance and distance["certified_distance_at_least_six"]:
                        counters["accepted"] += 1
                        accepted.append(construction)
                    else:
                        counters["distance_below_six"] += 1
                        if len(near_misses) < maximum_saved_near_misses:
                            near_misses.append(construction)

    return {
        "schema_version": SCHEMA_VERSION,
        "family": "paper-faithful-abelian-l4-j2",
        "group": spec.name,
        "target": "[[32,4,6]]",
        "maximum_check_weight": maximum_check_weight,
        "maximum_f_weight": maximum_f_weight,
        "counters": dict(sorted(counters.items())),
        "accepted": accepted,
        "near_misses": near_misses,
        "seconds": round(time.perf_counter() - started, 6),
    }


def build_code_from_record(record: dict[str, Any]) -> codes.CSSCode:
    """Reconstruct a qLDPC ``CSSCode`` from one serialized search record."""
    group_key = next(key for key, spec in GROUP_SPECS.items() if spec.name == record["group"])
    spec = GROUP_SPECS[group_key]
    supports = []
    for name in ("F", "G"):
        supports.extend(
            tuple(spec.index(element) for element in polynomial)
            for polynomial in record[name]
        )
    matrix_x, matrix_z = build_checks(
        group_key,
        (supports[0], supports[1]),
        (supports[2], supports[3]),
    )
    return codes.CSSCode(matrix_x, matrix_z)
