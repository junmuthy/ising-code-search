"""Exact GF(2) utilities for the relaxed k=8 C4 x C2 search.

The logical search works with quotient classes only.  No representative
disjointness, batching, or stabilizer dressing is performed anywhere here.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def rows_to_ints(matrix: np.ndarray) -> list[int]:
    output: list[int] = []
    for row in np.asarray(matrix, dtype=np.uint8):
        value = 0
        for index in np.flatnonzero(row):
            value |= 1 << int(index)
        output.append(value)
    return output


def ints_to_matrix(rows: Sequence[int], width: int) -> np.ndarray:
    return np.asarray(
        [[(row >> column) & 1 for column in range(width)] for row in rows],
        dtype=np.uint8,
    )


def rref(rows: Iterable[int], width: int) -> tuple[list[int], list[int]]:
    """Return an ordinary left-to-right binary RREF and its pivot columns."""
    work = [int(row) for row in rows if row]
    pivot_row = 0
    pivots: list[int] = []
    for column in range(width):
        selected = next(
            (index for index in range(pivot_row, len(work)) if (work[index] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        work[pivot_row], work[selected] = work[selected], work[pivot_row]
        pivot = work[pivot_row]
        for index in range(len(work)):
            if index != pivot_row and ((work[index] >> column) & 1):
                work[index] ^= pivot
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    return work[:pivot_row], pivots


def kernel_basis(check_rows: Sequence[int], width: int) -> list[int]:
    reduced, pivots = rref(check_rows, width)
    free = [column for column in range(width) if column not in set(pivots)]
    output = []
    for column in free:
        vector = 1 << column
        for row, pivot in zip(reduced, pivots, strict=True):
            if (row >> column) & 1:
                vector |= 1 << pivot
        output.append(vector)
    return output


def high_pivot_basis(rows: Iterable[int]) -> dict[int, int]:
    basis: dict[int, int] = {}
    for original in rows:
        row = int(original)
        while row:
            pivot = row.bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    return basis


def add_to_basis(row: int, basis: dict[int, int]) -> bool:
    value = int(row)
    while value:
        pivot = value.bit_length() - 1
        if pivot in basis:
            value ^= basis[pivot]
        else:
            basis[pivot] = value
            return True
    return False


def in_span(row: int, basis: dict[int, int]) -> bool:
    value = int(row)
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def quotient_basis(check_rows: Sequence[int], width: int) -> tuple[list[int], list[int]]:
    """Return independent stabilizers and a complement in their kernel."""
    stabilizers, _ = rref(check_rows, width)
    combined = high_pivot_basis(stabilizers)
    logicals: list[int] = []
    for vector in kernel_basis(stabilizers, width):
        if add_to_basis(vector, combined):
            logicals.append(vector)
    expected = width - 2 * len(stabilizers)
    if len(logicals) != expected:
        raise AssertionError((len(logicals), expected))
    if gf2_rank(pairing_rows(logicals, logicals)) != expected:
        raise AssertionError("logical quotient pairing is singular")
    return stabilizers, logicals


def css_logical_bases(
    check_x: Sequence[int], check_z: Sequence[int], width: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return independent CSS checks and complementary Z/X logical bases."""
    stabilizer_x, _ = rref(check_x, width)
    stabilizer_z, _ = rref(check_z, width)
    if any(parity(left & right) for left in stabilizer_x for right in stabilizer_z):
        raise ValueError("CSS checks do not commute")

    def complement(stabilizers: list[int], kernel_check: list[int]) -> list[int]:
        combined = high_pivot_basis(stabilizers)
        output: list[int] = []
        for vector in kernel_basis(kernel_check, width):
            if add_to_basis(vector, combined):
                output.append(vector)
        return output

    logical_z = complement(stabilizer_z, stabilizer_x)
    logical_x = complement(stabilizer_x, stabilizer_z)
    dimension = width - len(stabilizer_x) - len(stabilizer_z)
    if len(logical_z) != dimension or len(logical_x) != dimension:
        raise AssertionError("incorrect CSS quotient dimension")
    if gf2_rank(pairing_rows(logical_z, logical_x)) != dimension:
        raise AssertionError("singular CSS logical pairing")
    return stabilizer_x, stabilizer_z, logical_z, logical_x


def gf2_rank(rows: Iterable[int]) -> int:
    return len(high_pivot_basis(rows))


def parity(value: int) -> int:
    return value.bit_count() & 1


def pairing_rows(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return [
        sum(parity(row & column) << index for index, column in enumerate(right))
        for row in left
    ]


def is_permutation_matrix(rows: Sequence[int], size: int) -> bool:
    if len(rows) != size or any(row.bit_count() != 1 for row in rows):
        return False
    columns = [sum(((row >> column) & 1) for row in rows) for column in range(size)]
    return all(value == 1 for value in columns)


def combine_basis(mask: int, basis: Sequence[int]) -> int:
    output = 0
    for index, vector in enumerate(basis):
        if (mask >> index) & 1:
            output ^= vector
    return output


def permute_vector(vector: int, permutation: Sequence[int]) -> int:
    output = 0
    value = int(vector)
    while value:
        low = value & -value
        source = low.bit_length() - 1
        output |= 1 << int(permutation[source])
        value ^= low
    return output


def compose(left: Sequence[int], right: Sequence[int]) -> list[int]:
    """Composition left after right for old-to-new point maps."""
    return [int(left[int(right[index])]) for index in range(len(right))]


def permutation_order(permutation: Sequence[int]) -> int:
    identity = list(range(len(permutation)))
    power = identity
    for exponent in range(1, len(permutation) + 1):
        power = compose(permutation, power)
        if power == identity:
            return exponent
    raise AssertionError("invalid permutation")


def preserves_rowspace(
    check_rows: Sequence[int], permutation: Sequence[int], width: int
) -> bool:
    stabilizers, _ = rref(check_rows, width)
    basis = high_pivot_basis(stabilizers)
    return all(in_span(permute_vector(row, permutation), basis) for row in stabilizers)


def maps_rowspace(
    source_rows: Sequence[int], target_rows: Sequence[int], permutation: Sequence[int], width: int
) -> bool:
    source, _ = rref(source_rows, width)
    target, _ = rref(target_rows, width)
    if len(source) != len(target):
        return False
    basis = high_pivot_basis(target)
    return all(in_span(permute_vector(row, permutation), basis) for row in source)


def logical_signature(vector: int, logical_basis: Sequence[int]) -> int:
    return sum(parity(vector & row) << index for index, row in enumerate(logical_basis))


def find_regular_permutation_h_grid(
    check_rows: Sequence[int],
    width: int,
    translation_four: Sequence[int],
    translation_two: Sequence[int],
) -> dict[str, Any] | None:
    """Enumerate all nonzero logical seeds for one physical C4 x C2 pair.

    The CSS code is assumed exactly self-dual, so bare transversal Hadamard is
    the ZX fold and its logical Gram matrix is the physical overlap matrix.
    """
    stabilizers, logicals = quotient_basis(check_rows, width)
    if len(logicals) != 8:
        return None
    if compose(translation_four, translation_two) != compose(translation_two, translation_four):
        return None
    if not preserves_rowspace(stabilizers, translation_four, width):
        return None
    if not preserves_rowspace(stabilizers, translation_two, width):
        return None

    physical_order_four = permutation_order(translation_four)
    physical_order_two = permutation_order(translation_two)
    power_four_4 = list(range(width))
    power_two_2 = list(range(width))
    for _ in range(4):
        power_four_4 = compose(translation_four, power_four_4)
    for _ in range(2):
        power_two_2 = compose(translation_two, power_two_2)
    for vector in logicals:
        base_signature = logical_signature(vector, logicals)
        if logical_signature(permute_vector(vector, power_four_4), logicals) != base_signature:
            return None
        if logical_signature(permute_vector(vector, power_two_2), logicals) != base_signature:
            return None

    actions: list[list[int]] = []
    power_four = list(range(width))
    for _aa in range(4):
        for bb in range(2):
            actions.append(power_four if bb == 0 else compose(translation_two, power_four))
        power_four = compose(translation_four, power_four)

    # A cyclic vector can exist only if the eight group actions are linearly
    # independent as quotient endomorphisms.  This exact gate avoids scanning
    # 255 seeds for the common low-rank-action failure mode.
    action_images = [
        [permute_vector(vector, action) for vector in logicals]
        for action in actions
    ]
    action_matrices = [
        sum(
            logical_signature(image, logicals) << (8 * row)
            for row, image in enumerate(images)
        )
        for images in action_images
    ]
    if gf2_rank(action_matrices) != 8:
        return None

    for mask in range(1, 256):
        seed = combine_basis(mask, logicals)
        orbit = [combine_basis(mask, images) for images in action_images]
        signatures = [logical_signature(vector, logicals) for vector in orbit]
        if gf2_rank(signatures) != 8:
            continue
        pairing = pairing_rows(orbit, orbit)
        if not is_permutation_matrix(pairing, 8):
            continue
        return {
            "seed_class_mask": mask,
            "seed_support": bit_support(seed),
            "orbit_supports": [bit_support(vector) for vector in orbit],
            "orbit_weights": [vector.bit_count() for vector in orbit],
            "pairing_rows_bitmasks": pairing,
            "pairing": [[(row >> column) & 1 for column in range(8)] for row in pairing],
            "hadamard_permutation": [row.bit_length() - 1 for row in pairing],
            "logical_signatures": signatures,
            "stabilizer_rank": len(stabilizers),
            "physical_generator_orders": [physical_order_four, physical_order_two],
            "logical_generator_orders": [4, 2],
        }
    return None


def find_regular_permutation_h_grid_css(
    check_x: Sequence[int],
    check_z: Sequence[int],
    width: int,
    translation_four: Sequence[int],
    translation_two: Sequence[int],
    hadamard_fold: Sequence[int],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Test one known physical C4 x C2 action and one known ZX fold.

    This is the non-self-dual counterpart of
    :func:`find_regular_permutation_h_grid`.  All tests are on stabilizer row
    spaces and logical quotient classes, not on a particular row basis.
    """
    if diagnostics is None:
        diagnostics = {}
    diagnostics.clear()
    stabilizer_x, stabilizer_z, logical_z, logical_x = css_logical_bases(
        check_x, check_z, width
    )
    if len(logical_z) != 8:
        diagnostics["failure"] = "dimension"
        return None
    if compose(translation_four, translation_two) != compose(translation_two, translation_four):
        diagnostics["failure"] = "physical_generators_do_not_commute"
        return None
    for translation in (translation_four, translation_two):
        if not preserves_rowspace(stabilizer_x, translation, width):
            diagnostics["failure"] = "translation_not_x_automorphism"
            return None
        if not preserves_rowspace(stabilizer_z, translation, width):
            diagnostics["failure"] = "translation_not_z_automorphism"
            return None
    if not maps_rowspace(stabilizer_x, stabilizer_z, hadamard_fold, width):
        diagnostics["failure"] = "fold_does_not_map_x_to_z"
        return None
    if not maps_rowspace(stabilizer_z, stabilizer_x, hadamard_fold, width):
        diagnostics["failure"] = "fold_does_not_map_z_to_x"
        return None

    physical_order_four = permutation_order(translation_four)
    physical_order_two = permutation_order(translation_two)
    for translation, exponent in ((translation_four, 4), (translation_two, 2)):
        power = list(range(width))
        for _ in range(exponent):
            power = compose(translation, power)
        for vector in logical_z:
            if logical_signature(permute_vector(vector, power), logical_x) != logical_signature(
                vector, logical_x
            ):
                diagnostics["failure"] = f"logical_order_not_divisor_{exponent}"
                return None

    actions: list[list[int]] = []
    power_four = list(range(width))
    for _aa in range(4):
        actions.extend((power_four, compose(translation_two, power_four)))
        power_four = compose(translation_four, power_four)
    action_images = [
        [permute_vector(vector, action) for vector in logical_z]
        for action in actions
    ]
    action_matrices = [
        sum(
            logical_signature(image, logical_x) << (8 * row)
            for row, image in enumerate(images)
        )
        for images in action_images
    ]
    diagnostics["action_algebra_rank"] = gf2_rank(action_matrices)
    if diagnostics["action_algebra_rank"] != 8:
        diagnostics["failure"] = "action_algebra_rank"
        return None

    regular_seeds = 0
    for mask in range(1, 256):
        seed = combine_basis(mask, logical_z)
        orbit = [combine_basis(mask, images) for images in action_images]
        signatures = [logical_signature(vector, logical_x) for vector in orbit]
        if gf2_rank(signatures) != 8:
            continue
        regular_seeds += 1
        x_orbit = [permute_vector(vector, hadamard_fold) for vector in orbit]
        pairing = pairing_rows(orbit, x_orbit)
        if not is_permutation_matrix(pairing, 8):
            continue
        diagnostics["regular_seeds"] = regular_seeds
        diagnostics["failure"] = None
        return {
            "seed_class_mask": mask,
            "seed_support": bit_support(seed),
            "orbit_supports": [bit_support(vector) for vector in orbit],
            "orbit_weights": [vector.bit_count() for vector in orbit],
            "pairing_rows_bitmasks": pairing,
            "pairing": [[(row >> column) & 1 for column in range(8)] for row in pairing],
            "hadamard_permutation": [row.bit_length() - 1 for row in pairing],
            "logical_signatures": signatures,
            "rank_x": len(stabilizer_x),
            "rank_z": len(stabilizer_z),
            "physical_generator_orders": [physical_order_four, physical_order_two],
            "logical_generator_orders": [4, 2],
        }
    diagnostics["regular_seeds"] = regular_seeds
    diagnostics["failure"] = (
        "no_regular_seed" if regular_seeds == 0 else "hadamard_pairing_not_permutation"
    )
    return None


def bit_support(vector: int) -> list[int]:
    return [index for index in range(vector.bit_length()) if (vector >> index) & 1]


def column_signatures(rows: Sequence[int], width: int) -> list[int]:
    return [sum(((row >> column) & 1) << index for index, row in enumerate(rows)) for column in range(width)]


def find_logical_through_weight_five(
    check_rows: Sequence[int], width: int
) -> dict[str, Any] | None:
    """Return a minimum logical through weight five, or certify none exists."""
    stabilizers, logicals = quotient_basis(check_rows, width)
    syndromes = column_signatures(stabilizers, width)
    logical = column_signatures(logicals, width)

    def result(indices: Sequence[int]) -> dict[str, Any]:
        return {"weight": len(indices), "support": list(indices)}

    for aa in range(width):
        if syndromes[aa] == 0 and logical[aa] != 0:
            return result((aa,))

    pairs_by_syndrome: dict[int, list[tuple[int, int, int]]] = {}
    for aa, bb in itertools.combinations(range(width), 2):
        syndrome = syndromes[aa] ^ syndromes[bb]
        signature = logical[aa] ^ logical[bb]
        if syndrome == 0 and signature != 0:
            return result((aa, bb))
        pairs_by_syndrome.setdefault(syndrome, []).append((aa, bb, signature))

    for cc in range(width):
        for aa, bb, signature in pairs_by_syndrome.get(syndromes[cc], ()):
            if cc not in (aa, bb) and (signature ^ logical[cc]) != 0:
                return result((aa, bb, cc))

    for pairs in pairs_by_syndrome.values():
        for first_index, (aa, bb, first_signature) in enumerate(pairs):
            for cc, dd, second_signature in pairs[first_index + 1 :]:
                if len({aa, bb, cc, dd}) == 4 and (first_signature ^ second_signature) != 0:
                    return result((aa, bb, cc, dd))

    for aa, bb, cc in itertools.combinations(range(width), 3):
        syndrome = syndromes[aa] ^ syndromes[bb] ^ syndromes[cc]
        signature = logical[aa] ^ logical[bb] ^ logical[cc]
        for dd, ee, pair_signature in pairs_by_syndrome.get(syndrome, ()):
            if len({aa, bb, cc, dd, ee}) == 5 and (signature ^ pair_signature) != 0:
                return result((aa, bb, cc, dd, ee))
    return None


def find_css_z_logical_through_weight_five(
    check_x: Sequence[int], check_z: Sequence[int], width: int
) -> dict[str, Any] | None:
    """Return a Z logical of weight at most five, or certify none exists."""
    stabilizer_x, _stabilizer_z, _logical_z, logical_x = css_logical_bases(
        check_x, check_z, width
    )
    syndromes = column_signatures(stabilizer_x, width)
    logical = column_signatures(logical_x, width)

    def result(indices: Sequence[int]) -> dict[str, Any]:
        return {"weight": len(indices), "support": list(indices), "pauli": "Z"}

    for aa in range(width):
        if syndromes[aa] == 0 and logical[aa] != 0:
            return result((aa,))

    pairs_by_syndrome: dict[int, list[tuple[int, int, int]]] = {}
    for aa, bb in itertools.combinations(range(width), 2):
        syndrome = syndromes[aa] ^ syndromes[bb]
        signature = logical[aa] ^ logical[bb]
        if syndrome == 0 and signature != 0:
            return result((aa, bb))
        pairs_by_syndrome.setdefault(syndrome, []).append((aa, bb, signature))

    for cc in range(width):
        for aa, bb, signature in pairs_by_syndrome.get(syndromes[cc], ()):
            if cc not in (aa, bb) and (signature ^ logical[cc]) != 0:
                return result((aa, bb, cc))

    for pairs in pairs_by_syndrome.values():
        for first_index, (aa, bb, first_signature) in enumerate(pairs):
            for cc, dd, second_signature in pairs[first_index + 1 :]:
                if len({aa, bb, cc, dd}) == 4 and (first_signature ^ second_signature) != 0:
                    return result((aa, bb, cc, dd))

    for aa, bb, cc in itertools.combinations(range(width), 3):
        syndrome = syndromes[aa] ^ syndromes[bb] ^ syndromes[cc]
        signature = logical[aa] ^ logical[bb] ^ logical[cc]
        for dd, ee, pair_signature in pairs_by_syndrome.get(syndrome, ()):
            if len({aa, bb, cc, dd, ee}) == 5 and (signature ^ pair_signature) != 0:
                return result((aa, bb, cc, dd, ee))
    return None


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
