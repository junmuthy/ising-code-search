"""Candidate-specific ZX-fold and logical-action analysis for the S3 code."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli

from .s3_ising import (
    BOTTOM_ORDER,
    BOTTOM_X_ORDER,
    BOTTOM_Y_ORDER,
    TOP_DEGREE,
    ProductMonomial,
    S3L4W16Candidate,
    _translation_permutations,
    _zx_fold_permutation,
    build_s3_l4_w16_code,
    logical_translation_orbit,
    s3_internal_translation_orbits,
)

CERTIFIED_SEED_SUPPORT = (32, 75, 187, 261, 348, 373)


def certified_compact_candidate() -> S3L4W16Candidate:
    """Return the saved ``[[384,194,6]]`` candidate."""
    pp = ProductMonomial
    return S3L4W16Candidate(
        f0=tuple(
            sorted((pp(0, 0, 0), pp(0, 0, 2), pp(1, 5, 3), pp(5, 7, 0)))
        ),
        f1=tuple(
            sorted((pp(0, 4, 3), pp(0, 7, 3), pp(2, 7, 2), pp(5, 3, 0)))
        ),
    )


def build_certified_compact_code() -> codes.GALACode:
    return build_s3_l4_w16_code(certified_compact_candidate())


def certified_logical_grid() -> np.ndarray:
    seed = np.zeros(4 * TOP_DEGREE * BOTTOM_ORDER, dtype=np.uint8)
    seed[list(CERTIFIED_SEED_SUPPORT)] = 1
    return logical_translation_orbit(seed, num_blocks=4)


def gf2_rank(matrix: np.ndarray, field: Any) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=int).view(field)))


def permute_rows(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Apply an old-index to new-index qubit permutation to row vectors."""
    transformed = np.zeros_like(matrix)
    transformed[:, permutation] = matrix
    return transformed


def bottom_automorphisms() -> tuple[tuple[int, int, int, int], ...]:
    """All automorphisms of ``C_8 x C_4`` by generator images.

    A tuple ``(a,b,c,d)`` denotes ``x -> (a,b)`` and ``y -> (c,d)``.
    """
    automorphisms = []
    for image_x in itertools.product(range(BOTTOM_X_ORDER), range(BOTTOM_Y_ORDER)):
        for image_y in itertools.product(
            range(0, BOTTOM_X_ORDER, 2), range(BOTTOM_Y_ORDER)
        ):
            image = {
                (
                    (xx * image_x[0] + yy * image_y[0]) % BOTTOM_X_ORDER,
                    (xx * image_x[1] + yy * image_y[1]) % BOTTOM_Y_ORDER,
                )
                for xx in range(BOTTOM_X_ORDER)
                for yy in range(BOTTOM_Y_ORDER)
            }
            if len(image) == BOTTOM_ORDER:
                automorphisms.append((*image_x, *image_y))
    return tuple(automorphisms)


def _bottom_coordinate_data() -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
    """Coordinate lookup ``(top,a,b) <-> lift index`` on one 96-qubit block."""
    perm_x, perm_y = _translation_permutations()
    internal_orbits = s3_internal_translation_orbits(1)
    coordinates = np.empty(
        (TOP_DEGREE, BOTTOM_X_ORDER, BOTTOM_Y_ORDER), dtype=int
    )
    inverse: dict[int, tuple[int, int, int]] = {}
    for top, orbit in enumerate(internal_orbits):
        base = orbit[0]
        translated_x = base
        for xx in range(BOTTOM_X_ORDER):
            translated_y = translated_x
            for yy in range(BOTTOM_Y_ORDER):
                coordinates[top, xx, yy] = translated_y
                inverse[int(translated_y)] = (top, xx, yy)
                translated_y = int(perm_y[translated_y])
            translated_x = int(perm_x[translated_x])
    if len(inverse) != TOP_DEGREE * BOTTOM_ORDER:
        raise RuntimeError("failed to label every lift coordinate")
    return coordinates, inverse


def internal_affine_permutation(
    top_permutation: Sequence[int],
    bottom_automorphism: Sequence[int],
    *,
    translation: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Build a permutation of one natural ``S3 x C8 x C4`` lift block."""
    coordinates, inverse = _bottom_coordinate_data()
    image_x_x, image_x_y, image_y_x, image_y_y = bottom_automorphism
    output = np.empty(TOP_DEGREE * BOTTOM_ORDER, dtype=int)
    for old_index, (top, xx, yy) in inverse.items():
        new_x = (
            xx * image_x_x + yy * image_y_x + translation[0]
        ) % BOTTOM_X_ORDER
        new_y = (
            xx * image_x_y + yy * image_y_y + translation[1]
        ) % BOTTOM_Y_ORDER
        output[old_index] = coordinates[top_permutation[top], new_x, new_y]
    return output


def structured_physical_permutation(
    block_permutation: Sequence[int], internal_permutation: np.ndarray
) -> np.ndarray:
    block_size = TOP_DEGREE * BOTTOM_ORDER
    return np.hstack(
        [
            block_permutation[block] * block_size + internal_permutation
            for block in range(len(block_permutation))
        ]
    ).astype(int)


def _packed_rowspace_basis(matrix: np.ndarray) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = sum(1 << int(index) for index in np.flatnonzero(row))
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def _rows_lie_in_permuted_rowspace(
    row_supports: Sequence[np.ndarray],
    permutation: np.ndarray,
    target_basis: dict[int, int],
) -> bool:
    mapped_bits = [1 << int(index) for index in permutation]
    for support in row_supports:
        value = 0
        for index in support:
            value ^= mapped_bits[int(index)]
        while value:
            pivot = value.bit_length() - 1
            reducer = target_basis.get(pivot)
            if reducer is None:
                return False
            value ^= reducer
    return True


def is_zx_fold(
    matrix_x: np.ndarray,
    matrix_z: np.ndarray,
    permutation: np.ndarray,
    *,
    basis_x: dict[int, int] | None = None,
    basis_z: dict[int, int] | None = None,
    supports_x: Sequence[np.ndarray] | None = None,
    supports_z: Sequence[np.ndarray] | None = None,
) -> bool:
    basis_x = basis_x or _packed_rowspace_basis(matrix_x)
    basis_z = basis_z or _packed_rowspace_basis(matrix_z)
    supports_x = supports_x or tuple(np.flatnonzero(row) for row in matrix_x)
    supports_z = supports_z or tuple(np.flatnonzero(row) for row in matrix_z)
    return _rows_lie_in_permuted_rowspace(
        supports_x, permutation, basis_z
    ) and _rows_lie_in_permuted_rowspace(supports_z, permutation, basis_x)


def grid_pairing(grid_z: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return (grid_z @ permute_rows(grid_z, permutation).T) % 2


def permutation_order(permutation: np.ndarray) -> int:
    visited = np.zeros(len(permutation), dtype=bool)
    order = 1
    for start in range(len(permutation)):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            length += 1
            current = int(permutation[current])
        order = int(np.lcm(order, length))
    return order


def enumerate_structured_zx_folds(
    code: codes.CSSCode,
    grid_z: np.ndarray,
    *,
    progress: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate global structured folds modulo bottom translations."""
    matrix_x = np.asarray(code.matrix_x, dtype=np.uint8)
    matrix_z = np.asarray(code.matrix_z, dtype=np.uint8)
    basis_x = _packed_rowspace_basis(matrix_x)
    basis_z = _packed_rowspace_basis(matrix_z)
    supports_x = tuple(np.flatnonzero(row) for row in matrix_x)
    supports_z = tuple(np.flatnonzero(row) for row in matrix_z)
    automorphisms = bottom_automorphisms()
    identity_bottom = (1, 0, 0, 1)
    current_fold = _zx_fold_permutation(4)
    folds: list[dict[str, Any]] = []
    tested = 0
    first_direction_survivors = 0

    for block_permutation in itertools.permutations(range(4)):
        for top_permutation in itertools.permutations(range(TOP_DEGREE)):
            for bottom_map in automorphisms:
                tested += 1
                internal = internal_affine_permutation(
                    top_permutation, bottom_map
                )
                permutation = structured_physical_permutation(
                    block_permutation, internal
                )
                if not _rows_lie_in_permuted_rowspace(
                    supports_x, permutation, basis_z
                ):
                    continue
                first_direction_survivors += 1
                if not _rows_lie_in_permuted_rowspace(
                    supports_z, permutation, basis_x
                ):
                    continue
                pairing = grid_pairing(grid_z, permutation)
                folds.append(
                    {
                        "block_permutation": list(block_permutation),
                        "top_permutation": list(top_permutation),
                        "bottom_automorphism": list(bottom_map),
                        "grid_pairing_rank": gf2_rank(pairing, code.field),
                        "grid_pairing_weight": int(np.count_nonzero(pairing)),
                        "permutation_order": permutation_order(permutation),
                        "is_original_fold": bool(
                            np.array_equal(permutation, current_fold)
                            and bottom_map == identity_bottom
                        ),
                    }
                )
        if progress is not None:
            progress(tested, len(folds))
    return folds, {
        "tested": tested,
        "bottom_automorphisms": len(automorphisms),
        "first_direction_survivors": first_direction_survivors,
        "zx_folds": len(folds),
        "translations_quotiented": BOTTOM_ORDER,
    }


def logical_symplectic_action(
    code: codes.CSSCode, permutation: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the full logical symplectic matrix induced by H plus permutation."""
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    if not np.array_equal(
        (logical_x @ logical_z.T) % 2,
        np.eye(code.dimension, dtype=np.uint8),
    ):
        raise RuntimeError("qLDPC logical bases are not canonically paired")
    image_x_as_z = permute_rows(logical_x, permutation)
    image_z_as_x = permute_rows(logical_z, permutation)
    x_to_z = (image_x_as_z @ logical_x.T) % 2
    z_to_x = (image_z_as_x @ logical_z.T) % 2
    zero = np.zeros((code.dimension, code.dimension), dtype=np.uint8)
    symplectic = np.block([[zero, x_to_z], [z_to_x, zero]]).astype(np.uint8)
    return {
        "logical_x": logical_x,
        "logical_z": logical_z,
        "x_to_z": x_to_z,
        "z_to_x": z_to_x,
        "symplectic": symplectic,
    }


def rowspace_basis(matrix: np.ndarray, field: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=int).view(field)
    return np.asarray(array.row_space(), dtype=np.uint8)


def invariant_closure(
    generators: np.ndarray, action: np.ndarray, field: Any
) -> np.ndarray:
    basis = rowspace_basis(generators, field)
    while True:
        expanded = rowspace_basis(np.vstack([basis, (basis @ action) % 2]), field)
        if expanded.shape[0] == basis.shape[0]:
            return basis
        basis = expanded


def _right_inverse(matrix: np.ndarray, field: Any) -> np.ndarray:
    """Return ``R`` with ``matrix @ R = I`` for a full-row-rank GF(2) matrix."""
    array = np.asarray(matrix, dtype=int).view(field)
    reduced = np.asarray(array.row_reduce(), dtype=np.uint8)
    pivots = [int(np.flatnonzero(row)[0]) for row in reduced]
    square = np.asarray(array[:, pivots], dtype=int).view(field)
    inverse = np.linalg.inv(square)
    output = np.zeros((matrix.shape[1], matrix.shape[0]), dtype=np.uint8)
    output[pivots, :] = np.asarray(inverse, dtype=np.uint8)
    if not np.array_equal((matrix @ output) % 2, np.eye(matrix.shape[0], dtype=np.uint8)):
        raise RuntimeError("failed to construct a GF(2) right inverse")
    return output


def anchored_hyperbolic_basis(
    anchored_isotropic: np.ndarray,
    alternating_form: np.ndarray,
    field: Any,
) -> dict[str, np.ndarray]:
    """Extend a prescribed isotropic subspace to a hyperbolic basis.

    The returned U/V bases obey ``U B U.T = V B V.T = 0`` and
    ``U B V.T = I``.  The anchored rows are the first rows of U without any
    basis change, so the certified translation grid is preserved exactly.
    """
    aa = np.asarray(anchored_isotropic, dtype=np.uint8)
    bb = np.asarray(alternating_form, dtype=np.uint8)
    if np.any((aa @ bb @ aa.T) % 2):
        raise ValueError("the anchored subspace is not isotropic")
    right_inverse = _right_inverse((aa @ bb) % 2, field)
    partners = right_inverse.T
    partner_form = (partners @ bb @ partners.T) % 2
    correction = np.tril(partner_form, k=-1).astype(np.uint8)
    partners = (partners + correction @ aa) % 2
    anchored_size = aa.shape[0]
    if not np.array_equal(
        (aa @ bb @ partners.T) % 2,
        np.eye(anchored_size, dtype=np.uint8),
    ) or np.any((partners @ bb @ partners.T) % 2):
        raise RuntimeError("failed to construct isotropic anchored partners")

    anchored_hyperbolic = np.vstack([aa, partners])
    orthogonal = np.asarray(
        np.asarray((anchored_hyperbolic @ bb) % 2, dtype=int)
        .view(field)
        .null_space(),
        dtype=np.uint8,
    )
    remaining = [row.copy() for row in orthogonal]
    complement_u: list[np.ndarray] = []
    complement_v: list[np.ndarray] = []
    while remaining:
        uu = remaining.pop(0)
        partner_index = next(
            (
                index
                for index, vv in enumerate(remaining)
                if int((uu @ bb @ vv) % 2) == 1
            ),
            None,
        )
        if partner_index is None:
            raise RuntimeError("orthogonal complement is unexpectedly degenerate")
        vv = remaining.pop(partner_index)
        new_remaining = []
        for ww in remaining:
            adjusted = (
                ww
                + int((ww @ bb @ vv) % 2) * uu
                + int((ww @ bb @ uu) % 2) * vv
            ) % 2
            new_remaining.append(adjusted.astype(np.uint8))
        remaining = new_remaining
        complement_u.append(uu)
        complement_v.append(vv)

    basis_u = np.vstack([aa, *complement_u])
    basis_v = np.vstack([partners, *complement_v])
    transform = np.vstack([basis_u, basis_v])
    num_pairs = basis_u.shape[0]
    hyperbolic_form = np.block(
        [
            [np.zeros((num_pairs, num_pairs), dtype=np.uint8), np.eye(num_pairs, dtype=np.uint8)],
            [np.eye(num_pairs, dtype=np.uint8), np.zeros((num_pairs, num_pairs), dtype=np.uint8)],
        ]
    )
    if transform.shape != bb.shape or not np.array_equal(
        (transform @ bb @ transform.T) % 2, hyperbolic_form
    ):
        raise RuntimeError("failed to complete the hyperbolic basis")
    return {
        "basis_u": basis_u,
        "basis_v": basis_v,
        "anchored_partners": partners,
        "transform": transform,
        "hyperbolic_form": hyperbolic_form,
    }


def analyze_logical_action(
    code: codes.CSSCode,
    permutation: np.ndarray,
    grid_z: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    logical = logical_symplectic_action(code, permutation)
    action = logical["symplectic"]
    dimension = code.dimension
    identity = np.eye(2 * dimension, dtype=np.uint8)
    symplectic_form = np.block(
        [
            [np.zeros((dimension, dimension), dtype=np.uint8), np.eye(dimension, dtype=np.uint8)],
            [np.eye(dimension, dtype=np.uint8), np.zeros((dimension, dimension), dtype=np.uint8)],
        ]
    )
    grid_coordinates_z = (grid_z @ logical["logical_x"].T) % 2
    grid_coordinates = np.hstack(
        [np.zeros_like(grid_coordinates_z), grid_coordinates_z]
    )
    closure = invariant_closure(grid_coordinates, action, code.field)
    closure_pairing = (closure @ symplectic_form @ closure.T) % 2
    action_quadratic_matrix = (symplectic_form @ action.T) % 2
    action_quadratic_form_zero = bool(
        np.array_equal(action_quadratic_matrix, action_quadratic_matrix.T)
        and not np.any(np.diag(action_quadratic_matrix))
    )
    image_grid = (grid_coordinates @ action) % 2
    combined = rowspace_basis(np.vstack([grid_coordinates, image_grid]), code.field)
    physical_image = permute_rows(grid_z, permutation)
    fold_form_z = (grid_z @ physical_image.T) % 2
    global_fold_form_z = (
        logical["logical_z"] @ permute_rows(logical["logical_z"], permutation).T
    ) % 2
    hyperbolic: dict[str, np.ndarray] | None = None
    partner_physical_z: np.ndarray | None = None
    canonical_z_to_x: np.ndarray | None = None
    canonical_x_to_z: np.ndarray | None = None
    canonical_hadamard_swap_verified: bool | None = None
    if not np.any(np.diag(global_fold_form_z)) and gf2_rank(
        global_fold_form_z, code.field
    ) == dimension:
        hyperbolic = anchored_hyperbolic_basis(
            grid_coordinates_z, global_fold_form_z, code.field
        )
        partner_physical_z = (
            hyperbolic["anchored_partners"] @ logical["logical_z"]
        ) % 2
        transform = np.asarray(hyperbolic["transform"], dtype=int).view(code.field)
        inverse_transform = np.linalg.inv(transform)
        conjugate_transform = np.asarray(inverse_transform.T, dtype=np.uint8)
        new_logical_z = (
            hyperbolic["transform"] @ logical["logical_z"]
        ) % 2
        new_logical_x = (conjugate_transform @ logical["logical_x"]) % 2
        if not np.array_equal(
            (new_logical_x @ new_logical_z.T) % 2,
            np.eye(dimension, dtype=np.uint8),
        ):
            raise RuntimeError("hyperbolic logical basis is not canonically paired")
        canonical_z_to_x = (
            permute_rows(new_logical_z, permutation) @ new_logical_z.T
        ) % 2
        canonical_x_to_z = (
            permute_rows(new_logical_x, permutation) @ new_logical_x.T
        ) % 2
        canonical_hadamard_swap_verified = bool(
            np.array_equal(canonical_z_to_x, hyperbolic["hyperbolic_form"])
            and np.array_equal(canonical_x_to_z, hyperbolic["hyperbolic_form"])
        )
    coordinate_image = np.hstack(
        [
            (physical_image @ logical["logical_z"].T) % 2,
            np.zeros_like(grid_coordinates_z),
        ]
    )
    if not np.array_equal(image_grid, coordinate_image):
        raise RuntimeError("logical action disagrees with the physical grid image")
    verification = {
        "is_symplectic": bool(
            np.array_equal(
                (action @ symplectic_form @ action.T) % 2, symplectic_form
            )
        ),
        "action_order_two": bool(not np.any((action @ action) % 2 ^ identity)),
        "action_rank": gf2_rank(action, code.field),
        "fixed_space_dimension": 2 * dimension
        - gf2_rank(action ^ identity, code.field),
        "nilpotent_part_rank": gf2_rank(action ^ identity, code.field),
        "action_quadratic_form_zero": action_quadratic_form_zero,
        "grid_logical_rank": gf2_rank(grid_coordinates, code.field),
        "grid_image_logical_rank": gf2_rank(image_grid, code.field),
        "grid_image_intersection_dimension": 64 - combined.shape[0],
        "grid_orbit_closure_dimension": int(closure.shape[0]),
        "grid_orbit_closure_symplectic_rank": gf2_rank(
            closure_pairing, code.field
        ),
        "grid_orbit_closure_radical_dimension": int(
            closure.shape[0] - gf2_rank(closure_pairing, code.field)
        ),
        "nondegenerate_extension_dimension_lower_bound": int(
            2 * closure.shape[0] - gf2_rank(closure_pairing, code.field)
        ),
        "grid_fold_pairing_rank": gf2_rank(fold_form_z, code.field),
        "global_z_fold_form_rank": gf2_rank(global_fold_form_z, code.field),
        "global_z_fold_form_symmetric": bool(
            np.array_equal(global_fold_form_z, global_fold_form_z.T)
        ),
        "global_z_fold_form_alternating": bool(
            not np.any(np.diag(global_fold_form_z))
        ),
        "global_z_fold_form_diagonal_weight": int(
            np.count_nonzero(np.diag(global_fold_form_z))
        ),
        "hadamard_swap_pairs": dimension // 2 if hyperbolic is not None else None,
        "canonical_hadamard_swap_verified": canonical_hadamard_swap_verified,
        "grid_partner_count": (
            int(hyperbolic["anchored_partners"].shape[0])
            if hyperbolic is not None
            else None
        ),
        "grid_partner_z_weight_min": (
            int(np.count_nonzero(partner_physical_z, axis=1).min())
            if partner_physical_z is not None
            else None
        ),
        "grid_partner_z_weight_max": (
            int(np.count_nonzero(partner_physical_z, axis=1).max())
            if partner_physical_z is not None
            else None
        ),
        "grid_partner_z_weight_mean": (
            float(np.count_nonzero(partner_physical_z, axis=1).mean())
            if partner_physical_z is not None
            else None
        ),
        "grid_partner_z_pairwise_disjoint": (
            bool(np.all(np.sum(partner_physical_z, axis=0) <= 1))
            if partner_physical_z is not None
            else None
        ),
    }
    arrays = {
        **logical,
        "physical_permutation": permutation,
        "symplectic_form": symplectic_form,
        "action_quadratic_matrix": action_quadratic_matrix,
        "grid_coordinates": grid_coordinates,
        "grid_image_coordinates": image_grid,
        "grid_orbit_closure": closure,
        "grid_orbit_closure_pairing": closure_pairing,
        "global_z_fold_form": global_fold_form_z,
        "grid_z_fold_form": fold_form_z,
    }
    if hyperbolic is not None and partner_physical_z is not None:
        arrays.update(
            {
                "hyperbolic_basis_u": hyperbolic["basis_u"],
                "hyperbolic_basis_v": hyperbolic["basis_v"],
                "hyperbolic_basis_transform": hyperbolic["transform"],
                "hyperbolic_form": hyperbolic["hyperbolic_form"],
                "grid_partner_z_coordinates": hyperbolic["anchored_partners"],
                "grid_partner_physical_z": partner_physical_z,
                "canonical_z_to_x": canonical_z_to_x,
                "canonical_x_to_z": canonical_x_to_z,
            }
        )
    return verification, arrays
