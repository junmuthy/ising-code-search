#!/usr/bin/env python3
"""Test the published self-dual `[[64,8,8]]` BB code for Ising grids.

Source: Liang and Chen, arXiv:2510.05211v2.  The code uses
`f=1+x+y+y^-1` on the twisted torus `x^4 y^4=1, y^8=1`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from typing import Any, Iterable

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

from .logicals import batch_disjointness, dress_logicals_for_batches
from .model import gf2_rank
from .run_faithful_twisted_search import atomic_json

Element = tuple[int, int]
ELEMENTS: tuple[Element, ...] = tuple(
    (xx, yy) for xx in range(4) for yy in range(8)
)
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
IDENTITY = (0, 0)


def add(left: Element, right: Element) -> Element:
    raw_x = left[0] + right[0]
    return raw_x % 4, (left[1] + right[1] + 4 * (raw_x // 4)) % 8


def scale(multiplier: int, element: Element) -> Element:
    if multiplier < 0:
        return scale(-multiplier, inverse(element))
    output = IDENTITY
    for _ in range(multiplier):
        output = add(output, element)
    return output


def inverse(element: Element) -> Element:
    return next(candidate for candidate in ELEMENTS if add(element, candidate) == IDENTITY)


def affine_permutation(
    image_x: Element,
    image_y: Element,
    translation: Element = IDENTITY,
) -> np.ndarray:
    return np.asarray(
        [
            INDEX[
                add(
                    translation,
                    add(scale(xx, image_x), scale(yy, image_y)),
                )
            ]
            for xx, yy in ELEMENTS
        ],
        dtype=int,
    )


def linear_automorphisms() -> Iterable[tuple[Element, Element, np.ndarray]]:
    for image_x, image_y in itertools.product(ELEMENTS, repeat=2):
        permutation = affine_permutation(image_x, image_y)
        if len(set(map(int, permutation))) == len(ELEMENTS):
            yield image_x, image_y, permutation


def translation_permutation(element: Element) -> np.ndarray:
    return affine_permutation((1, 0), (0, 1), element)


def translation_equivalent(
    source: set[Element], target: set[Element]
) -> bool:
    return any(
        {add(translation, element) for element in source} == target
        for translation in ELEMENTS
    )


def permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    output = np.zeros((len(permutation), len(permutation)), dtype=np.uint8)
    output[np.arange(len(permutation)), permutation] = 1
    return output


def published_code() -> codes.CSSCode:
    px = permutation_matrix(translation_permutation((1, 0)))
    py = permutation_matrix(translation_permutation((0, 1)))
    polynomial = (
        np.eye(32, dtype=np.uint8)
        + px
        + py
        + py.T
    ) % 2
    check = np.hstack([polynomial, polynomial.T])
    return codes.CSSCode(check, check, promise_equal_distance_xz=True)


def permute_columns(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def full_permutation(
    bottom: np.ndarray, *, swap_halves: bool
) -> np.ndarray:
    if swap_halves:
        return np.hstack([32 + bottom, bottom]).astype(int)
    return np.hstack([bottom, 32 + bottom]).astype(int)


def matrix_order(matrix: np.ndarray, maximum: int = 32) -> int | None:
    identity = np.eye(len(matrix), dtype=np.uint8)
    power = identity.copy()
    for exponent in range(1, maximum + 1):
        power = (power @ matrix) % 2
        if np.array_equal(power, identity):
            return exponent
    return None


def permutation_order(permutation: np.ndarray, maximum: int = 64) -> int | None:
    identity = np.arange(len(permutation))
    power = identity.copy()
    for exponent in range(1, maximum + 1):
        power = permutation[power]
        if np.array_equal(power, identity):
            return exponent
    return None


def induced_action(
    code: codes.CSSCode,
    permutation: np.ndarray,
    logical_z: np.ndarray,
    logical_x: np.ndarray,
    pairing_inverse: np.ndarray,
) -> np.ndarray:
    translated = np.zeros_like(logical_z)
    translated[:, permutation] = logical_z
    return ((translated @ logical_x.T) @ pairing_inverse) % 2


def action_algebra(
    tx: np.ndarray, ty: np.ndarray, field: type[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    actions = []
    power_x = np.eye(len(tx), dtype=np.uint8)
    for _xx in range(4):
        power_y = np.eye(len(ty), dtype=np.uint8)
        for _yy in range(2):
            actions.append((power_x @ power_y) % 2)
            power_y = (power_y @ ty) % 2
        power_x = (power_x @ tx) % 2
    rank = gf2_rank(
        np.asarray(actions, dtype=np.uint8).reshape(8, -1),
        field,
    )
    return actions, rank


def orbit_from_class(
    vector: np.ndarray,
    actions: list[np.ndarray],
    logical_z: np.ndarray,
) -> np.ndarray:
    classes = np.asarray([(vector @ action) % 2 for action in actions], dtype=np.uint8)
    return (classes @ logical_z) % 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dressing-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    code = published_code()
    check = np.asarray(code.matrix_x, dtype=np.uint8)
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    pairing = (logical_z @ logical_x.T) % 2
    pairing_inverse = np.asarray(
        np.linalg.inv(pairing.view(code.field)), dtype=np.uint8
    )

    symmetries: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    base_support = {IDENTITY, (1, 0), (0, 1), inverse((0, 1))}
    inverse_support = {inverse(element) for element in base_support}
    linear_tested = 0
    for image_x, image_y, linear in linear_automorphisms():
        linear_tested += 1
        image_support = {
            IDENTITY,
            image_x,
            image_y,
            inverse(image_y),
        }
        allowed_swaps = []
        if translation_equivalent(image_support, base_support):
            allowed_swaps.append(False)
        if translation_equivalent(image_support, inverse_support):
            allowed_swaps.append(True)
        if not allowed_swaps:
            continue
        for translation in ELEMENTS:
            bottom = np.asarray(
                [
                    INDEX[add(translation, ELEMENTS[int(target)])]
                    for target in linear
                ],
                dtype=int,
            )
            for swap in allowed_swaps:
                permutation = full_permutation(bottom, swap_halves=swap)
                key = permutation.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                if gf2_rank(
                    np.vstack([check, permute_columns(check, permutation)]),
                    code.field,
                ) != code.code_x.rank:
                    continue
                action = induced_action(
                    code, permutation, logical_z, logical_x, pairing_inverse
                )
                symmetries.append(
                    {
                        "image_x": list(image_x),
                        "image_y": list(image_y),
                        "translation": list(translation),
                        "swap_halves": swap,
                        "physical_order": permutation_order(permutation),
                        "logical_order": matrix_order(action),
                        "permutation": permutation,
                        "action": action,
                    }
                )
        if linear_tested % 16 == 0:
            print(
                json.dumps(
                    {
                        "stage": "affine_symmetries",
                        "linear_automorphisms_tested": linear_tested,
                        "code_symmetries_found": len(symmetries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    order_four = [item for item in symmetries if item["logical_order"] == 4]
    order_two = [item for item in symmetries if item["logical_order"] == 2]
    grids: list[dict[str, Any]] = []
    for xx in order_four:
        for yy in order_two:
            if not np.array_equal(
                xx["permutation"][yy["permutation"]],
                yy["permutation"][xx["permutation"]],
            ):
                continue
            if not np.array_equal((xx["action"] @ yy["action"]) % 2, (yy["action"] @ xx["action"]) % 2):
                continue
            actions, algebra_rank = action_algebra(
                xx["action"], yy["action"], code.field
            )
            if algebra_rank != 8:
                continue
            print(
                json.dumps(
                    {
                        "stage": "regular_action_pair",
                        "order_four_checked": len(order_four),
                        "order_two_checked": len(order_two),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            for integer in range(1, 1 << code.dimension):
                vector = np.asarray(
                    [(integer >> bit) & 1 for bit in range(code.dimension)],
                    dtype=np.uint8,
                )
                orbit = orbit_from_class(vector, actions, logical_z)
                class_rank = gf2_rank(
                    np.asarray([(vector @ action) % 2 for action in actions]),
                    code.field,
                )
                if class_rank != 8:
                    continue
                pairing_rank = gf2_rank((orbit @ orbit.T) % 2, code.field)
                if pairing_rank != 8:
                    continue
                batches = batch_disjointness(orbit)
                dressed = []
                for name, raw in batches.items():
                    if raw["disjoint"]:
                        result = {
                            "scheme": name,
                            "feasible": True,
                            "already_disjoint_without_dressing": True,
                        }
                    else:
                        result = dress_logicals_for_batches(
                            code, orbit, name, time_limit=args.dressing_seconds
                        )
                    dressed.append(result)
                grids.append(
                    {
                        "x_symmetry": {
                            key: value
                            for key, value in xx.items()
                            if key not in {"permutation", "action"}
                        },
                        "y_symmetry": {
                            key: value
                            for key, value in yy.items()
                            if key not in {"permutation", "action"}
                        },
                        "seed_class": vector.astype(int).tolist(),
                        "orbit_supports": [
                            np.flatnonzero(row).astype(int).tolist() for row in orbit
                        ],
                        "orbit_weights": np.count_nonzero(orbit, axis=1).astype(int).tolist(),
                        "raw_batching": batches,
                        "dressed_batching": dressed,
                    }
                )
                np.savez_compressed(
                    args.output_dir / "published-bb64-checks-and-grid.npz",
                    matrix_x=check,
                    matrix_z=check,
                    grid_orbit=orbit,
                    grid_x_permutation=xx["permutation"],
                    grid_y_permutation=yy["permutation"],
                )
                break
            if grids:
                break
        if grids:
            break

    summary = {
        "source": "Liang and Chen, arXiv:2510.05211v2",
        "source_url": "https://arxiv.org/abs/2510.05211",
        "source_location": "Table 1, Figure 1, and Appendix A",
        "polynomial": "f=1+x+y+y^-1, g=f^dagger",
        "twisted_torus": ["x^4 y^4=1", "y^8=1"],
        "n": code.num_qubits,
        "k": code.dimension,
        "published_distance": 8,
        "check_weights": sorted(set(map(int, np.count_nonzero(check, axis=1)))),
        "affine_code_symmetries": len(symmetries),
        "logical_order_four_symmetries": len(order_four),
        "logical_order_two_symmetries": len(order_two),
        "regular_c4xc2_grid_found": bool(grids),
        "grid": grids[0] if grids else None,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
