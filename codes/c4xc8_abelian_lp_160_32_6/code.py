"""Construction of the connected C4 x C8 automorphism-fold [[160,32,6]] code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODS = (4, 8)
GROUP = tuple((aa, bb) for aa in range(MODS[0]) for bb in range(MODS[1]))
GROUP_ORDER = len(GROUP)
NUM_QUBITS = 5 * GROUP_ORDER

A_SUPPORT = ((0, 0), (3, 3), (3, 6))
B_SUPPORT = ((0, 0), (2, 4), (3, 0))
PHI_X = (3, 2)
PHI_Y = (0, 1)


def group_index(element: tuple[int, int]) -> int:
    return (element[0] % MODS[0]) * MODS[1] + element[1] % MODS[1]


def group_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return ((left[0] + right[0]) % MODS[0], (left[1] + right[1]) % MODS[1])


def group_neg(element: tuple[int, int]) -> tuple[int, int]:
    return ((-element[0]) % MODS[0], (-element[1]) % MODS[1])


def group_scale(coefficient: int, element: tuple[int, int]) -> tuple[int, int]:
    return ((coefficient * element[0]) % MODS[0], (coefficient * element[1]) % MODS[1])


def phi(element: tuple[int, int]) -> tuple[int, int]:
    return group_add(group_scale(element[0], PHI_X), group_scale(element[1], PHI_Y))


def theta(element: tuple[int, int]) -> tuple[int, int]:
    return phi(group_neg(element))


def transformed_antipode(support) -> tuple[tuple[int, int], ...]:
    return tuple(phi(group_neg(element)) for element in support)


C_SUPPORT = transformed_antipode(A_SUPPORT)
D_SUPPORT = transformed_antipode(B_SUPPORT)


def group_matrix(support) -> np.ndarray:
    """Return the regular group-algebra matrix with row g supported on g+support."""
    matrix = np.zeros((GROUP_ORDER, GROUP_ORDER), dtype=np.uint8)
    for row, element in enumerate(GROUP):
        for term in support:
            matrix[row, group_index(group_add(element, term))] ^= 1
    return matrix


def build_checks() -> tuple[np.ndarray, np.ndarray]:
    aa, bb, cc, dd = map(group_matrix, (A_SUPPORT, B_SUPPORT, C_SUPPORT, D_SUPPORT))
    zero = np.zeros_like(aa)
    hx = np.block([[aa, zero, bb, zero, cc.T], [zero, aa, zero, bb, dd.T]])
    hz = np.block([[cc, dd, zero, zero, aa.T], [zero, zero, cc, dd, bb.T]])
    return hx, hz


def gf2_rank(matrix: np.ndarray) -> int:
    pivots: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    augmented = np.concatenate(
        [np.asarray(matrix, dtype=np.uint8) % 2, np.eye(size, dtype=np.uint8)], axis=1
    )
    row = 0
    for column in range(size):
        candidates = np.flatnonzero(augmented[row:, column])
        if not len(candidates):
            raise ValueError("matrix is singular over GF(2)")
        pivot = row + int(candidates[0])
        augmented[[row, pivot]] = augmented[[pivot, row]]
        for other in np.flatnonzero(augmented[:, column]):
            other = int(other)
            if other != row:
                augmented[other] ^= augmented[row]
        row += 1
    return augmented[:, size:]


def build_logical_bases() -> tuple[np.ndarray, np.ndarray]:
    """Return the systematic symplectic Z and X logical bases."""
    aa, bb, cc, dd = map(group_matrix, (A_SUPPORT, B_SUPPORT, C_SUPPORT, D_SUPPORT))
    p_left = (aa.T @ gf2_inverse(bb.T)) % 2
    p_right = (cc.T @ gf2_inverse(dd.T)) % 2
    zero = np.zeros((GROUP_ORDER, GROUP_ORDER), dtype=np.uint8)
    one = np.eye(GROUP_ORDER, dtype=np.uint8)
    logical_z = np.hstack([one, zero, p_left, zero, zero])
    logical_x = np.hstack([one, p_right, zero, zero, zero])
    return logical_z, logical_x


def translation_permutation(shift: tuple[int, int]) -> np.ndarray:
    within = np.asarray(
        [group_index(group_add(element, shift)) for element in GROUP], dtype=np.int64
    )
    return np.concatenate(
        [block * GROUP_ORDER + within for block in range(5)]
    ).astype(np.int64)


def fold_permutation() -> np.ndarray:
    within = [group_index(theta(element)) for element in GROUP]
    block_map = (0, 2, 1, 3, 4)
    return np.asarray(
        [
            block_map[block] * GROUP_ORDER + within[index]
            for block in range(5)
            for index in range(GROUP_ORDER)
        ],
        dtype=np.int64,
    )


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def presentation() -> dict:
    return {
        "schema_version": 1,
        "parameters": {"n": 160, "k": 32, "d": 6},
        "construction": "abelian five-block lifted/balanced product",
        "group": {"name": "C4 x C8", "orders": list(MODS)},
        "indexing": "qubit = 32*block + 8*x_exponent + y_exponent",
        "blocks": 5,
        "a": [list(term) for term in A_SUPPORT],
        "b": [list(term) for term in B_SUPPORT],
        "c": [list(term) for term in C_SUPPORT],
        "d": [list(term) for term in D_SUPPORT],
        "phi": {"x": list(PHI_X), "y": list(PHI_Y)},
        "theta_formula": "theta(r,s) = (r, -2*r-s mod 8)",
        "translations": {"C4": [1, 0], "C8": [0, 1]},
        "fold_block_map_zero_based": [0, 2, 1, 3, 4],
        "source_construction": "arXiv:2607.27644v1",
        "novelty": "This explicit abelian C4 x C8 instance is not listed in the source paper.",
    }


def write_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    hx, hz = build_checks()
    logical_z, logical_x = build_logical_bases()
    np.savez_compressed(directory / "checks.npz", hx=hx, hz=hz)
    np.savez_compressed(
        directory / "logical_bases.npz",
        logical_z=logical_z,
        logical_x=logical_x,
        translation_c4=translation_permutation((1, 0)),
        translation_c8=translation_permutation((0, 1)),
        fold=fold_permutation(),
    )
    (directory / "presentation.json").write_text(
        json.dumps(presentation(), indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-artifacts", type=Path)
    args = parser.parse_args()
    hx, hz = build_checks()
    logical_z, logical_x = build_logical_bases()
    result = {
        "n": int(hx.shape[1]),
        "rank_x": gf2_rank(hx),
        "rank_z": gf2_rank(hz),
        "k": int(hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)),
        "commutes": bool(not np.any((hx @ hz.T) % 2)),
        "check_weights_x": sorted(set(map(int, hx.sum(axis=1)))),
        "check_weights_z": sorted(set(map(int, hz.sum(axis=1)))),
        "logical_pairing_identity": bool(
            np.array_equal(
                (logical_z @ logical_x.T) % 2,
                np.eye(GROUP_ORDER, dtype=np.uint8),
            )
        ),
    }
    if args.write_artifacts is not None:
        write_artifacts(args.write_artifacts)
        result["wrote_artifacts"] = str(args.write_artifacts)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
