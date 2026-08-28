"""Geometry-first ``n=128`` batched half-grid search.

The represented lift is ``GL(2,2) x C8 x C4`` and the minimal two-block
GALA layout has four physical ``C8 x C4`` translation sheets.  The target
logical supports form one translated 32-site grid.  Red and black logical
sites are injected in separate parity batches, so supports need only be
disjoint within each batch.

This module deliberately resolves the support/fold geometry before any
polynomial generators are synthesized.  In this layout that first stage
produces an exact obstruction: two-batch disjointness limits a seed to two
points per sheet, while the strict contragredient GL fold has no fixed sheet.
For every possible weight-seven or weight-eight seed, the translation-
circulant ZX pairing consequently has even augmentation and is singular.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .half_grid import StructuredGLFold, gf2_rank, iter_structured_gl_folds
from .s3_linear import BOTTOM_ORDER, BOTTOM_X_ORDER, BOTTOM_Y_ORDER

BATCHED_HALF_GRID_SCHEMA_VERSION = 1
NUM_PHYSICAL_BLOCKS = 2
NUM_SHEETS = 2 * NUM_PHYSICAL_BLOCKS
NUM_LOGICALS = BOTTOM_ORDER
NUM_QUBITS = NUM_SHEETS * BOTTOM_ORDER
MINIMUM_DISTANCE = 7
MAXIMUM_BATCHED_SEED_WEIGHT = 2 * NUM_SHEETS


def lattice_index(xx: int, yy: int) -> int:
    """Flatten one ``C8 x C4`` lattice coordinate."""
    return (xx % BOTTOM_X_ORDER) * BOTTOM_Y_ORDER + yy % BOTTOM_Y_ORDER


def lattice_coordinate(index: int) -> tuple[int, int]:
    """Invert :func:`lattice_index`."""
    return divmod(index, BOTTOM_Y_ORDER)


def site_parity(index: int) -> int:
    """Return the checkerboard color of one lattice coordinate."""
    xx, yy = lattice_coordinate(index)
    return (xx + yy) % 2


def physical_index(sheet: int, site: int) -> int:
    return sheet * BOTTOM_ORDER + site


def sheet_permutation(fold: StructuredGLFold) -> tuple[int, ...]:
    """Return the fold induced on ``(block, top-coordinate)`` sheets."""
    if fold.num_blocks != NUM_PHYSICAL_BLOCKS:
        raise ValueError("the n=128 geometry requires exactly two blocks")
    images = []
    for block in range(NUM_PHYSICAL_BLOCKS):
        images.append(2 * fold.forward_blocks[block] + 1)
        images.append(2 * fold.backward_blocks[block])
    return tuple(images)


def iter_n128_folds() -> Iterable[StructuredGLFold]:
    """Enumerate every translation-commuting strict GL sheet fold."""
    return iter_structured_gl_folds(NUM_PHYSICAL_BLOCKS)


def translate_seed(seed: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a sheet-by-lattice seed by ``(dx,dy)``."""
    return np.roll(
        np.roll(seed, dx % BOTTOM_X_ORDER, axis=1),
        dy % BOTTOM_Y_ORDER,
        axis=2,
    )


def translation_orbit(seed: np.ndarray) -> np.ndarray:
    """Return all 32 physical supports in ``C8``-major order."""
    rows = [
        translate_seed(seed, xx, yy).reshape(NUM_QUBITS)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    return np.asarray(rows, dtype=np.uint8)


def fold_seed(seed: np.ndarray, fold: StructuredGLFold) -> np.ndarray:
    """Apply a strict translation-commuting ZX fold to one support."""
    output = np.zeros_like(seed)
    for old_sheet, new_sheet in enumerate(sheet_permutation(fold)):
        output[new_sheet] = seed[old_sheet]
    return output


def pairing_matrix(seed: np.ndarray, fold: StructuredGLFold) -> np.ndarray:
    """Compute ``<Z_g, P Z_h>`` for the full translated orbit."""
    zz = translation_orbit(seed)
    xx = translation_orbit(fold_seed(seed, fold))
    return (zz @ xx.T) % 2


def batch_disjointness(seed: np.ndarray) -> dict[str, Any]:
    """Check disjointness separately on the two checkerboard batches."""
    orbit = translation_orbit(seed)
    sites = [
        (xx, yy)
        for xx in range(BOTTOM_X_ORDER)
        for yy in range(BOTTOM_Y_ORDER)
    ]
    batch_rows = {
        parity: [
            index for index, (xx, yy) in enumerate(sites)
            if (xx + yy) % 2 == parity
        ]
        for parity in (0, 1)
    }
    maxima = {
        str(parity): int(np.sum(orbit[indices], axis=0).max(initial=0))
        for parity, indices in batch_rows.items()
    }
    return {
        "batch_sizes": {str(key): len(value) for key, value in batch_rows.items()},
        "maximum_multiplicity": maxima,
        "disjoint_within_each_batch": all(value <= 1 for value in maxima.values()),
        "disjoint_all_at_once": bool(np.sum(orbit, axis=0).max(initial=0) <= 1),
    }


def sheet_occupancies(seed: np.ndarray) -> tuple[int, ...]:
    return tuple(int(np.count_nonzero(seed[sheet])) for sheet in range(NUM_SHEETS))


def pairing_augmentation_from_occupancies(
    occupancies: Sequence[int], fold: StructuredGLFold
) -> int:
    """Parity of every row sum in the translation-circulant pairing."""
    images = sheet_permutation(fold)
    return sum(
        int(occupancies[sheet]) * int(occupancies[images[sheet]])
        for sheet in range(NUM_SHEETS)
    ) % 2


def analyze_seed(seed: np.ndarray, fold: StructuredGLFold) -> dict[str, Any]:
    """Evaluate one batched support and its strict ZX pairing."""
    seed = np.asarray(seed, dtype=np.uint8)
    if seed.shape != (NUM_SHEETS, BOTTOM_X_ORDER, BOTTOM_Y_ORDER):
        raise ValueError("unexpected seed shape")
    occupancies = sheet_occupancies(seed)
    pairing = pairing_matrix(seed, fold)
    row_weights = np.count_nonzero(pairing, axis=1)
    column_weights = np.count_nonzero(pairing, axis=0)
    all_ones = np.ones(NUM_LOGICALS, dtype=np.uint8)
    return {
        "weight": int(np.count_nonzero(seed)),
        "sheet_occupancies": list(occupancies),
        "batching": batch_disjointness(seed),
        "fold": fold.to_dict(),
        "sheet_permutation": list(sheet_permutation(fold)),
        "sheet_fixed_points": sum(
            image == sheet for sheet, image in enumerate(sheet_permutation(fold))
        ),
        "pairing_augmentation_formula": pairing_augmentation_from_occupancies(
            occupancies, fold
        ),
        "pairing_row_parities": sorted(set((row_weights % 2).astype(int).tolist())),
        "all_ones_in_pairing_kernel": bool(not np.any((pairing @ all_ones) % 2)),
        "zx_pairing_rank": gf2_rank(pairing),
        "pairing_is_permutation": bool(
            np.all(row_weights == 1) and np.all(column_weights == 1)
        ),
    }


def random_batched_seed(rng: random.Random, *, weight: int) -> np.ndarray:
    """Sample a support disjoint within the two parity batches.

    Every sheet contains at most one even and one odd lattice point.  At
    weights seven and eight this is the most general possible occupancy
    pattern compatible with the prescribed parity batching.
    """
    if not 1 <= weight <= MAXIMUM_BATCHED_SEED_WEIGHT:
        raise ValueError("batched seed weight must lie between one and eight")
    slots = list(itertools.product(range(NUM_SHEETS), range(2)))
    chosen = rng.sample(slots, weight)
    by_parity = {
        parity: [
            lattice_index(xx, yy)
            for xx in range(BOTTOM_X_ORDER)
            for yy in range(BOTTOM_Y_ORDER)
            if (xx + yy) % 2 == parity
        ]
        for parity in (0, 1)
    }
    seed = np.zeros(
        (NUM_SHEETS, BOTTOM_X_ORDER, BOTTOM_Y_ORDER), dtype=np.uint8
    )
    for sheet, parity in chosen:
        site = rng.choice(by_parity[parity])
        xx, yy = lattice_coordinate(site)
        seed[sheet, xx, yy] = 1
    return seed


def exact_geometry_certificate() -> dict[str, Any]:
    """Exhaust every allowed occupancy pattern and certify the no-go.

    Two-batch disjointness implies occupancy at most two on each of the four
    sheets.  Distance at least seven therefore leaves only weights seven and
    eight.  Coordinate choices cannot alter the augmentation calculation, so
    exhausting the small occupancy space is exact for the obstruction.
    """
    folds = tuple(iter_n128_folds())
    cases = []
    all_zero = True
    for weight in range(MINIMUM_DISTANCE, MAXIMUM_BATCHED_SEED_WEIGHT + 1):
        occupancies = [
            values
            for values in itertools.product(range(3), repeat=NUM_SHEETS)
            if sum(values) == weight
        ]
        for fold_index, fold in enumerate(folds):
            augmentations = Counter(
                pairing_augmentation_from_occupancies(values, fold)
                for values in occupancies
            )
            all_zero &= set(augmentations) <= {0}
            cases.append(
                {
                    "weight": weight,
                    "fold_index": fold_index,
                    "fold": fold.to_dict(),
                    "sheet_permutation": list(sheet_permutation(fold)),
                    "fixed_points": sum(
                        image == sheet
                        for sheet, image in enumerate(sheet_permutation(fold))
                    ),
                    "occupancy_patterns": len(occupancies),
                    "augmentation_counts": {
                        str(key): value for key, value in sorted(augmentations.items())
                    },
                }
            )
    return {
        "schema_version": BATCHED_HALF_GRID_SCHEMA_VERSION,
        "lift": "GL(2,2) x C8 x C4",
        "n": NUM_QUBITS,
        "target_logicals": NUM_LOGICALS,
        "minimum_distance": MINIMUM_DISTANCE,
        "batches": 2,
        "batch_size": NUM_LOGICALS // 2,
        "maximum_seed_weight_under_batching": MAXIMUM_BATCHED_SEED_WEIGHT,
        "num_strict_structured_folds": len(folds),
        "all_folds_fixed_point_free": all(
            all(image != sheet for sheet, image in enumerate(sheet_permutation(fold)))
            for fold in folds
        ),
        "all_allowed_augmentations_zero": all_zero,
        "pairing_singular_for_every_allowed_seed": all_zero,
        "generator_search_pruned": all_zero,
        "reason": (
            "Every allowed translation-circulant ZX pairing has even row parity, "
            "so the all-ones logical vector is in its kernel."
        ),
        "cases": cases,
    }


def sampled_geometry_screen(*, seed: int, samples_per_fold: int) -> dict[str, Any]:
    """Sample concrete supports as an independent computational check."""
    if samples_per_fold < 1:
        raise ValueError("sample count must be positive")
    rng = random.Random(seed)
    ranks: Counter[int] = Counter()
    maximum_rank = 0
    violations = []
    tested = 0
    started = time.perf_counter()
    for weight in range(MINIMUM_DISTANCE, MAXIMUM_BATCHED_SEED_WEIGHT + 1):
        for fold_index, fold in enumerate(iter_n128_folds()):
            for trial in range(samples_per_fold):
                candidate = random_batched_seed(rng, weight=weight)
                analysis = analyze_seed(candidate, fold)
                tested += 1
                rank = int(analysis["zx_pairing_rank"])
                ranks[rank] += 1
                maximum_rank = max(maximum_rank, rank)
                if not (
                    analysis["batching"]["disjoint_within_each_batch"]
                    and analysis["all_ones_in_pairing_kernel"]
                    and rank < NUM_LOGICALS
                ):
                    violations.append(
                        {"weight": weight, "fold_index": fold_index, "trial": trial, **analysis}
                    )
    return {
        "seed": seed,
        "samples_per_fold_per_weight": samples_per_fold,
        "tested": tested,
        "rank_counts": {str(key): value for key, value in sorted(ranks.items())},
        "maximum_sampled_rank": maximum_rank,
        "full_rank_hits": ranks[NUM_LOGICALS],
        "violations": violations,
        "seconds": round(time.perf_counter() - started, 6),
    }
