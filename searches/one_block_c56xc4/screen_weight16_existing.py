#!/usr/bin/env python3
"""Screen the certified C56 x C4 ideals for check presentations of weight <= 16.

The weight-12 run enumerated normalized quotient polynomials with one, two, or
three nonzero GF(8) coefficients.  Each coefficient inflates to the binary
weight-four C7 simplex seed, so adding normalized four-term polynomials raises
the physical check-weight ceiling from 12 to 16.

This extension deliberately reuses the 731 ideals and their exact distances
from the completed weight-12 run.  Four-term polynomials are used as candidate
generators for both reciprocal sectors; they are not introduced as additional
code ideals in this focused screen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


SOURCE_DIRECTORY = pathlib.Path(__file__).resolve().parent
if not (SOURCE_DIRECTORY / "algebra.py").exists():
    default_source = pathlib.Path(
        "/home/judah_unmuth/gala-code-search/searches/one_block_c56xc4"
    )
    sys.path.insert(0, str(default_source))
else:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from algebra import (  # noqa: E402
    GF8_INV,
    GF8_MUL,
    GF8_SIZE,
    QUOTIENT_ORDER,
    SparsePolynomial,
    canonical_sparse_polynomial,
    gf8_rref,
    physical_seed_from_sparse,
    quotient_ideal_basis,
    tanner_connected,
)
from search import (  # noqa: E402
    CatalogEntry,
    build_sparse_catalog,
    enumerate_ideal_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def iter_normalized_tetranomials(
    *, checkpoint_every: int = 10_000
) -> Iterator[SparsePolynomial]:
    """Yield translation/scalar-normalized zero-augmentation tetranomials."""
    seen: set[tuple[int, ...]] = set()
    unit = np.zeros(QUOTIENT_ORDER, dtype=np.uint8)
    unit[0] = 1
    raw = 0
    emitted = 0
    for positions in itertools.combinations(range(1, QUOTIENT_ORDER), 3):
        for coefficient_one in range(1, GF8_SIZE):
            for coefficient_two in range(1, GF8_SIZE):
                coefficient_three = 1 ^ coefficient_one ^ coefficient_two
                if coefficient_three == 0:
                    continue
                raw += 1
                polynomial = unit.copy()
                polynomial[positions[0]] = coefficient_one
                polynomial[positions[1]] = coefficient_two
                polynomial[positions[2]] = coefficient_three
                key = canonical_sparse_polynomial(polynomial)
                if key in seen:
                    continue
                seen.add(key)
                emitted += 1
                yield SparsePolynomial("tetranomial", key)
                if emitted % checkpoint_every == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "building_weight_16_catalog",
                                "normalized_tetranomials": emitted,
                                "raw_examined": raw,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


def contains_many(
    target_rref: np.ndarray[Any, Any], vectors: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Return a mask for vectors in the GF(8) row span of an RREF basis."""
    if not len(target_rref):
        return ~np.any(vectors, axis=1)
    work = vectors.copy()
    for row in target_rref:
        nonzero = np.flatnonzero(row)
        if not len(nonzero):
            continue
        pivot = int(nonzero[0])
        factors = work[:, pivot].copy()
        work ^= GF8_MUL[factors[:, None], row[None, :]]
    return ~np.any(work, axis=1)


def fast_rank(matrix: np.ndarray[Any, Any]) -> int:
    """Compute GF(8) rank with vectorized row elimination."""
    work = np.asarray(matrix, dtype=np.uint8).copy()
    if not len(work):
        return 0
    pivot_row = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        work[[pivot_row, selected]] = work[[selected, pivot_row]]
        inverse = int(GF8_INV[work[pivot_row, column]])
        work[pivot_row] = GF8_MUL[inverse, work[pivot_row]]
        factors = work[:, column].copy()
        factors[pivot_row] = 0
        work ^= GF8_MUL[factors[:, None], work[pivot_row][None, :]]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def select_orbit_generators(
    target: np.ndarray[Any, Any],
    entries: Sequence[CatalogEntry],
    ideal_cache: dict[str, np.ndarray[Any, Any]],
) -> tuple[list[CatalogEntry], int]:
    """Greedily select orbit generators after exact low-weight span screening."""
    generated = np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8)
    selected: list[CatalogEntry] = []
    for entry in entries:
        key = entry.polynomial.candidate_id
        ideal = ideal_cache.get(key)
        if ideal is None:
            ideal = (
                entry.ideal_basis
                if len(entry.ideal_basis)
                else quotient_ideal_basis([entry.polynomial.vector])
            )
            ideal_cache[key] = ideal
        expanded = gf8_rref(np.vstack([generated, ideal]))[0]
        if len(expanded) == len(generated):
            continue
        generated = expanded
        selected.append(entry)
        if len(generated) == len(target):
            break
    return selected, len(generated)


def screen_target(
    target: np.ndarray[Any, Any],
    catalog: Sequence[CatalogEntry],
    vectors: np.ndarray[Any, Any],
    ideal_cache: dict[str, np.ndarray[Any, Any]],
) -> tuple[bool, list[CatalogEntry], int]:
    target = gf8_rref(target)[0]
    if not len(target):
        return True, [], 0
    membership = contains_many(target, vectors)
    eligible_indices = np.flatnonzero(membership)
    eligible = [catalog[int(index)] for index in eligible_indices]
    selected, selected_rank = select_orbit_generators(target, eligible, ideal_cache)
    return selected_rank == len(target), selected, selected_rank


def generators_to_dict(
    generators: Sequence[CatalogEntry], *, dagger: bool
) -> list[dict[str, Any]]:
    return [
        {
            "polynomial_id": entry.polynomial.candidate_id,
            "support": [list(term) for term in entry.polynomial.support],
            "quotient_weight": entry.polynomial.support_size,
            "physical_check_weight": 4 * entry.polynomial.support_size,
            "dagger_sector": dagger,
        }
        for entry in generators
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    previous_records = {
        record["candidate_id"]: record
        for record in (
            json.loads(line) for line in (args.previous / "candidates.jsonl").open()
        )
    }
    old_catalog = build_sparse_catalog()
    candidates, raw_counts = enumerate_ideal_candidates(old_catalog)
    if set(previous_records) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("previous result IDs do not match the reconstructed ideals")

    catalog = list(old_catalog)
    for polynomial in iter_normalized_tetranomials():
        # The full ideal is computed only for selected presentation generators.
        catalog.append(
            CatalogEntry(
                polynomial=polynomial,
                ideal_basis=np.zeros((0, QUOTIENT_ORDER), dtype=np.uint8),
                ideal_key=(),
            )
        )
    catalog.sort(
        key=lambda entry: (
            entry.polynomial.support_size,
            entry.polynomial.candidate_id,
        )
    )
    vectors = np.asarray([entry.polynomial.vector for entry in catalog], dtype=np.uint8)
    tetranomial_count = sum(
        entry.polynomial.family == "tetranomial" for entry in catalog
    )
    catalog_summary = {
        "scope": "existing_731_distance_certified_ideals_only",
        "previous_result": str(args.previous),
        "raw_original_ideal_counts": raw_counts,
        "old_sparse_polynomials": len(old_catalog),
        "normalized_tetranomials": tetranomial_count,
        "total_generator_catalog": len(catalog),
        "physical_check_weight_ceiling": 16,
    }
    atomic_json(args.output / "catalog-summary.json", catalog_summary)
    print(json.dumps({"stage": "screen", **catalog_summary}, sort_keys=True), flush=True)

    output_jsonl = args.output / "candidates.jsonl"
    joint_count = 0
    connected_count = 0
    high_distance_joint = 0
    high_distance_connected = 0
    distance_seven_joint = 0
    distance_seven_connected = 0
    check_weight_histogram: Counter[int] = Counter()
    ideal_cache: dict[str, np.ndarray[Any, Any]] = {
        entry.polynomial.candidate_id: entry.ideal_basis for entry in old_catalog
    }
    for index, candidate in enumerate(candidates, start=1):
        previous = previous_records[candidate.candidate_id]
        ideal = gf8_rref(candidate.ideal_basis)[0]
        # Reuse the annihilator basis saved by the original algebraic construction.
        from algebra import quotient_annihilator_basis

        annihilator = quotient_annihilator_basis(list(ideal))
        ideal_ok, ideal_generators, ideal_rank = screen_target(
            ideal, catalog, vectors, ideal_cache
        )
        annihilator_ok, annihilator_generators, annihilator_rank = screen_target(
            annihilator, catalog, vectors, ideal_cache
        )
        joint = ideal_ok and annihilator_ok
        selected = [
            (entry, False) for entry in ideal_generators
        ] + [
            (entry, True) for entry in annihilator_generators
        ]
        check_weight = (
            max(
                (4 * entry.polynomial.support_size for entry, _ in selected),
                default=0,
            )
            if joint
            else None
        )
        connected = False
        if joint:
            physical_seeds = [
                physical_seed_from_sparse(entry.polynomial, dagger=dagger)
                for entry, dagger in selected
            ]
            connected = tanner_connected(physical_seeds)
            joint_count += 1
            connected_count += int(connected)
            check_weight_histogram[int(check_weight)] += 1
            if previous["distance"] >= 6:
                high_distance_joint += 1
                high_distance_connected += int(connected)
            if previous["distance"] == 7:
                distance_seven_joint += 1
                distance_seven_connected += int(connected)

        record = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "families": list(candidate.families),
            "n": previous["n"],
            "k": previous["k"],
            "distance": previous["distance"],
            "weight_16_generation": joint,
            "tanner_connected": connected,
            "structurally_accepted": joint and connected,
            "check_weight": check_weight,
            "ideal_generated_rank": ideal_rank,
            "ideal_target_dimension": len(ideal),
            "annihilator_generated_rank": annihilator_rank,
            "annihilator_target_dimension": len(annihilator),
            "ideal_generators": generators_to_dict(ideal_generators, dagger=False),
            "annihilator_generators": generators_to_dict(
                annihilator_generators, dagger=True
            ),
        }
        append_jsonl(output_jsonl, record)

        progress = {
            "completed": index,
            "total": len(candidates),
            "joint_weight_16_presentations": joint_count,
            "connected_weight_16_presentations": connected_count,
            "distance_at_least_six_joint": high_distance_joint,
            "distance_at_least_six_connected": high_distance_connected,
            "distance_seven_joint": distance_seven_joint,
            "distance_seven_connected": distance_seven_connected,
            "check_weight_histogram": dict(sorted(check_weight_histogram.items())),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if index % args.checkpoint_every == 0 or index == len(candidates):
            atomic_json(args.output / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    summary = {
        "complete": True,
        **catalog_summary,
        "existing_ideals_screened": len(candidates),
        "joint_weight_16_presentations": joint_count,
        "connected_weight_16_presentations": connected_count,
        "distance_at_least_six_joint": high_distance_joint,
        "distance_at_least_six_connected": high_distance_connected,
        "distance_seven_joint": distance_seven_joint,
        "distance_seven_connected": distance_seven_connected,
        "check_weight_histogram": dict(sorted(check_weight_histogram.items())),
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
