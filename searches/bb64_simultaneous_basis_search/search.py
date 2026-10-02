#!/usr/bin/env python3
"""Exact affine simultaneous-basis search for the self-dual [[64,8,8]] BB code.

The same ordered logical basis must satisfy all of:
  * a regular permutation C4 x C2 action;
  * a permutation ZX Gram matrix under transversal physical H;
  * an exact-cover partition into internally disjoint Z batches, with every
    batch containing at least two logicals.

All affine physical code automorphisms and all 255 nonzero logical seed classes
are screened exactly.  Stabilizer dressing is attempted only after the first
two class-level conditions pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


Element = tuple[int, int]
ELEMENTS: tuple[Element, ...] = tuple((xx, yy) for xx in range(4) for yy in range(8))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
IDENTITY: Element = (0, 0)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dressing-seconds", type=float, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def emit(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


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


def independent_rows(matrix: np.ndarray) -> np.ndarray:
    selected = []
    rank = 0
    for row in np.asarray(matrix, dtype=np.uint8):
        candidate = np.asarray([*selected, row], dtype=np.uint8)
        candidate_rank = gf2_rank(candidate)
        if candidate_rank > rank:
            selected.append(row)
            rank = candidate_rank
    return np.asarray(selected, dtype=np.uint8)


def add(left: Element, right: Element) -> Element:
    raw_x = left[0] + right[0]
    return raw_x % 4, (left[1] + right[1] + 4 * (raw_x // 4)) % 8


def inverse(element: Element) -> Element:
    return next(candidate for candidate in ELEMENTS if add(element, candidate) == IDENTITY)


def scale(multiplier: int, element: Element) -> Element:
    if multiplier < 0:
        return scale(-multiplier, inverse(element))
    output = IDENTITY
    for _ in range(multiplier):
        output = add(output, element)
    return output


def affine_permutation(
    image_x: Element, image_y: Element, translation: Element = IDENTITY
) -> np.ndarray:
    return np.asarray(
        [
            INDEX[add(translation, add(scale(xx, image_x), scale(yy, image_y)))]
            for xx, yy in ELEMENTS
        ],
        dtype=np.int64,
    )


def linear_automorphisms() -> Iterable[tuple[Element, Element, np.ndarray]]:
    for image_x, image_y in itertools.product(ELEMENTS, repeat=2):
        permutation = affine_permutation(image_x, image_y)
        if len(set(map(int, permutation))) == 32:
            yield image_x, image_y, permutation


def translation_equivalent(source: set[Element], target: set[Element]) -> bool:
    return any(
        {add(translation, element) for element in source} == target
        for translation in ELEMENTS
    )


def full_permutation(bottom: np.ndarray, *, swap_halves: bool) -> np.ndarray:
    if swap_halves:
        return np.hstack([32 + bottom, bottom]).astype(np.int64)
    return np.hstack([bottom, 32 + bottom]).astype(np.int64)


def permute_rows(vectors: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vectors)
    output[:, permutation] = vectors
    return output


def same_row_space(left: np.ndarray, right: np.ndarray) -> bool:
    rank = gf2_rank(left)
    return rank == gf2_rank(right) == gf2_rank(np.vstack([left, right]))


def matrix_power(matrix: np.ndarray, exponent: int) -> np.ndarray:
    output = np.eye(len(matrix), dtype=np.uint8)
    base = np.asarray(matrix, dtype=np.uint8)
    while exponent:
        if exponent & 1:
            output = (output @ base) % 2
        base = (base @ base) % 2
        exponent //= 2
    return output


def matrix_order(matrix: np.ndarray, limit: int = 64) -> int | None:
    identity = np.eye(len(matrix), dtype=np.uint8)
    power = identity.copy()
    for exponent in range(1, limit + 1):
        power = (power @ matrix) % 2
        if np.array_equal(power, identity):
            return exponent
    return None


def permutation_order(permutation: np.ndarray) -> int:
    identity = np.arange(len(permutation), dtype=np.int64)
    power = identity.copy()
    for exponent in range(1, len(permutation) + 1):
        power = permutation[power]
        if np.array_equal(power, identity):
            return exponent
    raise ValueError("invalid permutation")


def permutation_matrix_test(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=np.uint8)
    return bool(np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1))


def group_actions(action_x: np.ndarray, action_y: np.ndarray) -> list[np.ndarray]:
    return [
        (matrix_power(action_x, xx) @ matrix_power(action_y, yy)) % 2
        for xx in range(4)
        for yy in range(2)
    ]


def physical_symmetries(
    check: np.ndarray, logical_z: np.ndarray, logical_x: np.ndarray
) -> list[dict[str, Any]]:
    base_support = {IDENTITY, (1, 0), (0, 1), inverse((0, 1))}
    inverse_support = {inverse(element) for element in base_support}
    output: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    linear_tested = 0
    for image_x, image_y, linear in linear_automorphisms():
        linear_tested += 1
        image_support = {IDENTITY, image_x, image_y, inverse(image_y)}
        allowed_swaps = []
        if translation_equivalent(image_support, base_support):
            allowed_swaps.append(False)
        if translation_equivalent(image_support, inverse_support):
            allowed_swaps.append(True)
        for translation in ELEMENTS:
            bottom = np.asarray(
                [INDEX[add(translation, ELEMENTS[int(target)])] for target in linear],
                dtype=np.int64,
            )
            for swap_halves in allowed_swaps:
                permutation = full_permutation(bottom, swap_halves=swap_halves)
                key = permutation.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                translated = permute_rows(check, permutation)
                if not same_row_space(check, translated):
                    continue
                translated_logical_z = permute_rows(logical_z, permutation)
                action = (translated_logical_z @ logical_x.T) % 2
                output.append(
                    {
                        "image_x": image_x,
                        "image_y": image_y,
                        "translation": translation,
                        "swap_halves": swap_halves,
                        "permutation": permutation,
                        "physical_order": permutation_order(permutation),
                        "action": action,
                        "logical_order": matrix_order(action),
                    }
                )
        if linear_tested % 16 == 0:
            emit(
                "affine_symmetries",
                linear_automorphisms_tested=linear_tested,
                code_symmetries_found=len(output),
            )
    return output


def batch_disjointness(logicals: np.ndarray, partition: Sequence[Sequence[int]]) -> bool:
    flattened = [index for batch in partition for index in batch]
    return bool(
        sorted(flattened) == list(range(8))
        and len(flattened) == len(set(flattened))
        and all(len(batch) >= 2 for batch in partition)
        and all(np.all(np.sum(logicals[list(batch)], axis=0) <= 1) for batch in partition)
    )


def find_raw_partition(logicals: np.ndarray) -> list[list[int]] | None:
    """Find the fewest internally disjoint batches, forbidding singletons."""
    compatible = np.zeros((8, 8), dtype=bool)
    for left in range(8):
        for right in range(left + 1, 8):
            compatible[left, right] = compatible[right, left] = not bool(
                np.any(logicals[left] & logicals[right])
            )

    def valid_block(block: Sequence[int]) -> bool:
        return all(compatible[left, right] for left, right in itertools.combinations(block, 2))

    def recurse(remaining: tuple[int, ...], target_batches: int) -> list[list[int]] | None:
        if not remaining:
            return [] if target_batches == 0 else None
        if target_batches <= 0 or len(remaining) < 2 * target_batches:
            return None
        anchor = remaining[0]
        others = remaining[1:]
        maximum_size = len(remaining) - 2 * (target_batches - 1)
        for size in range(maximum_size, 1, -1):
            for tail in itertools.combinations(others, size - 1):
                block = (anchor, *tail)
                if not valid_block(block):
                    continue
                block_set = set(block)
                rest = tuple(index for index in remaining if index not in block_set)
                suffix = recurse(rest, target_batches - 1)
                if suffix is not None:
                    return [list(block), *suffix]
        return None

    for num_batches in range(1, 5):
        partition = recurse(tuple(range(8)), num_batches)
        if partition is not None:
            assert batch_disjointness(logicals, partition)
            return partition
    return None


def dress_for_batch_count(
    base: np.ndarray,
    stabilizers: np.ndarray,
    num_batches: int,
    *,
    time_limit: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Find arbitrary stabilizer dressings and an exact-cover batch assignment.

    Z3 represents the GF(2) stabilizer additions as native XOR constraints.  This
    is a feasibility problem, not a support-weight optimization; a fixed-partition
    weight pass can be performed after the simultaneous basis has been certified.
    """
    import z3

    coefficients = [
        [z3.Bool(f"c_{logical}_{stab}") for stab in range(len(stabilizers))]
        for logical in range(8)
    ]
    representatives = [
        [z3.Bool(f"r_{logical}_{qubit}") for qubit in range(64)]
        for logical in range(8)
    ]
    assignment = [
        [z3.Bool(f"a_{logical}_{batch}") for batch in range(num_batches)]
        for logical in range(8)
    ]
    solver = z3.Solver()
    solver.set(timeout=max(1, round(1000 * time_limit)))
    for logical in range(8):
        solver.add(
            z3.PbEq([(assignment[logical][batch], 1) for batch in range(num_batches)], 1)
        )
        for qubit in range(64):
            parity = z3.BoolVal(bool(base[logical, qubit]))
            for stab in np.flatnonzero(stabilizers[:, qubit]):
                parity = z3.Xor(parity, coefficients[logical][int(stab)])
            solver.add(representatives[logical][qubit] == parity)
    # Remove only the arbitrary permutation of batch labels.
    solver.add(assignment[0][0])
    for batch in range(num_batches):
        solver.add(
            z3.PbGe([(assignment[logical][batch], 1) for logical in range(8)], 2)
        )
    for left in range(8):
        for right in range(left + 1, 8):
            for batch in range(num_batches):
                for qubit in range(64):
                    solver.add(
                        z3.Or(
                            z3.Not(assignment[left][batch]),
                            z3.Not(assignment[right][batch]),
                            z3.Not(representatives[left][qubit]),
                            z3.Not(representatives[right][qubit]),
                        )
                    )
    started = time.perf_counter()
    status = solver.check()
    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "num_batches": num_batches,
        "status": str(status),
        "seconds": round(elapsed, 6),
        "time_limit": time_limit,
        "solver": "Z3 native XOR feasibility",
    }
    if status == z3.unsat:
        return {**record, "feasible": False}, None
    if status != z3.sat:
        return {**record, "feasible": None, "reason_unknown": solver.reason_unknown()}, None
    model = solver.model()
    coefficient_value = np.asarray(
        [
            [int(z3.is_true(model.eval(value, model_completion=True))) for value in row]
            for row in coefficients
        ],
        dtype=np.uint8,
    )
    dressed = base ^ ((coefficient_value @ stabilizers) % 2)
    assignment_value = np.asarray(
        [
            [int(z3.is_true(model.eval(value, model_completion=True))) for value in row]
            for row in assignment
        ],
        dtype=np.uint8,
    )
    partition = [
        np.flatnonzero(assignment_value[:, batch]).astype(int).tolist()
        for batch in range(num_batches)
    ]
    assert batch_disjointness(dressed, partition)
    return (
        {
            **record,
            "feasible": True,
            "partition": partition,
            "batch_sizes": [len(batch) for batch in partition],
            "objective": None,
            "weights": np.count_nonzero(dressed, axis=1).astype(int).tolist(),
            "supports": [np.flatnonzero(row).astype(int).tolist() for row in dressed],
            "stabilizer_dressing_coefficients": coefficient_value.astype(int).tolist(),
        },
        dressed,
    )


def symmetry_metadata(symmetry: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_x": list(symmetry["image_x"]),
        "image_y": list(symmetry["image_y"]),
        "translation": list(symmetry["translation"]),
        "swap_halves": symmetry["swap_halves"],
        "physical_order": symmetry["physical_order"],
        "logical_order": symmetry["logical_order"],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    saved = np.load(args.input_npz)
    check = np.asarray(saved["matrix_z"], dtype=np.uint8)
    logical_z = np.asarray(saved["raw_grid_z"], dtype=np.uint8)
    logical_x = np.asarray(saved["raw_dual_x"], dtype=np.uint8)
    stabilizers = independent_rows(check)
    base_pairing = (logical_z @ logical_z.T) % 2
    assert np.array_equal((logical_z @ logical_x.T) % 2, np.eye(8, dtype=np.uint8))

    symmetries = physical_symmetries(check, logical_z, logical_x)
    order_four = [item for item in symmetries if item["logical_order"] == 4]
    order_two = [item for item in symmetries if item["logical_order"] == 2]
    emit(
        "symmetries_complete",
        affine_code_symmetries=len(symmetries),
        logical_order_four=len(order_four),
        logical_order_two=len(order_two),
    )

    action_pairs = []
    for symmetry_x in order_four:
        for symmetry_y in order_two:
            permutation_x = symmetry_x["permutation"]
            permutation_y = symmetry_y["permutation"]
            if not np.array_equal(permutation_x[permutation_y], permutation_y[permutation_x]):
                continue
            action_x = symmetry_x["action"]
            action_y = symmetry_y["action"]
            if not np.array_equal((action_x @ action_y) % 2, (action_y @ action_x) % 2):
                continue
            actions = group_actions(action_x, action_y)
            if gf2_rank(np.asarray(actions).reshape(8, -1)) != 8:
                continue
            action_pairs.append((symmetry_x, symmetry_y, actions))
    emit("action_pairs_complete", regular_action_pairs=len(action_pairs))

    counters = {
        "action_pairs": len(action_pairs),
        "seed_classes_considered": 0,
        "regular_seed_orbits": 0,
        "permutation_pairing_orbits": 0,
        "raw_disjoint_orbits": 0,
        "deduplicated_class_candidates": 0,
    }
    candidates: list[dict[str, Any]] = []
    candidate_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for pair_index, (symmetry_x, symmetry_y, actions) in enumerate(action_pairs):
        for mask in range(1, 256):
            counters["seed_classes_considered"] += 1
            seed = np.asarray([(mask >> bit) & 1 for bit in range(8)], dtype=np.uint8)
            classes = np.asarray([(seed @ action) % 2 for action in actions], dtype=np.uint8)
            if gf2_rank(classes) != 8:
                continue
            counters["regular_seed_orbits"] += 1
            pairing = (classes @ base_pairing @ classes.T) % 2
            if not permutation_matrix_test(pairing):
                continue
            counters["permutation_pairing_orbits"] += 1
            # Batch feasibility depends only on the unordered set of eight
            # logical classes.  Retain the first regular physical action that
            # realizes each such set.
            class_key = b"".join(
                sorted(
                    np.packbits(row, bitorder="little").tobytes()
                    for row in classes
                )
            )
            if class_key in seen:
                continue
            seen.add(class_key)
            base = (classes @ logical_z) % 2
            raw_partition = find_raw_partition(base)
            if raw_partition is not None:
                counters["raw_disjoint_orbits"] += 1
            record = {
                "pair_index": pair_index,
                "seed_mask": mask,
                "seed_class": seed.astype(int).tolist(),
                "x_symmetry": symmetry_metadata(symmetry_x),
                "y_symmetry": symmetry_metadata(symmetry_y),
                "pairing": pairing.astype(int).tolist(),
                "hadamard_permutation": np.argmax(pairing, axis=1).astype(int).tolist(),
                "base_weights": np.count_nonzero(base, axis=1).astype(int).tolist(),
                "base_supports": [np.flatnonzero(row).astype(int).tolist() for row in base],
                "raw_disjoint_partition": raw_partition,
            }
            candidates.append(record)
            candidate_arrays.append(
                (
                    base,
                    pairing,
                    symmetry_x["permutation"],
                    symmetry_y["permutation"],
                    classes,
                )
            )
        if (pair_index + 1) % args.checkpoint_every == 0:
            counters["deduplicated_class_candidates"] = len(candidates)
            emit("class_scan", pair_index=pair_index + 1, **counters)
            atomic_json(
                args.output_dir / "checkpoint.json",
                {"pair_index": pair_index + 1, "counters": counters},
            )

    counters["deduplicated_class_candidates"] = len(candidates)
    emit("class_scan_complete", **counters)
    atomic_json(args.output_dir / "class_candidates.json", candidates)

    success: dict[str, Any] | None = None
    solver_records = []
    for candidate_index, (record, arrays) in enumerate(zip(candidates, candidate_arrays)):
        base, pairing, permutation_x, permutation_y, classes = arrays
        if record["raw_disjoint_partition"] is not None:
            partition = record["raw_disjoint_partition"]
            dressed = base
            solver_record = {
                "num_batches": len(partition),
                "partition": partition,
                "batch_sizes": [len(batch) for batch in partition],
                "feasible": True,
                "already_disjoint_without_dressing": True,
                "weights": np.count_nonzero(base, axis=1).astype(int).tolist(),
            }
        else:
            dressed = None
            solver_record = None
            for num_batches in range(1, 5):
                result, proposed = dress_for_batch_count(
                    base,
                    stabilizers,
                    num_batches,
                    time_limit=args.dressing_seconds,
                )
                solver_records.append({"candidate_index": candidate_index, **result})
                emit(
                    "dressing",
                    candidate_index=candidate_index,
                    candidates=len(candidates),
                    num_batches=num_batches,
                    status=result["status"],
                    feasible=result["feasible"],
                    seconds=result["seconds"],
                )
                if proposed is not None:
                    dressed = proposed
                    solver_record = result
                    break
        if dressed is None or solver_record is None:
            continue
        # Dressing cannot change logical pairings, but verify physically.
        assert np.array_equal((dressed @ dressed.T) % 2, pairing)
        partition = solver_record["partition"]
        assert batch_disjointness(dressed, partition)
        logical_x_basis = dressed[np.argmax(pairing, axis=1)]
        assert np.array_equal((dressed @ logical_x_basis.T) % 2, np.eye(8, dtype=np.uint8))
        success = {
            **record,
            "candidate_index": candidate_index,
            "dressing": solver_record,
            "final_weights": np.count_nonzero(dressed, axis=1).astype(int).tolist(),
            "final_supports": [np.flatnonzero(row).astype(int).tolist() for row in dressed],
        }
        batch_of_logical = np.zeros(8, dtype=np.int64)
        for batch_index, batch in enumerate(partition):
            batch_of_logical[batch] = batch_index
        np.savez_compressed(
            args.output_dir / "simultaneous_basis.npz",
            matrix_x=check,
            matrix_z=check,
            logical_z=dressed,
            logical_x=logical_x_basis,
            zx_pairing=pairing,
            hadamard_permutation=np.argmax(pairing, axis=1),
            grid_x_physical_permutation=permutation_x,
            grid_y_physical_permutation=permutation_y,
            grid_orbit_classes=classes,
            disjoint_batch_of_logical=batch_of_logical,
            disjoint_batch_sizes=np.asarray([len(batch) for batch in partition], dtype=np.int64),
        )
        atomic_json(args.output_dir / "success.json", success)
        emit(
            "success",
            candidate_index=candidate_index,
            num_batches=len(partition),
            batch_sizes=[len(batch) for batch in partition],
        )
        break

    atomic_json(args.output_dir / "dressing_results.json", solver_records)
    summary = {
        "source_code": "Liang-Chen self-dual [[64,8,8]] BB code",
        "searched_physical_family": "all affine automorphisms of the twisted 32-element torus preserving the code",
        "search_exactness": (
            "affine symmetries, action pairs, and 255 logical seed classes exact; "
            "dressing exact when Z3 returns sat or unsat, and unresolved on timeout"
        ),
        "affine_code_symmetries": len(symmetries),
        "logical_order_four_symmetries": len(order_four),
        "logical_order_two_symmetries": len(order_two),
        "counters": counters,
        "dressing_attempts": len(solver_records),
        "dressing_status_histogram": {
            status: sum(item["status"] == status for item in solver_records)
            for status in sorted({item["status"] for item in solver_records})
        },
        "simultaneous_basis_found": success is not None,
        "success": success,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    emit("complete", **{key: value for key, value in summary.items() if key != "success"})


if __name__ == "__main__":
    main()
