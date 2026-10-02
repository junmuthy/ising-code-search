#!/usr/bin/env python3
"""Enumerate exact Tanner automorphisms of the saved 16-row X presentation."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np


NUM_QUBITS = 32
NUM_CHECKS = 16
LOGICAL_ORDER = 4
THICKNESS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def compose(left: list[int], right: list[int]) -> list[int]:
    """Return left after right."""
    return [left[right[index]] for index in range(len(right))]


def power(permutation: list[int], exponent: int) -> list[int]:
    output = list(range(len(permutation)))
    for _ in range(exponent):
        output = compose(permutation, output)
    return output


def permutation_order(permutation: list[int]) -> int:
    unseen = set(range(len(permutation)))
    order = 1
    while unseen:
        start = min(unseen)
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            length += 1
            current = permutation[current]
        order = int(np.lcm(order, length))
    return order


def cycle_decomposition(permutation: list[int]) -> list[list[int]]:
    unseen = set(range(len(permutation)))
    cycles = []
    while unseen:
        start = min(unseen)
        cycle = []
        current = start
        while current in unseen:
            unseen.remove(current)
            cycle.append(current)
            current = permutation[current]
        cycles.append(cycle)
    return cycles


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    payload = json.loads(args.presentation.read_text())
    supports = payload["checks_x"]
    graph = nx.Graph()
    for check in range(NUM_CHECKS):
        graph.add_node(("c", check), kind="check")
    for data in range(NUM_QUBITS):
        graph.add_node(("q", data), kind="data")
    for check, support in enumerate(supports):
        for data in support:
            graph.add_edge(("c", check), ("q", int(data)))

    translation_data = [
        ((data // THICKNESS + 1) % LOGICAL_ORDER) * THICKNESS + data % THICKNESS
        for data in range(NUM_QUBITS)
    ]
    translation_checks = payload["translation_action_on_x_checks"]
    check_orbit_index = {
        check: orbit
        for orbit, members in enumerate(payload["translation_check_orbits"])
        for check in members
    }
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=lambda left, right: left["kind"] == right["kind"],
    )
    counters: Counter[str] = Counter()
    useful = []
    started = time.perf_counter()
    for mapping in matcher.isomorphisms_iter():
        counters["tanner_automorphisms"] += 1
        data = [int(mapping[("q", index)][1]) for index in range(NUM_QUBITS)]
        checks = [int(mapping[("c", index)][1]) for index in range(NUM_CHECKS)]
        orbit_images = []
        permutes_orbits = True
        for members in payload["translation_check_orbits"]:
            images = {check_orbit_index[checks[index]] for index in members}
            if len(images) != 1:
                permutes_orbits = False
                break
            orbit_images.append(images.pop())
        if not permutes_orbits:
            continue
        counters["permutes_four_check_orbits"] += 1
        orbit_cycles = cycle_decomposition(orbit_images)
        if sorted(map(len, orbit_cycles)) != [4]:
            continue
        counters["cycles_four_check_orbits"] += 1

        conjugation = compose(compose(data, translation_data), power(data, permutation_order(data) - 1))
        normalizer_exponent = None
        for exponent in (0, 1, 2, 3):
            if conjugation == power(translation_data, exponent):
                normalizer_exponent = exponent
                break
        if normalizer_exponent is None:
            continue
        counters["normalizes_physical_C4"] += 1
        commutes = normalizer_exponent == 1
        if commutes:
            counters["commutes_with_physical_C4"] += 1
        check_order = permutation_order(checks)
        data_order = permutation_order(data)
        generated_check_orbit = set()
        for first_power in range(data_order):
            for second_power in range(LOGICAL_ORDER):
                generated_check_orbit.add(power(translation_checks, second_power)[power(checks, first_power)[0]])
        regular_on_checks = len(generated_check_orbit) == NUM_CHECKS
        if regular_on_checks:
            counters["with_C4_transitive_on_16_checks"] += 1
        q_fourth = power(data, 4)
        fourth_power_translation = next(
            (
                exponent
                for exponent in range(LOGICAL_ORDER)
                if q_fourth == power(translation_data, exponent)
            ),
            None,
        )
        record = {
            "data_permutation": data,
            "check_permutation": checks,
            "orbit_permutation": orbit_images,
            "data_cycles": cycle_decomposition(data),
            "check_cycles": cycle_decomposition(checks),
            "data_order": data_order,
            "check_order": check_order,
            "normalizes_T_to_power": normalizer_exponent,
            "commutes_with_T": commutes,
            "with_T_transitive_on_16_checks": regular_on_checks,
            "Q_fourth_equals_T_power": fourth_power_translation,
        }
        useful.append(record)
        if data_order == 16 and check_order == 16:
            counters["order_16_on_data_and_checks"] += 1
        if args.checkpoint_every and counters["tanner_automorphisms"] % args.checkpoint_every == 0:
            print(
                f"checkpoint automorphisms={counters['tanner_automorphisms']} "
                f"orbit-cyclers={counters['cycles_four_check_orbits']} "
                f"useful={len(useful)} seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )

    useful.sort(
        key=lambda item: (
            not (item["data_order"] == 16 and item["check_order"] == 16),
            not item["commutes_with_T"],
            item["data_order"],
            item["data_permutation"],
        )
    )
    output = {
        "schema_version": 1,
        "search_scope": "all exact color-preserving Tanner-graph automorphisms of the saved 16-row X presentation",
        "counters": dict(sorted(counters.items())),
        "seconds": round(time.perf_counter() - started, 6),
        "found_second_translation": bool(useful),
        "found_order_16_translation": any(
            item["data_order"] == item["check_order"] == 16 for item in useful
        ),
        "candidates": useful,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "candidates"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
