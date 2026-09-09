#!/usr/bin/env python3
"""Search fold-symmetric five-block LP codes over C4 x C8."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import defaultdict, deque

import numpy as np


MODS = (4, 8)
GROUP = [(a, b) for a in range(MODS[0]) for b in range(MODS[1])]
ORDER = len(GROUP)
N = 5 * ORDER


def idx(g: tuple[int, int]) -> int:
    return (g[0] % MODS[0]) * MODS[1] + g[1] % MODS[1]


def add(g: tuple[int, int], h: tuple[int, int]) -> tuple[int, int]:
    return ((g[0] + h[0]) % MODS[0], (g[1] + h[1]) % MODS[1])


def neg(g: tuple[int, int]) -> tuple[int, int]:
    return ((-g[0]) % MODS[0], (-g[1]) % MODS[1])


def scalar(n: int, g: tuple[int, int]) -> tuple[int, int]:
    return ((n * g[0]) % MODS[0], (n * g[1]) % MODS[1])


def element_order(g: tuple[int, int]) -> int:
    for n in (1, 2, 4, 8):
        if scalar(n, g) == (0, 0):
            return n
    raise AssertionError(g)


def automorphism(image_x: tuple[int, int], image_y: tuple[int, int]):
    def phi(g: tuple[int, int]) -> tuple[int, int]:
        return add(scalar(g[0], image_x), scalar(g[1], image_y))
    return phi


def involutive_automorphisms():
    output = []
    for image_x in GROUP:
        if element_order(image_x) > 4:
            continue
        for image_y in GROUP:
            phi = automorphism(image_x, image_y)
            images = {phi(g) for g in GROUP}
            if len(images) != ORDER:
                continue
            if any(phi(phi(g)) != g for g in GROUP):
                continue
            if all(phi(g) == g for g in GROUP):
                continue
            output.append((image_x, image_y, phi))
    return output


def group_matrix(support: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((ORDER, ORDER), dtype=np.uint8)
    for row, g in enumerate(GROUP):
        for term in support:
            matrix[row, idx(add(g, term))] ^= 1
    return matrix


def transformed_antipode(support, phi):
    return [phi(neg(g)) for g in support]


def build_checks(a, b, phi):
    c = transformed_antipode(a, phi)
    d = transformed_antipode(b, phi)
    A, B, C, D = map(group_matrix, (a, b, c, d))
    z = np.zeros_like(A)
    hx = np.block([[A, z, B, z, C.T], [z, A, z, B, D.T]])
    hz = np.block([[C, D, z, z, A.T], [z, z, C, D, B.T]])
    return hx, hz, c, d


def rank2(matrix: np.ndarray) -> int:
    return len(rowspace_basis(matrix))


def rowspace_basis(matrix: np.ndarray) -> dict[int, int]:
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
    return pivots


def in_span(value: int, pivots: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            return False
        value ^= pivots[pivot]
    return True


def column_syndromes(matrix: np.ndarray) -> list[int]:
    values = []
    for col in matrix.T:
        values.append(int.from_bytes(np.packbits(col, bitorder="little").tobytes(), "little"))
    return values


def low_logical(hcheck: np.ndarray, hstab: np.ndarray, maximum: int = 5):
    syndromes = column_syndromes(hcheck)
    stabilizers = rowspace_basis(hstab)

    def nontrivial(mask: int) -> bool:
        return not in_span(mask, stabilizers)

    by_single = defaultdict(list)
    for i, syndrome in enumerate(syndromes):
        by_single[syndrome].append(i)
        if syndrome == 0 and nontrivial(1 << i):
            return 1, [i]
    if maximum == 1:
        return None

    pairs = defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            syndrome = syndromes[i] ^ syndromes[j]
            mask = (1 << i) | (1 << j)
            pairs[syndrome].append((i, j, mask))
            if syndrome == 0 and nontrivial(mask):
                return 2, [i, j]
    if maximum == 2:
        return None

    for i in range(N):
        for j in range(i + 1, N):
            target = syndromes[i] ^ syndromes[j]
            for k in by_single.get(target, ()):
                if k <= j:
                    continue
                mask = (1 << i) | (1 << j) | (1 << k)
                if nontrivial(mask):
                    return 3, [i, j, k]
    if maximum == 3:
        return None

    for same_syndrome_pairs in pairs.values():
        if len(same_syndrome_pairs) < 2:
            continue
        for left_index in range(len(same_syndrome_pairs)):
            i, j, left = same_syndrome_pairs[left_index]
            for k, l, right in same_syndrome_pairs[left_index + 1:]:
                if left & right:
                    continue
                mask = left | right
                if nontrivial(mask):
                    return 4, sorted((i, j, k, l))
    if maximum == 4:
        return None

    for i in range(N):
        si = syndromes[i]
        for j in range(i + 1, N):
            sij = si ^ syndromes[j]
            pair_mask = (1 << i) | (1 << j)
            for k in range(j + 1, N):
                target = sij ^ syndromes[k]
                triple_mask = pair_mask | (1 << k)
                for l, m, other_mask in pairs.get(target, ()):
                    if triple_mask & other_mask:
                        continue
                    mask = triple_mask | other_mask
                    if nontrivial(mask):
                        return 5, sorted((i, j, k, l, m))
    return None


def connected(hx: np.ndarray, hz: np.ndarray) -> bool:
    total = N + hx.shape[0] + hz.shape[0]
    adjacency = [[] for _ in range(total)]
    for offset, checks in ((N, hx), (N + hx.shape[0], hz)):
        for row, support in enumerate(checks):
            check = offset + row
            for qubit in np.flatnonzero(support):
                qubit = int(qubit)
                adjacency[check].append(qubit)
                adjacency[qubit].append(check)
    seen = {0}
    todo = deque([0])
    while todo:
        vertex = todo.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return len(seen) == total


def translation_permutation(shift: tuple[int, int]) -> np.ndarray:
    base = np.asarray([idx(add(g, shift)) for g in GROUP], dtype=np.int64)
    return np.concatenate([block * ORDER + base for block in range(5)])


def permute_columns(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(matrix)
    output[:, permutation] = matrix
    return output


def same_rowspace(left: np.ndarray, right: np.ndarray) -> bool:
    return rank2(left) == rank2(right) == rank2(np.vstack([left, right]))


def fold_permutation(phi) -> np.ndarray:
    theta = [idx(phi(neg(g))) for g in GROUP]
    block_map = [0, 2, 1, 3, 4]
    return np.asarray([
        block_map[block] * ORDER + theta[within]
        for block in range(5)
        for within in range(ORDER)
    ], dtype=np.int64)


def support_to_json(support):
    return [[int(a), int(b)] for a, b in support]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=160326)
    parser.add_argument("--checkpoint", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    automorphisms = involutive_automorphisms()
    print(json.dumps({
        "stage": "start",
        "involutive_nonidentity_automorphisms": len(automorphisms),
        "trials": args.trials,
        "seed": args.seed,
    }, sort_keys=True), flush=True)
    rejected = defaultdict(int)
    started = time.perf_counter()
    nonidentity = GROUP[1:]
    for trial in range(1, args.trials + 1):
        if args.checkpoint and trial > 1 and (trial - 1) % args.checkpoint == 0:
            print(json.dumps({
                "stage": "checkpoint",
                "trials_completed": trial - 1,
                "seconds": round(time.perf_counter() - started, 3),
                "rejected": dict(rejected),
            }, sort_keys=True), flush=True)
        image_x, image_y, phi = rng.choice(automorphisms)
        a = [(0, 0), *rng.sample(nonidentity, 2)]
        b = [(0, 0), *rng.sample(nonidentity, 2)]
        if len(set(a)) < 3 or len(set(b)) < 3:
            rejected["duplicate"] += 1
            continue
        hx, hz, c, d = build_checks(a, b, phi)
        if np.any((hx @ hz.T) % 2):
            raise AssertionError("CSS failure")
        if rank2(hx) != 64 or rank2(hz) != 64:
            rejected["rank"] += 1
            continue
        if not connected(hx, hz):
            rejected["disconnected"] += 1
            continue
        witness = low_logical(hx, hz, maximum=5)
        if witness is not None:
            rejected[f"d{witness[0]}"] += 1
            continue
        fold = fold_permutation(phi)
        if not same_rowspace(permute_columns(hx, fold), hz):
            raise AssertionError("fold failure")
        for shift in ((1, 0), (0, 1)):
            translation = translation_permutation(shift)
            if not same_rowspace(permute_columns(hx, translation), hx):
                raise AssertionError("translation X failure")
            if not same_rowspace(permute_columns(hz, translation), hz):
                raise AssertionError("translation Z failure")
        syzygies = []
        for first, third in ((b, a),):
            mask = 0
            for term in map(neg, first):
                mask |= 1 << idx(term)
            for term in map(neg, third):
                mask |= 1 << (2 * ORDER + idx(term))
            if not in_span(mask, rowspace_basis(hz)):
                syzygies.append([i for i in range(N) if mask >> i & 1])
        record = {
            "stage": "hit",
            "trial": trial,
            "seconds": round(time.perf_counter() - started, 6),
            "parameters": "[[160,32,6]]",
            "check_weight": 9,
            "group": "C4 x C8",
            "phi_x": support_to_json([image_x])[0],
            "phi_y": support_to_json([image_y])[0],
            "a": support_to_json(a),
            "b": support_to_json(b),
            "c": support_to_json(c),
            "d": support_to_json(d),
            "weight_six_z_syzygies": syzygies,
            "rejected": dict(rejected),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        return
    print(json.dumps({
        "stage": "done",
        "trials": args.trials,
        "seconds": round(time.perf_counter() - started, 3),
        "rejected": dict(rejected),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
