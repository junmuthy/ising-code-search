"""Bounded multi-orbit CSS exploration; no family completeness claims.

Rows are bit-packed physical supports. All rank, commutation, orbit closure,
and logical-module acceptance checks are exact; construction is heuristic.
"""
from __future__ import annotations

from collections import Counter
import itertools
import random

from sparse_core import (basis, common_h, commute, grid_module, permute, power,
                         rank, reduce, rref, same_space)
from sparse_models import SmallGroup, translations
from sparse_twisted import fold_catalog
from algebra import combine, nullspace
from search import sparse_vectors


def orbit(word, actions):
    return sorted({permute(word, p) for p in actions})


def module_generators(rows, generators):
    """dim M / I M for a verified finite 2-group module.

    Nakayama's lemma makes this the minimum number of orbit generators.
    Counting supplied seed rows alone does not establish noncyclicity.
    """
    b = basis(rows)
    image = []
    for p in generators:
        for v in b.values():
            moved = permute(v, p)
            assert not reduce(moved, b)
            image.append(v ^ moved)
    return len(b) - rank(image)


def geometry(family, variant=0):
    if family == 'four_sheet':
        e, sheets = SmallGroup('c4c4'), 4
        actions = [[s * 16 + e.mul(h, g) for s in range(4) for h in range(16)] for g in range(16)]
        px, py = translations()
        central = None
        folds = []
        sigmas = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0),
                  (1, 0, 2, 3), (0, 1, 3, 2)]
        for a, b, c, d in ((1, 0, 0, 1), (3, 0, 0, 3), (0, 1, 1, 0),
                           (1, 0, 2, 3), (1, 2, 0, 3), (3, 0, 0, 1)):
            theta = [4 * ((a*i+b*j) % 4) + (c*i+d*j) % 4 for i in range(4) for j in range(4)]
            assert power(theta, 2) == list(range(16))
            for sigma in sigmas:
                for shift in (0, 1, 4, 5):
                    offsets = [0] * 4
                    for s in range(4):
                        if sigma[s] > s:
                            offsets[s] = shift
                            offsets[sigma[s]] = theta[e.inverse[shift]]
                    p = [16*sigma[s] + e.mul(theta[g], offsets[s]) for s in range(4) for g in range(16)]
                    assert power(p, 2) == list(range(64))
                    folds.append(p)
    elif family == 'two_sheet':
        e, sheets = SmallGroup('c8c4'), 2
        actions = [e.right(g) for g in range(32)]
        px, py, central = e.physical()
        folds = list(fold_catalog())
    elif family == 'nonabelian':
        e, sheets = SmallGroup('extension', variant // 2, variant % 2), 2
        actions = [e.right(g) for g in range(32)]
        px, py, central = e.physical()
        folds = []
        for p in e.folds():
            theta = [q % 32 for q in p[:32]]
            if power(theta, 2) != list(range(32)):
                continue
            for shift in (0, e.generators[0], e.generators[1]):
                offsets = [shift, theta[e.inverse[shift]]]
                f = [(1-s)*32 + e.mul(offsets[s], theta[g]) for s in range(2) for g in range(32)]
                if power(f, 2) == list(range(64)):
                    folds.append(f)
    else:
        raise ValueError(family)
    folds = [list(p) for p in dict.fromkeys(map(tuple, folds))]
    generators = [px, py] + ([] if central is None else [central])
    # Each fold must normalize the actual shared physical translation group.
    group = set(map(tuple, actions))
    for p in folds:
        inverse = [p.index(q) for q in range(64)]
        for g in generators:
            assert tuple(p[g[inverse[q]]] for q in range(64)) in group
        assert tuple(power(p, 2)) in group
    return dict(family=family, variant=variant, sheets=sheets, actions=actions,
                px=px, py=py, central=central, generators=generators, folds=folds)


def factor(word, geo, rng):
    # Group differences preserve invariant constraint kernels. They sample
    # smaller orbit ranks; they are not a classification of module types.
    for _ in range(rng.choice((0, 0, 1, 1, 2))):
        p = rng.choice(geo['actions'][1:])
        word ^= permute(word, p)
    return word


def propose(rows, fold, geo, rng, ceiling, count):
    if not rows:
        for _ in range(count * 6):
            weight = rng.randint(4, ceiling)
            word = sum(1 << q for q in rng.sample(range(64), weight))
            if rng.randrange(2):
                word = factor(word, geo, rng)
            if word and 4 <= word.bit_count() <= ceiling:
                yield word
        return
    inverse = [fold.index(q) for q in range(64)]
    constraints = [permute(v, p) for p in (fold, inverse) for v in rows]
    kernel = nullspace(constraints, 64)
    seen = set()
    for raw in sparse_vectors(kernel, rng, count, ceiling):
        for word in (raw, factor(raw, geo, rng), factor(raw, geo, rng)):
            if word and word.bit_count() <= ceiling and word not in seen:
                assert all((word & c).bit_count() % 2 == 0 for c in constraints)
                seen.add(word)
                yield word


def construct(geo, fold, rng, ceiling, stats, *, proposals=64):
    rows, seeds, increments = [], [], []
    for stage in range(6):
        current = rank(rows)
        choices = []
        for word in propose(rows, fold, geo, rng, ceiling, proposals):
            stats['seed_proposals'] += 1
            added = orbit(word, geo['actions'])
            candidate = rref(rows + added)
            newrank = len(candidate)
            if current == 0 and newrank == 24:
                stats['first_seed_rank24_skipped'] += 1
                continue
            if newrank <= current or newrank > 24:
                stats['seed_rank_reject'] += 1
                continue
            dual = [permute(v, fold) for v in candidate]
            if not commute(candidate, dual):
                stats['seed_commutation_reject'] += 1
                continue
            assert same_space([permute(v, power(fold, 2)) for v in candidate], candidate)
            choices.append((word, added, newrank))
            if newrank == 24 or len(choices) >= 8:
                break
        if not choices:
            stats[f'construction_stalled_rank_{current}'] += 1
            return None
        complete = [item for item in choices if item[2] == 24]
        word, added, newrank = rng.choice(complete or choices)
        seeds.append(word)
        increments.append(newrank - current)
        rows = list(dict.fromkeys(rows + added))
        if newrank == 24:
            stats['rank24_constructed'] += 1
            return dict(hx=rows, hz=[permute(v, fold) for v in rows], seeds=seeds,
                        orbit_rank_increments=increments,
                        module_generator_count=module_generators(rows, geo['generators']))
    stats['construction_stage_cap'] += 1
    return None


def low_witness(h, stabilizers, ceiling=3):
    """Small exact negative filter; returned errors are tested modulo stabilizers."""
    columns = [sum(((v >> q) & 1) << i for i, v in enumerate(rref(h))) for q in range(64)]
    space = basis(stabilizers)
    for weight in range(1, ceiling + 1):
        for support in itertools.combinations(range(64), weight):
            syndrome = 0
            word = 0
            for q in support:
                syndrome ^= columns[q]
                word |= 1 << q
            if syndrome == 0 and reduce(word, space):
                return list(support)
    return None


def evaluate(candidate, geo, fold):
    hx, hz = candidate['hx'], candidate['hz']
    for side, h, stabilizers in (('Z', hx, hz), ('X', hz, hx)):
        witness = low_witness(h, stabilizers, 2)
        if witness is not None:
            return dict(status='short_logical', sector=side, witness=witness), None
    info, grid = grid_module(hx, hz, geo['px'], geo['py'], geo['central'])
    if grid is None:
        return info, None
    from algebra import low_logical
    for side, h, dual in (('Z', hx, grid['x']), ('X', hz, grid['z'])):
        witness = low_logical(h, dual, 64)
        if witness is not None:
            return dict(status='short_logical', sector=side, witness=witness), None
    hinfo, logical = common_h(hx, hz, grid, fold)
    if logical is None:
        return dict(status='common_h_rejected_for_this_fold', detail=hinfo), None
    from extract_component import components
    sizes = sorted(map(len, components(rref(hx) + rref(hz), 64)))
    if sizes != [64]:
        return dict(status='disconnected', component_sizes=sizes), None
    return dict(status='accepted', hadamard=hinfo), dict(logical, fold=fold)
