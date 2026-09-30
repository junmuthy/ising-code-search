"""Exact clean-backaction scheduling; the order-four fold is applied forward."""
from __future__ import annotations

import json
import random
import time

import z3

from model import digest, move


def verify(code, record):
    times = {}
    for t, layer in enumerate(record['layers']):
        assert len({g['data'] for g in layer}) == len(layer)
        assert len({(g['type'], g['check']) for g in layer}) == len(layer)
        for g in layer:
            edge = g['type'], g['check'], g['data']
            assert edge not in times
            times[edge] = t
    expected = {(s, r, q) for s in ('X', 'Z') for r, row in enumerate(code['supports'][s]) for q in row}
    assert set(times) == expected and len(record['layers']) == record['depth']
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            assert sum(times['X', xr, q] < times['Z', zr, q] for q in set(xs) & set(zs)) % 2 == 0
    return dict(collision_free=True, exact_edge_coverage=True, clean_backaction=True,
                cnots=len(times), layer_sizes=list(map(len, record['layers'])),
                minimum_depth_lower_bound=max(map(len, code['supports']['X'] + code['supports']['Z'])))


def record(code, times, depth, method):
    layers = [[] for _ in range(depth)]
    for (kind, row, q), t in sorted(times.items()):
        layers[t].append(dict(type=kind, check=row, data=q))
    core = dict(depth=depth, layers=layers, basis_id=code['basis_id'], method=method)
    core['schedule_id'] = digest(json.dumps(core, sort_keys=True).encode())[:16]
    core['validation'] = verify(code, core)
    return core


def separated_schedule(code, seed=1):
    """Exact bipartite edge coloring by perfect matchings of a regular completion.

    X and Z orderings are sampled independently. A complete X block followed
    by a complete Z block has clean backaction for every CSS-orthogonal pair.
    """
    rng = random.Random(seed)
    times = {}
    for kind in ('X', 'Z'):
        rows = code['supports'][kind]
        degree = max(max(map(len, rows)), max(sum(q in row for row in rows) for q in range(64)))
        counts = [[0] * 64 for _ in range(64)]
        original = {(r, q) for r, row in enumerate(rows) for q in row}
        for r, q in original:
            counts[r][q] = 1
        left = [degree - sum(row) for row in counts]
        right = [degree - sum(row[q] for row in counts) for q in range(64)]
        order = list(range(64))
        rng.shuffle(order)
        for r in order:
            columns = list(range(64))
            rng.shuffle(columns)
            for q in columns:
                add = min(left[r], right[q])
                counts[r][q] += add
                left[r] -= add
                right[q] -= add
        assert not any(left + right)
        for color in range(degree):
            adjacency = [[q for q in range(64) if counts[r][q]] for r in range(64)]
            for row in adjacency:
                rng.shuffle(row)
            matched = {}
            def augment(r, seen):
                for q in adjacency[r]:
                    if q in seen:
                        continue
                    seen.add(q)
                    if q not in matched or augment(matched[q], seen):
                        matched[q] = r
                        return True
                return False
            rng.shuffle(order)
            assert all(augment(r, set()) for r in order)
            for q, r in matched.items():
                counts[r][q] -= 1
                if (r, q) in original:
                    times[kind, r, q] = color + (0 if kind == 'X' else 12)
                    original.remove((r, q))
        assert not original and degree == 12
    return record(code, times, 24, f'separated_independent_edge_color_seed_{seed}')


def folded_solver(code, depth, seed=1, timeout_ms=5000, separated=False):
    # Independent check lists need not be closed under P^2; only use the
    # explicitly verified X -> Z row pairing, never P == P^-1.
    assert [move(w, code['fold']) for w in code['checks']['X']] == code['checks']['Z']
    edges = [(r, q) for r, row in enumerate(code['supports']['X']) for q in row]
    if code['presentation'] == 'redundant':
        def key(r, q):
            return q // 32, ((q % 32) // 4 - r // 4) % 8, (q % 4 - r % 4) % 4
    else:
        def key(r, q):
            return r, q
    keys = sorted({key(r, q) for r, q in edges})
    lookup = {k: i for i, k in enumerate(keys)}
    variables = [z3.Int(f't{i}') for i in range(len(keys))]
    xmap = {(r, q): lookup[key(r, q)] for r, q in edges}
    zmap = {(r, code['fold'][q]): i for (r, q), i in xmap.items()}
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=seed)
    for v in variables:
        solver.add(v >= 0, v < (depth // 2 if separated else depth))
    def tick(kind, r, q):
        return variables[xmap[r, q]] if kind == 'X' else depth - 1 - variables[zmap[r, q]]
    for kind in ('X', 'Z'):
        for r, row in enumerate(code['supports'][kind]):
            solver.add(z3.Distinct([tick(kind, r, q) for q in row]))
    for q in range(64):
        solver.add(z3.Distinct([tick(s, r, q) for s in ('X', 'Z')
                               for r, row in enumerate(code['supports'][s]) if q in row]))
    if not separated:
        constraints = set()
        for xr, xs in enumerate(code['supports']['X']):
            for zr, zs in enumerate(code['supports']['Z']):
                odd = set()
                for q in set(xs) & set(zs):
                    pair = tuple(sorted((xmap[xr, q], zmap[zr, q])))
                    odd.symmetric_difference_update([pair])
                if odd:
                    constraints.add(tuple(sorted(odd)))
        for terms in sorted(constraints):
            solver.add(z3.Sum([z3.If(variables[a] + variables[b] < depth - 1, 1, 0)
                              for a, b in terms]) % 2 == 0)
    def next_record():
        start = time.monotonic()
        result = solver.check()
        info = dict(status=str(result), seconds=time.monotonic() - start, depth=depth,
                    variables=len(variables), separated=separated, seed=seed,
                    reason=solver.reason_unknown() if result == z3.unknown else None)
        if result != z3.sat:
            return None, info
        values = [solver.model().eval(v).as_long() for v in variables]
        solver.add(z3.Sum([z3.If(v != a, 1, 0) for v, a in zip(variables, values)]) >= max(1, len(values) // 4))
        times = {(s, r, q): (values[xmap[r, q]] if s == 'X' else depth - 1 - values[zmap[r, q]])
                 for s in ('X', 'Z') for r, row in enumerate(code['supports'][s]) for q in row}
        return record(code, times, depth, 'paired_fold_reverse' + ('_separated' if separated else '')), info
    return next_record
