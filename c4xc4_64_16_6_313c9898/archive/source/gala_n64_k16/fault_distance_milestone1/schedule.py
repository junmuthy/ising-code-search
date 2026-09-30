"""Translation-invariant, fold/time-reversed syndrome schedules."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import time

import z3

from model import ROOT, digest, load, move, save


def geometry(code):
    def key(row, data):
        sheet, q = divmod(data, 32)
        return sheet, (q // 4 - row // 4) % 8, (q % 4 - row % 4) % 4
    keys = sorted({key(row, data) for row, support in enumerate(code['supports']['X']) for data in support})
    assert len(keys) == 12
    lookup = {value: i for i, value in enumerate(keys)}
    x = {(row, data): lookup[key(row, data)] for row, support in enumerate(code['supports']['X']) for data in support}
    z = {(row, code['fold'][data]): orbit for (row, data), orbit in x.items()}
    return keys, dict(X=x, Z=z)


def verify(code, record):
    depth, layers = record['depth'], record['layers']
    assert len(layers) == depth
    times = {}
    for tick, layer in enumerate(layers):
        assert len({g['data'] for g in layer}) == len(layer)
        assert len({(g['type'], g['check']) for g in layer}) == len(layer)
        for gate in layer:
            key = gate['type'], gate['check'], gate['data']
            assert key not in times
            times[key] = tick
    expected = {(kind, row, data) for kind in ('X', 'Z') for row, support in enumerate(code['supports'][kind]) for data in support}
    assert set(times) == expected and len(times) == 768
    for xrow, xs in enumerate(code['supports']['X']):
        for zrow, zs in enumerate(code['supports']['Z']):
            assert sum(times['X', xrow, q] < times['Z', zrow, q] for q in set(xs) & set(zs)) % 2 == 0
    for row, support in enumerate(code['supports']['X']):
        for q in support:
            assert times['X', row, q] + times['Z', row, code['fold'][q]] == depth - 1
    for generator in ('px', 'py'):
        p = code['data'][generator]
        for kind, rows in code['checks'].items():
            lookup = {word: i for i, word in enumerate(rows)}
            row_map = [lookup[move(word, p)] for word in rows]
            for row, support in enumerate(code['supports'][kind]):
                for q in support:
                    assert times[kind, row, q] == times[kind, row_map[row], p[q]]
    return dict(collision_free=True, clean_backaction=True, fold_time_reversal=True,
                translation_invariant=True, exact_edge_coverage=True, cnots=768,
                layer_sizes=list(map(len, layers)), minimum_depth_lower_bound=12,
                depth_optimal=(depth == 12))


def materialize(code, colors, depth):
    keys, maps = geometry(code)
    layers = [[] for _ in range(depth)]
    for kind in ('X', 'Z'):
        for (row, data), orbit in sorted(maps[kind].items()):
            tick = colors[orbit] if kind == 'X' else depth - 1 - colors[orbit]
            layers[tick].append(dict(type=kind, check=row, data=data))
    core = dict(depth=depth, colors=colors, edge_orbits=keys, layers=layers,
                candidate_sha256=code['provenance']['sha256']['candidate.json'])
    core['schedule_id'] = digest(json.dumps(core, sort_keys=True).encode())[:16]
    core['validation'] = verify(code, core)
    return core


def problem(code, depth, seed=1, timeout_ms=10000):
    _, maps = geometry(code)
    colors = [z3.Int(f'c_{i}') for i in range(12)]
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=seed)
    solver.add(*[z3.And(c >= 0, c < depth) for c in colors], z3.Distinct(colors))
    def tick(kind, row, data):
        color = colors[maps[kind][row, data]]
        return color if kind == 'X' else depth - 1 - color
    for data in range(64):
        solver.add(z3.Distinct([tick(kind, row, data) for kind in ('X', 'Z')
                                for row, support in enumerate(code['supports'][kind]) if data in support]))
    # Cancel duplicate comparisons before adding exact backaction constraints.
    constraints = set()
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            counts = Counter(tuple(sorted((maps['X'][xr, q], maps['Z'][zr, q]))) for q in set(xs) & set(zs))
            terms = tuple(sorted(pair for pair, count in counts.items() if count % 2))
            if terms:
                constraints.add(terms)
    for terms in sorted(constraints):
        solver.add(z3.Sum([z3.If(colors[a] + colors[b] < depth - 1, 1, 0) for a, b in terms]) % 2 == 0)
    # Only reverse the entire timeline; arbitrary layer relabeling is not safe.
    solver.add(2 * colors[0] < depth)
    return solver, colors, len(constraints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--timeout-ms', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    code = load()
    baseline = materialize(code, list(range(12)), 24)
    save(output / 'baseline_depth24.json', baseline)
    report = dict(status='running', schedules=[], baseline=baseline['schedule_id'], attempts=[])
    started = time.monotonic()
    for depth in (12, 14, 16):
        solver, colors, constraints = problem(code, depth, timeout_ms=args.timeout_ms)
        for index in range(args.count):
            result = solver.check()
            report['attempts'].append(dict(depth=depth, index=index, status=str(result), constraints=constraints,
                                           reason=solver.reason_unknown() if result == z3.unknown else None))
            print(json.dumps(report['attempts'][-1]), flush=True)
            if result != z3.sat:
                break
            values = [solver.model().eval(c).as_long() for c in colors]
            record = materialize(code, values, depth)
            path = f"schedule_{record['schedule_id']}.json"
            save(output / path, record)
            report['schedules'].append(dict(file=path, schedule_id=record['schedule_id'], depth=depth))
            save(output / 'report.json', report)
            solver.add(z3.Or([c != value for c, value in zip(colors, values)]))
        if report['schedules']:
            break
    report.update(status='complete', elapsed_seconds=time.monotonic() - started)
    save(output / 'report.json', report)


if __name__ == '__main__':
    main()
