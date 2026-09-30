"""Bounded scheduling campaign for the fixed all-weight-six-logical code.

All outputs go to new directories. Archived code and prior experiments are
read-only inputs. Circuit certificates are separate from screening results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import z3

from orbit_campaign import save
from orbit_schedule import (HERE, assess, binary, materialize, old_schedule,
                            prepare, separated, solver)
from sparse_core import permute, rank, same_space
from sparse_finalize import balanced_basis

ROOT = HERE / 'orbit_campaign'
KEY = '313c98984982b039f1068bfd6001a4ffceb370b2091c3edbc9d3f8b915aaa028'
LOGICALS = ROOT / 'handoff_v1/alternate_w12_all_weight6_logicals/candidate.json'
OLD = ROOT / 'schedules_313c9898_v1'
CATALOG = ROOT / 'all_finalists_v1' / KEY / 'weight_enumeration.json'


def pinned_data():
    data = json.loads(LOGICALS.read_text())
    assert all(w.bit_count() == 6 for k in ('x', 'z') for w in data['certificate'][k])
    assert rank(data['hx']) == rank(data['hz']) == 24
    return data


def source_schedule(folder):
    old = json.loads((folder / 'code.json').read_text())
    data = pinned_data()
    assert all(same_space(old['checks'][s], data[k]) for s, k in (('X', 'hx'), ('Z', 'hz')))
    code = prepare(dict(data, hx=old['checks']['X'], hz=old['checks']['Z']), old['presentation'])
    schedule = json.loads((folder / 'schedule.json').read_text())
    old_schedule.verify(code, schedule)
    assert schedule['basis_id'] == code['basis_id']
    return code, schedule


def select_indices(words, count, seed):
    """Choose a full-rank measured subset, optionally with extra checks."""
    if not rank(words) <= count <= len(words):
        raise ValueError('retained check count cannot span the original space')
    order = list(range(len(words)))
    random.Random(seed).shuffle(order)
    chosen, selected = [], []
    for i in order:
        if rank(selected + [words[i]]) > len(selected):
            chosen.append(i)
            selected.append(words[i])
    chosen += [i for i in order if i not in chosen][:count-len(chosen)]
    assert len(chosen) == count and same_space([words[i] for i in chosen], words)
    return sorted(chosen)


def restrict_rank(code, schedule, seed, count=24, paired=True):
    chosen = {s: select_indices(rows, count, seed + (0 if s == 'X' or paired else 100000))
              for s, rows in code['checks'].items()}
    checks = {s: [rows[i] for i in chosen[s]] for s, rows in code['checks'].items()}
    new = prepare(dict(code['data'], hx=checks['X'], hz=checks['Z']),
                  'independent_subset' if count == 24 else 'partial_redundancy')
    lookup = {s: {old: new for new, old in enumerate(chosen[s])} for s in ('X', 'Z')}
    times = {(g['type'], lookup[g['type']][g['check']], g['data']): t
             for t, layer in enumerate(schedule['layers']) for g in layer
             if g['check'] in lookup[g['type']]}
    result = materialize(new, times, schedule['depth'],
                         f'rank_subset_{schedule["schedule_id"]}_{seed}_{count}_{paired}')
    result.update(source_schedule_id=schedule['schedule_id'], retained_check_indices=chosen,
                  paired_subsets=paired, seed=seed)
    return new, result


def optimal_basis(seed):
    data = pinned_data()
    words = json.loads(CATALOG.read_text())['hx']['low_weight_rows']
    hx, load = balanced_basis(words, data['certificate']['fold'], trials=32, seed=seed)
    assert Counter(w.bit_count() for w in hx) == {8: 16, 12: 8}
    hz = [permute(v, data['certificate']['fold']) for v in hx]
    code = prepare(dict(data, hx=hx, hz=hz), 'optimal_independent')
    code['basis_selection'] = load
    return code


def translation_edges(code):
    """Recover the 12-edge cyclic orbit parameterization from physical actions."""
    words = code['checks']['X']
    index = {word: i for i, word in enumerate(words)}
    assert len(index) == len(words) == 32
    seed = words[0]
    points = binary.supports([seed])[0]
    todo = [list(range(64))]
    seen, edges = set(), {}
    while todo:
        p = todo.pop()
        word = permute(seed, p)
        if word in seen:
            continue
        seen.add(word)
        r = index[word]
        for k, q in enumerate(points):
            edges[r, p[q]] = k
        for action in (code['data']['px'], code['data']['py']):
            todo.append([action[q] for q in p])
    assert seen == set(words) and len(edges) == sum(w.bit_count() for w in words)
    return edges


def general_solver(code, depth, seed, milliseconds):
    """No fold/time-reversal constraint: only coverage, collisions, backaction."""
    edges = [(s, r, q) for s in ('X', 'Z') for r, row in enumerate(code['supports'][s]) for q in row]
    variables = {e: z3.Int(f't{i}') for i, e in enumerate(edges)}
    sat = z3.Solver()
    sat.set(timeout=milliseconds, random_seed=seed)
    for v in variables.values():
        sat.add(v >= 0, v < depth)
    for kind in ('X', 'Z'):
        for r, row in enumerate(code['supports'][kind]):
            sat.add(z3.Distinct([variables[kind, r, q] for q in row]))
    for q in range(64):
        sat.add(z3.Distinct([v for e, v in variables.items() if e[2] == q]))
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            overlap = set(xs) & set(zs)
            if overlap:
                sat.add(z3.Sum([z3.If(variables['X', xr, q] < variables['Z', zr, q], 1, 0)
                                for q in overlap]) % 2 == 0)
    def next_schedule():
        start = time.monotonic()
        status = sat.check()
        info = dict(status=str(status), seconds=time.monotonic()-start, depth=depth,
                    variables=len(variables), reason=sat.reason_unknown() if status == z3.unknown else None)
        if status != z3.sat:
            return None, info
        model = sat.model()
        times = {e: model.eval(v).as_long() for e, v in variables.items()}
        sat.add(z3.Or([v != times[e] for e, v in variables.items()]))
        return materialize(code, times, depth, 'unpaired_clean_backaction'), info
    return next_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'subsets', 'optimal', 'redundant', 'general'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--retain', type=int, default=24)
    parser.add_argument('--unpaired', action='store_true')
    parser.add_argument('--depths', type=int, nargs='+', default=[12, 14, 16])
    parser.add_argument('--solver-ms', type=int, default=10000)
    parser.add_argument('--heuristic-seconds', type=float, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', mode=args.mode, parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  contract=dict(code=KEY, logical_basis=str(LOGICALS), noise='full_wait', rounds=3,
                                observables=16, memory_bases=['X', 'Z']), schedules=[], attempts=[],
                  input_sha256={str(p): binary.digest(p.read_bytes()) for p in (LOGICALS, CATALOG)},
                  source_sha256={str(p): binary.digest(p.read_bytes()) for p in
                                 (Path(__file__).resolve(), HERE/'orbit_schedule.py', HERE/'sparse_finalize.py',
                                  HERE/'distance6_fault_search/circuit.py', HERE/'distance6_fault_search/schedule.py')})
    sources = sorted(OLD.glob('redundant_*'))
    def checkpoint():
        save(args.output/'report.json', report)
    seen = set()
    def screen(code, schedule):
        fingerprint = binary.digest(json.dumps(dict(checks=code['checks'], layers=schedule['layers']), sort_keys=True).encode())
        if fingerprint in seen:
            report['attempts'].append(dict(status='duplicate_schedule', fingerprint=fingerprint))
            checkpoint()
            return
        seen.add(fingerprint)
        result = assess(code, schedule, args.output/schedule['schedule_id'], args.heuristic_seconds)
        result['physical_schedule_sha256'] = fingerprint
        report['schedules'].append(result)
        checkpoint()
    checkpoint()
    if args.mode == 'baseline':
        for path in sources + sorted(OLD.glob('independent_*')):
            code, schedule = source_schedule(path)
            screen(code, schedule)
    elif args.mode == 'subsets':
        for trial in range(args.count):
            code, schedule = source_schedule(sources[trial % len(sources)])
            new, reduced = restrict_rank(code, schedule, args.seed+trial, args.retain, not args.unpaired)
            screen(new, reduced)
    else:
        for trial in range(args.count):
            seed = args.seed + trial
            if args.mode == 'redundant':
                code, _ = source_schedule(sources[0])
                code['edge_map'] = translation_edges(code)
            else:
                code = optimal_basis(seed)
            for depth in args.depths:
                fn = general_solver if args.mode == 'general' else solver
                next_schedule = fn(code, depth, seed=seed, milliseconds=args.solver_ms)
                schedule, info = next_schedule()
                report['attempts'].append(dict(trial=trial, seed=seed, method=args.mode,
                                                checks=code['checks'], **info))
                print(json.dumps(dict(stage='solver', trial=trial, method=args.mode, **info)), flush=True)
                checkpoint()
                if schedule is not None:
                    screen(code, schedule)
                    break
    report['status'] = 'complete_bounded_screen'
    checkpoint()


if __name__ == '__main__':
    main()
