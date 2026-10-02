"""Schedule and memory-screen generalized orbit codes, without editing old tools."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import z3

from sparse_core import HERE, permute, rank, same_space, space_key
from orbit_extension import geometry
from orbit_campaign import save


def load_module(name, path, aliases=None):
    previous = {k: sys.modules.get(k) for k in (aliases or {})}
    sys.modules.update(aliases or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in previous.items():
            if v is None:
                del sys.modules[k]
            else:
                sys.modules[k] = v


D6 = HERE/'distance6_fault_search'
binary = load_module('orbit_binary_helpers', D6/'model.py')
circuits = load_module('orbit_memory_circuit', D6/'circuit.py')
old_schedule = load_module('orbit_schedule_verifier', D6/'schedule.py', {'model': binary})
faults = load_module('orbit_fault_tools', D6/'faults.py', {'model': binary, 'circuit': circuits})


def prepare(data, presentation):
    checks = dict(X=data['hx'], Z=data['hz'])
    edge_map = None
    if presentation == 'full_orbits':
        assert data['construction'] == 'multi_orbit_four_sheet'
        actions = geometry('four_sheet')['actions']
        checks['X'] = [permute(seed, p) for seed in data['seeds'] for p in actions]
        checks['Z'] = [permute(v, data['certificate']['fold']) for v in checks['X']]
        edge_map, keys = {}, {}
        for s, seed in enumerate(data['seeds']):
            for g, p in enumerate(actions):
                for q in range(64):
                    if seed >> q & 1:
                        key = s, q
                        if key not in keys:
                            keys[key] = len(keys)
                        edge_map[s*len(actions) + g, p[q]] = keys[key]
    for s, key in (('X', 'hx'), ('Z', 'hz')):
        assert same_space(checks[s], data[key]) and rank(checks[s]) == 24
    return dict(data=data, checks=checks, fold=data['certificate']['fold'],
                logicals=dict(X=data['certificate']['x'], Z=data['certificate']['z']),
                supports={s: binary.supports(rows) for s, rows in checks.items()},
                logical_supports={s: binary.supports(data['certificate'][key]) for s, key in (('X', 'x'), ('Z', 'z'))},
                relations={s: binary.relations(rows) for s, rows in checks.items()},
                basis_id=space_key(checks['X'], checks['Z']), presentation=presentation,
                edge_map=edge_map)


def materialize(code, times, depth, method):
    schedule = old_schedule.record(code, times, depth, method)
    combined = [sum(q in row for rows in code['supports'].values() for row in rows) for q in range(64)]
    schedule['validation']['minimum_depth_lower_bound'] = max(max(combined), max(map(len, code['supports']['X'] + code['supports']['Z'])))
    return schedule


def separated(code, seed):
    rng = random.Random(seed)
    times, offset = {}, 0
    for kind in ('X', 'Z'):
        rows = code['supports'][kind]
        size = max(64, len(rows))
        counts = [[0] * size for _ in range(size)]
        original = {(r, q) for r, row in enumerate(rows) for q in row}
        for r, q in original:
            counts[r][q] = 1
        degree = max(max(map(sum, counts)), max(sum(row[q] for row in counts) for q in range(size)))
        left = [degree - sum(row) for row in counts]
        right = [degree - sum(row[q] for row in counts) for q in range(size)]
        order = list(range(size))
        rng.shuffle(order)
        for r in order:
            columns = list(range(size))
            rng.shuffle(columns)
            for q in columns:
                add = min(left[r], right[q])
                counts[r][q] += add
                left[r] -= add
                right[q] -= add
        assert not any(left + right)
        for color in range(degree):
            adjacency = [[q for q in range(size) if counts[r][q]] for r in range(size)]
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
                    times[kind, r, q] = offset + color
                    original.remove((r, q))
        assert not original
        offset += degree
    return materialize(code, times, offset, f'general_separated_seed_{seed}')


def solver(code, depth, seed=1, milliseconds=5000):
    assert [permute(v, code['fold']) for v in code['checks']['X']] == code['checks']['Z']
    edges = [(r, q) for r, row in enumerate(code['supports']['X']) for q in row]
    xmap = code['edge_map'] or {edge: i for i, edge in enumerate(edges)}
    zmap = {(r, code['fold'][q]): i for (r, q), i in xmap.items()}
    variables = [z3.Int(f'c{i}') for i in range(max(xmap.values()) + 1)]
    s = z3.Solver()
    s.set(timeout=milliseconds, random_seed=seed)
    for c in variables:
        s.add(c >= 0, c < depth)
    def tick(kind, r, q):
        return variables[xmap[r, q]] if kind == 'X' else depth - 1 - variables[zmap[r, q]]
    for kind in ('X', 'Z'):
        for r, row in enumerate(code['supports'][kind]):
            s.add(z3.Distinct([tick(kind, r, q) for q in row]))
    for q in range(64):
        s.add(z3.Distinct([tick(kind, r, q) for kind in ('X', 'Z') for r, row in enumerate(code['supports'][kind]) if q in row]))
    constraints = set()
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            odd = set()
            for q in set(xs) & set(zs):
                odd.symmetric_difference_update([tuple(sorted((xmap[xr, q], zmap[zr, q])))])
            if odd:
                constraints.add(tuple(sorted(odd)))
    for terms in constraints:
        s.add(z3.Sum([z3.If(variables[a] + variables[b] < depth - 1, 1, 0) for a, b in terms]) % 2 == 0)
    def next_schedule():
        start = time.monotonic()
        result = s.check()
        info = dict(status=str(result), seconds=time.monotonic()-start, depth=depth,
                    variables=len(variables), reason=s.reason_unknown() if result == z3.unknown else None)
        if result != z3.sat:
            return None, info
        colors = [s.model().eval(v).as_long() for v in variables]
        s.add(z3.Or([v != a for v, a in zip(variables, colors)]))
        times = {(kind, r, q): (colors[xmap[r, q]] if kind == 'X' else depth - 1 - colors[zmap[r, q]])
                 for kind in ('X', 'Z') for r, row in enumerate(code['supports'][kind]) for q in row}
        return materialize(code, times, depth, 'generic_fold_reverse'), info
    return next_schedule


def assess(code, schedule, destination, seconds):
    destination.mkdir(parents=True, exist_ok=False)
    save(destination/'schedule.json', schedule)
    # Tuple-key edge maps are construction intermediates, not circuit inputs.
    save(destination/'code.json', {k: v for k, v in code.items() if k != 'edge_map'})
    result = dict(schedule_id=schedule['schedule_id'], depth=schedule['depth'],
                  presentation=code['presentation'], cnots=schedule['validation']['cnots'], cases={})
    for basis in ('X', 'Z'):
        c = circuits.build(code, schedule, basis, noise='full_wait', rounds=3)
        ideal = c.without_noise()
        ideal.detector_error_model(allow_gauge_detectors=False)
        d, o = ideal.compile_detector_sampler(seed=17).sample(64, separate_observables=True)
        assert not np.any(d) and not np.any(o)
        path = destination/f'{basis}.stim'
        c.to_file(path)
        h = faults.bounded_heuristic(path, destination/f'{basis}_heuristic.json', seconds)
        result['cases'][basis] = dict(upper_bound=h.get('upper_bound'), heuristic=h,
                                     circuit_sha256=binary.digest(str(c).encode()), qubits=c.num_qubits,
                                     detectors=c.num_detectors, ideal_validated=True)
    result['status'] = ('rejected_by_physical_witness' if any(c['upper_bound'] is not None and c['upper_bound'] < 5 for c in result['cases'].values())
                        else 'screen_survivor_not_certified')
    save(destination/'result.json', result)
    print(json.dumps(dict(stage='screened', schedule=result['schedule_id'], presentation=result['presentation'],
                          depth=result['depth'], upper={s:c['upper_bound'] for s,c in result['cases'].items()})), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--solver-ms', type=int, default=5000)
    p.add_argument('--heuristic-seconds', type=float, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.candidate.read_text())
    report = dict(status='running', candidate=str(args.candidate), candidate_sha256=binary.digest(args.candidate.read_bytes()),
                  contract=dict(rounds=3, noise='full_wait', bases=['X', 'Z']), schedules=[], solver_attempts=[],
                  source_sha256={str(f):binary.digest(f.read_bytes()) for f in
                                 (Path(__file__).resolve(), D6/'circuit.py', D6/'model.py', D6/'schedule.py',
                                  D6/'faults.py', HERE/'fault_distance_milestone1/faults.py')})
    for presentation in ('independent', 'full_orbits'):
        code = prepare(data, presentation)
        baseline = separated(code, 20260914)
        report['schedules'].append(assess(code, baseline, args.output/(presentation+'_'+baseline['schedule_id']), args.heuristic_seconds))
        for depth in range(baseline['validation']['minimum_depth_lower_bound'], 15):
            next_schedule = solver(code, depth, milliseconds=args.solver_ms)
            found = 0
            for _ in range(args.count):
                schedule, info = next_schedule()
                report['solver_attempts'].append(dict(presentation=presentation, **info))
                print(json.dumps(dict(stage='solver', presentation=presentation, **info)), flush=True)
                if schedule is None:
                    break
                report['schedules'].append(assess(code, schedule, args.output/(presentation+'_'+schedule['schedule_id']), args.heuristic_seconds))
                found += 1
                save(args.output/'report.json', report)
            if found:
                break
    report['status'] = 'complete_bounded_screen'
    save(args.output/'report.json', report)


if __name__ == '__main__':
    main()
