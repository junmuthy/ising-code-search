"""Insert weight-eight checks around inherited weight-twelve orderings.

This is a separate driver so the earlier bounded campaign's source remains
unchanged. Fixed inherited edges reduce the free scheduling problem.
"""
import argparse
import json
from pathlib import Path
import random
import time

import z3

from orbit_campaign import save
from orbit_schedule import assess, binary, materialize, prepare
from schedule_313c9898 import CATALOG, OLD, pinned_data, source_schedule
from sparse_core import permute, rank, same_space


def seeded_basis(source, seed, original_count=8):
    data = pinned_data()
    small = [w for w in json.loads(CATALOG.read_text())['hx']['low_weight_rows'] if w.bit_count() == 8]
    assert len(small) == rank(small) == 16
    original = source['checks']['X']
    order = list(range(len(original)))
    random.Random(seed).shuffle(order)
    selected, rows = [], small[:]
    for i in order:
        if rank(rows + [original[i]]) > len(rows):
            selected.append(i)
            rows.append(original[i])
    assert len(selected) == 8 and len(rows) == rank(rows) == 24
    selected += [i for i in order if i not in selected][:original_count-8]
    hx = small + [original[i] for i in selected]
    hz = [permute(w, data['certificate']['fold']) for w in hx]
    assert same_space(hx, data['hx']) and same_space(hz, data['hz'])
    return prepare(dict(data, hx=hx, hz=hz), 'seeded_optimal' if original_count == 8 else 'seeded_partial'), selected


def solve(code, selected, reference, depth, seed, milliseconds):
    assert depth >= 12 and (depth - 12) % 2 == 0
    edges = [(s, r, q) for s in ('X', 'Z') for r, row in enumerate(code['supports'][s]) for q in row]
    lookup = {old: 16 + new for new, old in enumerate(selected)}
    fixed = {(g['type'], lookup[g['check']], g['data']): t + (depth-12 if t >= 6 else 0)
             for t, layer in enumerate(reference['layers']) for g in layer if g['check'] in lookup}
    variables = {e: z3.IntVal(fixed[e]) if e in fixed else z3.Int(f't{i}') for i, e in enumerate(edges)}
    sat = z3.Solver()
    sat.set(timeout=milliseconds, random_seed=seed)
    for e, v in variables.items():
        if e not in fixed:
            sat.add(v >= 0, v < depth)
    for s in ('X', 'Z'):
        for r, row in enumerate(code['supports'][s]):
            sat.add(z3.Distinct([variables[s, r, q] for q in row]))
    for q in range(64):
        sat.add(z3.Distinct([v for e, v in variables.items() if e[2] == q]))
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            overlap = set(xs) & set(zs)
            if overlap:
                sat.add(z3.Sum([z3.If(variables['X', xr, q] < variables['Z', zr, q], 1, 0)
                                for q in overlap]) % 2 == 0)
    start = time.monotonic()
    status = sat.check()
    info = dict(status=str(status), seconds=time.monotonic()-start, depth=depth,
                fixed_edges=len(fixed), free_edges=len(edges)-len(fixed),
                reason=sat.reason_unknown() if status == z3.unknown else None)
    if status != z3.sat:
        return None, info
    model = sat.model()
    times = {e: model.eval(v).as_long() for e, v in variables.items()}
    schedule = materialize(code, times, depth, 'weight8_insertion_fixed_weight12')
    schedule.update(source_schedule_id=reference['schedule_id'], retained_original_indices=selected,
                    seed=seed, inserted_empty_middle_layers=depth-12)
    return schedule, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--original-count', type=int, default=8, choices=[8, 12, 16])
    p.add_argument('--depths', type=int, nargs='+', default=[12, 14, 16])
    p.add_argument('--solver-ms', type=int, default=10000)
    p.add_argument('--heuristic-seconds', type=float, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', attempts=[], schedules=[],
                  parameters={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_sha256={str(Path(__file__).resolve()): binary.digest(Path(__file__).read_bytes())},
                  contract=dict(rounds=3, noise='full_wait', memory_bases=['X','Z'], observables=16))
    sources = sorted(OLD.glob('redundant_*'))
    for trial in range(args.count):
        seed = 20260914 + trial
        source, reference = source_schedule(sources[trial % len(sources)])
        code, selected = seeded_basis(source, seed, args.original_count)
        for depth in args.depths:
            schedule, info = solve(code, selected, reference, depth, seed, args.solver_ms)
            report['attempts'].append(dict(trial=trial, selected=selected, source=reference['schedule_id'], **info))
            print(json.dumps(dict(stage='seeded_solver', trial=trial, **info)), flush=True)
            save(args.output/'report.json', report)
            if schedule is not None:
                report['schedules'].append(assess(code, schedule, args.output/schedule['schedule_id'], args.heuristic_seconds))
                save(args.output/'report.json', report)
                break
    report['status'] = 'complete_bounded_screen'
    save(args.output/'report.json', report)


if __name__ == '__main__':
    main()
