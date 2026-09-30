"""Remove duplicate measured checks from a valid full-orbit schedule."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from orbit_schedule import assess, materialize, prepare
from orbit_campaign import save
from sparse_core import permute, rank, same_space


def restrict(code, schedule, seed):
    rng = random.Random(seed)
    copies = defaultdict(list)
    for i, word in enumerate(code['checks']['X']):
        copies[word].append(i)
    chosen = sorted(rng.choice(indices) for indices in copies.values())
    hx = [code['checks']['X'][i] for i in chosen]
    hz = [permute(v, code['fold']) for v in hx]
    assert len(hx) == rank(hx) == 24
    assert same_space(hx, code['checks']['X']) and same_space(hz, code['checks']['Z'])
    assert all(code['checks']['Z'][i] == hz[j] for j, i in enumerate(chosen))
    new = prepare(dict(code['data'], hx=hx, hz=hz), 'independent')
    lookup = {old: new for new, old in enumerate(chosen)}
    times = {}
    for t, layer in enumerate(schedule['layers']):
        for g in layer:
            if g['check'] in lookup:
                times[g['type'], lookup[g['check']], g['data']] = t
    result = materialize(new, times, schedule['depth'], f'restrict_{schedule["schedule_id"]}_seed_{seed}')
    result['source_schedule_id'] = schedule['schedule_id']
    result['retained_check_indices'] = chosen
    return new, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = json.loads((args.source/'code.json').read_text())
    schedule = json.loads((args.source/'schedule.json').read_text())
    report = dict(status='running', source=str(args.source), schedules=[])
    for seed in range(20260914, 20260914 + args.count):
        new, record = restrict(code, schedule, seed)
        report['schedules'].append(assess(new, record, args.output/record['schedule_id'], 4))
        save(args.output/'report.json', report)
    report['status'] = 'complete_bounded_screen'
    save(args.output/'report.json', report)


if __name__ == '__main__':
    main()
