"""Checkpointed, bounded first milestone outside the old cyclic family."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from orbit_extension import construct, evaluate, geometry
from sparse_core import HERE, save_certificate, space_key


def save(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=('two_sheet', 'four_sheet', 'nonabelian'), required=True)
    p.add_argument('--maxweight', type=int, required=True)
    p.add_argument('--trials', type=int, default=128)
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--proposals', type=int, default=48)
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
               [Path(__file__).resolve(), HERE/'orbit_extension.py', HERE/'sparse_core.py',
                HERE/'sparse_models.py', HERE/'sparse_twisted.py',
                HERE.parent/'gala_n128_k32/algebra.py', HERE.parent/'gala_n128_k32/search.py']}
    geometries = [geometry(args.family, i) for i in range(4 if args.family == 'nonabelian' else 1)]
    save(args.output/'geometry.json', geometries)
    rng = random.Random(args.seed)
    stats, seen, module_counts = Counter(), set(), Counter()
    start = time.monotonic()
    report = dict(status='running', parameters={**vars(args), 'output': str(args.output)},
                  source_sha256=sources, accepted=[], classification='bounded heuristic construction; exact acceptance',
                  completed_trials=0)
    def checkpoint(announce=False):
        report.update(counts=dict(stats), module_generator_counts=dict(module_counts),
                      elapsed_seconds=time.monotonic()-start, distinct_code_fold_pairs=len(seen))
        save(args.output/'summary.json', report)
        if announce:
            print(json.dumps({k: report[k] for k in ('status', 'completed_trials', 'counts', 'elapsed_seconds')}), flush=True)
    checkpoint(True)
    with (args.output/'trials.jsonl').open('x') as journal:
        for trial in range(args.trials):
            if time.monotonic()-start >= args.seconds:
                report['stop_reason'] = 'wall_time_cap_between_trials'
                break
            geo = geometries[trial % len(geometries)]
            fold_index = (trial // len(geometries)) % len(geo['folds'])
            fold = geo['folds'][fold_index]
            candidate = construct(geo, fold, rng, args.maxweight, stats, proposals=args.proposals)
            row = dict(trial=trial, variant=geo['variant'], fold_index=fold_index)
            if candidate:
                key = space_key(candidate['hx'], candidate['hz'])
                module_counts[candidate['module_generator_count']] += 1
                row.update(candidate, row_space_id=key)
                if (key, tuple(fold)) in seen:
                    row['status'] = 'duplicate_code_fold'
                elif args.family == 'two_sheet' and candidate['module_generator_count'] == 1:
                    row['status'] = 'cyclic_not_new_direction'
                else:
                    seen.add((key, tuple(fold)))
                    info, certificate = evaluate(candidate, geo, fold)
                    row.update(info)
                    if certificate:
                        data = dict(candidate, n=64, k=16, px=geo['px'], py=geo['py'], central=geo['central'],
                                    certificate=certificate, logical_shape=[4, 4],
                                    construction='multi_orbit_' + args.family, max_check_weight=args.maxweight)
                        destination = args.output/'certified'/key
                        if not destination.exists():
                            save_certificate(data, destination)
                        report['accepted'].append(dict(row_space_id=key, path=str(destination)))
                stats[row['status']] += 1
            else:
                row['status'] = 'no_completion_sampled'
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            report['completed_trials'] = trial + 1
            checkpoint((trial + 1) % 32 == 0)
        else:
            report['stop_reason'] = 'configured_trial_count'
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in sources.items())
    report['status'] = 'completed_bounded_pilot'
    checkpoint(True)


if __name__ == '__main__':
    main()
