"""Reconstruct all orbit trials and independently audit exported code/circuit data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from orbit_campaign import save
from orbit_extension import module_generators, orbit
from sparse_core import HERE, basis, grid_module, independent_audit, permute, rank, reduce, same_space, space_key, unpack
from check_weight_reduction import enumerate_space
from orbit_schedule import binary, circuits, load_module, old_schedule


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_trials(root):
    totals, runs = Counter(), []
    for summary_path in sorted(root.glob('20260914_*sheet*_v*/summary.json')) + sorted(root.glob('20260914_nonabelian_*/summary.json')):
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text())
        assert summary['status'] == 'completed_bounded_pilot'
        sources = {}
        for name, expected in summary['source_sha256'].items():
            path = Path(name)
            if fingerprint(path) != expected:
                path = root/'source_v1'/path.name
            assert fingerprint(path) == expected
            sources[name] = str(path)
        geometries = json.loads((folder/'geometry.json').read_text())
        records = [json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
        assert len(records) == summary['completed_trials']
        for index, record in enumerate(records):
            assert record['trial'] == index
            if 'hx' not in record:
                assert record['status'] == 'no_completion_sampled'
                continue
            geo = geometries[record['variant']]
            fold = geo['folds'][record['fold_index']]
            hx = list(dict.fromkeys(v for seed in record['seeds'] for v in orbit(seed, geo['actions'])))
            hz = [permute(v, fold) for v in hx]
            assert hx == record['hx'] and hz == record['hz']
            assert rank(hx) == rank(hz) == 24
            assert max(map(int.bit_count, hx+hz)) <= summary['parameters']['maxweight']
            ax, az = unpack(hx, 64), unpack(hz, 64)
            assert not np.any(ax @ az.T % 2)
            assert space_key(hx, hz) == record['row_space_id']
            assert module_generators(hx, geo['generators']) == record['module_generator_count']
            status = record['status']
            if status == 'short_logical':
                support = record['witness']
                assert 1 <= len(support) <= 5 and len(set(support)) == len(support)
                word = sum(1 << q for q in support)
                checks, stabilizers = (hx, hz) if record['sector'] == 'Z' else (hz, hx)
                assert all((word & v).bit_count() % 2 == 0 for v in checks)
                assert reduce(word, basis(stabilizers))
            elif status == 'cyclic_not_new_direction':
                assert record['module_generator_count'] == 1
            elif status in ('norm_zero', 'wrong_logical_relations'):
                assert grid_module(hx, hz, geo['px'], geo['py'], geo['central'])[0]['status'] == status
            # Positive exports receive the full independent distance/gate audit
            # below. Fold-only negatives remain scoped to the saved fold.
            totals[status] += 1
            totals['reconstructed_rank24_trials'] += 1
        runs.append(dict(path=str(folder), completed_trials=len(records), verified_sources=sources))
        print(json.dumps(dict(stage='trial_audit', folder=folder.name, trials=len(records))), flush=True)
    return dict(totals=dict(totals), runs=runs)


def audit_finalists(root):
    report = json.loads((root/'all_finalists_v1/report.json').read_text())
    records = []
    # Full re-audit of the two weight-ten codes and the best weight-twelve
    # independent-support option; other hits keep their existing export audits.
    chosen = [r for r in report['candidates'] if r['minimum_maximum_check_weight'] < 12]
    chosen.append(next(r for r in report['candidates'] if r['id'].startswith('313c9898')))
    for record in chosen:
        folder = root/'all_finalists_v1'/record['id']
        original = json.loads((folder/'original/candidate.json').read_text())
        independent = json.loads((folder/'independent/candidate.json').read_text())
        audit, _ = independent_audit(independent)
        assert audit['metrics']['row_space_component_sizes'] == [64]
        saved = json.loads((folder/'weight_enumeration.json').read_text())
        for key in ('hx', 'hz'):
            assert same_space(original[key], independent[key])
            recalculated = enumerate_space(independent[key], ceiling=12)
            for name in ('weight_histogram', 'rank_filtration', 'minimum_basis_weight_counts',
                         'minimum_basis_total_weight', 'minimum_possible_maximum_check_weight'):
                assert json.loads(json.dumps(recalculated[name])) == saved[key][name]
        records.append(dict(id=record['id'], status='independent_distance_gates_weights_verified',
                            audit=audit, candidate_sha256=fingerprint(folder/'independent/candidate.json')))
        print(json.dumps(dict(stage='finalist_audit', id=record['id'])), flush=True)
    # These three have different stabilizer weight distributions. This proves
    # inequivalence by qubit permutation and CSS generator changes, not novelty
    # under arbitrary local Clifford transformations or relative to literature.
    histograms = [json.loads((root/'all_finalists_v1'/r['id']/'weight_enumeration.json').read_text())['hx']['weight_histogram'] for r in records]
    assert len({json.dumps(h, sort_keys=True) for h in histograms}) == 3
    return records


def audit_circuits(root):
    audit = load_module('orbit_independent_saved_replay', HERE/'fault_distance_milestone1/audit.py',
                        {'model':binary, 'circuit':circuits, 'schedule':old_schedule,
                         'faults':__import__('orbit_schedule').faults})
    records = []
    for path in sorted(root.glob('schedules_*/**/result.json')):
        folder = path.parent
        result = json.loads(path.read_text())
        code = json.loads((folder/'code.json').read_text())
        schedule = json.loads((folder/'schedule.json').read_text())
        old_schedule.verify(code, schedule)
        for basis, case in result['cases'].items():
            c = stim.Circuit.from_file(folder/f'{basis}.stim')
            assert c == circuits.build(code, schedule, basis, noise='full_wait', rounds=3)
            assert binary.digest(str(c).encode()) == case['circuit_sha256']
            entry = dict(folder=str(folder), basis=basis, circuit_rebuilt=True)
            if case['heuristic'].get('witness'):
                entry['replay'] = audit.replay_saved(c, case['heuristic']['witness'], schedule['depth'])
            else:
                assert case['upper_bound'] is None
                entry['status'] = 'no_upper_certificate_from_heuristic'
            records.append(entry)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    report = dict(trials=audit_trials(args.root), finalists=audit_finalists(args.root),
                  circuits=audit_circuits(args.root), status='passed',
                  source_sha256=fingerprint(Path(__file__)),
                  scope='All recorded multi-orbit completions reconstructed; three selected code audits and every saved circuit screen rebuilt. No family-level exclusion inferred.')
    save(args.output, report)
    print(json.dumps(dict(status='passed', finalist_count=len(report['finalists']), circuits=len(report['circuits']))), flush=True)


if __name__ == '__main__':
    main()
