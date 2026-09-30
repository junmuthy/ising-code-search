"""Bounded, resumable-by-case schedule search; exact certificates on finalists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from model import ROOT, SHARED, GALA, archived, digest, load, save
from circuit import build
from schedule import folded_solver, separated_schedule, verify
from faults import (bounded_heuristic, certify_through_three, certify_through_four,
                    estimate_four_fault_table, explain_masks, inventory, replay)


def log(record):
    print(json.dumps(record), flush=True)


def witness_from_exact(circuit, result):
    if result['witness'] is None:
        return None
    return replay(circuit, [explain_masks(circuit, e['detector_mask'], e['observable_mask'])
                            for e in result['witness']['effects']])


def screen(code, schedule, folder, seconds):
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'schedule.json', schedule)
    save(folder / 'code.json', code)
    result = dict(schedule_id=schedule['schedule_id'], basis_id=code['basis_id'],
                  presentation=code['presentation'], basis_seed=code['basis_seed'],
                  depth=schedule['depth'], cnots=schedule['validation']['cnots'], cases={})
    for basis in ('X', 'Z'):
        c = build(code, schedule, basis)
        ideal = c.without_noise()
        ideal.detector_error_model(allow_gauge_detectors=False)
        d, o = ideal.compile_detector_sampler(seed=71).sample(64, separate_observables=True)
        assert not np.any(d) and not np.any(o)
        path = folder / f'{basis}.stim'
        c.to_file(path)
        h = bounded_heuristic(path, folder / f'{basis}_heuristic.json', seconds)
        case = dict(heuristic=h, circuit_sha256=digest(str(c).encode()),
                    qubits=c.num_qubits, detectors=c.num_detectors,
                    ideal_validation=True, upper_bound=h.get('upper_bound'), lower_bound=1)
        result['cases'][basis] = case
        save(folder / 'result.json', result)
        log(dict(stage='heuristic', schedule=schedule['schedule_id'], basis=basis,
                 depth=schedule['depth'], presentation=code['presentation'], upper=case['upper_bound']))
        if case['upper_bound'] is not None and case['upper_bound'] < 5:
            result['status'] = 'rejected_by_replayed_witness'
            break
    else:
        result['status'] = 'heuristic_survivor_not_certified'
    save(folder / 'result.json', result)
    return result


def certify(folder, *, max_table_gib=8):
    import stim
    result = json.loads((folder / 'result.json').read_text())
    for basis in ('X', 'Z'):
        c = stim.Circuit.from_file(folder / f'{basis}.stim')
        estimate = estimate_four_fault_table(c)
        save(folder / f'{basis}_memory_preflight.json', estimate)
        assert estimate['table_bytes'] <= max_table_gib * 2**30
        def progress(event):
            save(folder / f'{basis}_progress.json', event)
            log(dict(schedule=result['schedule_id'], basis=basis, **event))
        log(dict(stage='exact_through_four_start', basis=basis, **estimate))
        exact = certify_through_four(c, heartbeat_seconds=20, progress=progress)
        exact['circuit_sha256'] = digest(str(c).encode())
        save(folder / f'{basis}_exact_through_four.json', exact)
        witness = witness_from_exact(c, exact)
        if witness:
            (folder / f'{basis}_exact_witness.stim').write_text(witness.pop('faulted_circuit'))
            save(folder / f'{basis}_exact_witness.json', witness)
            result['cases'][basis]['upper_bound'] = witness['fault_count']
        result['cases'][basis]['lower_bound'] = exact['lower_bound']
        save(folder / 'result.json', result)
        if exact['lower_bound'] < 5:
            result['status'] = 'rejected_by_exact_search'
            break
    else:
        result['status'] = 'certified_fault_distance_at_least_five_both_bases'
    save(folder / 'result.json', result)
    log(dict(stage='certification_complete', schedule=result['schedule_id'], status=result['status']))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-family', type=int, default=4)
    p.add_argument('--heuristic-seconds', type=float, default=4)
    p.add_argument('--solver-ms', type=int, default=3000)
    p.add_argument('--certify', type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = inventory()
    provenance['shared_source_sha256'] = {str(SHARED / f): digest((SHARED / f).read_bytes())
                                        for f in ('model.py', 'faults.py')}
    report = dict(status='running', contract=dict(rounds=3, noise='full_wait', bases=['X', 'Z'],
                  target_lower_bound=5, probability=0.001), provenance=provenance,
                  solver_attempts=[], schedules=[])
    for name in ('candidate.json', 'metadata.json', 'certificates/check_weight_enumeration.json',
                 'certificates/distance_audit.json'):
        path = args.output / 'inputs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archived(name))
    start = time.monotonic()
    def accept(code, schedule):
        path = args.output / schedule['schedule_id']
        if path.exists():
            return
        report['schedules'].append(screen(code, schedule, path, args.heuristic_seconds))
        save(args.output / 'report.json', report)
    # Sample the low-CNOT presentation first, with independent X/Z orderings.
    for i in range(args.per_family):
        code = load('independent', None if i == 0 else 20260911 + i)
        accept(code, separated_schedule(code, 991 + i))
    # Simultaneous paired-fold schedules: independent and redundant controls.
    for presentation in ('independent', 'redundant'):
        code = load(presentation)
        remaining = args.per_family
        for depth in (12, 14, 16, 24):
            next_record = folded_solver(code, depth, seed=29 + depth,
                                        timeout_ms=args.solver_ms, separated=depth == 24)
            for _ in range(remaining):
                schedule, info = next_record()
                report['solver_attempts'].append(dict(presentation=presentation, **info))
                log(dict(stage='scheduler', presentation=presentation, **info))
                save(args.output / 'report.json', report)
                if schedule is None:
                    break
                accept(code, schedule)
                remaining -= 1
            if remaining <= 0:
                break
    survivors = [r for r in report['schedules'] if r['status'] == 'heuristic_survivor_not_certified']
    survivors.sort(key=lambda r: (-min(c['upper_bound'] or 6 for c in r['cases'].values()), r['depth'], r['cnots']))
    report['certifications'] = []
    for r in survivors[:args.certify]:
        result = certify(args.output / r['schedule_id'])
        report['certifications'].append(result)
        save(args.output / 'report.json', report)
        if result['status'] == 'certified_fault_distance_at_least_five_both_bases':
            report['selected_schedule_id'] = result['schedule_id']
            break
    assert inventory() == {k: v for k, v in provenance.items() if k != 'shared_source_sha256'}
    for file, value in provenance['shared_source_sha256'].items():
        assert digest(Path(file).read_bytes()) == value
    report.update(status='complete', elapsed_seconds=time.monotonic() - start,
                  target_achieved='selected_schedule_id' in report)
    save(args.output / 'report.json', report)
    log(dict(status='complete', elapsed_seconds=report['elapsed_seconds'], target_achieved=report['target_achieved']))


if __name__ == '__main__':
    main()
