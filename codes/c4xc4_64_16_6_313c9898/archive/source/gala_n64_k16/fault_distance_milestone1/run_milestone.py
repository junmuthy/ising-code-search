"""Bounded first milestone: eight schedules, then one detailed memory audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from model import ROOT, digest, load, save
from schedule import verify
from circuit import build
from faults import (bounded_heuristic, certify_through_three, explain_masks,
                    estimate_four_fault_table, inventory, replay, static_upper)


def assess(code, schedule, folder, *, basis, experiment, noise, heuristic_seconds):
    folder.mkdir(parents=True, exist_ok=False)
    circuit = build(code, schedule, basis, experiment=experiment, noise=noise)
    path = folder / 'circuit.stim'
    circuit.to_file(path)
    ideal = circuit.without_noise()
    ideal.detector_error_model(allow_gauge_detectors=False)
    d, o = ideal.compile_detector_sampler(seed=42).sample(128, separate_observables=True)
    assert not np.any(d) and not np.any(o)
    configuration = dict(schedule_id=schedule['schedule_id'], basis=basis,
                         experiment=experiment, rounds=3, noise=noise, probability=0.001,
                         circuit_sha256=digest(str(circuit).encode()))
    def progress(event):
        save(folder / 'progress.json', dict(configuration=configuration, stage=event))
        print(json.dumps(dict(case=str(folder.name), **event)), flush=True)
    print(json.dumps(dict(stage='exact_through_three_start', **configuration)), flush=True)
    exact = certify_through_three(circuit, heartbeat_seconds=20, progress=progress)
    save(folder / 'exact_through_three.json', exact)
    upper, witness = None, None
    if exact['witness'] is not None:
        errors = [explain_masks(circuit, e['detector_mask'], e['observable_mask']) for e in exact['witness']['effects']]
        witness = replay(circuit, errors)
        upper = witness['fault_count']
        heuristic = dict(status='not_needed_exact_witness', upper_bound=upper)
    else:
        heuristic = bounded_heuristic(path, folder / 'heuristic.json', heuristic_seconds)
        if heuristic.get('witness'):
            witness = heuristic['witness']
            upper = witness['fault_count']
        else:
            location = 'memory' if experiment == 'memory' and noise in ('full', 'full_wait') else 'bulk'
            witness = static_upper(circuit, code, basis, location)
            upper = witness['fault_count']
            witness['source'] = 'saved_static_weight_eight_logical_at_permitted_fault_locations'
    if 'faulted_circuit' in witness:
        (folder / 'witness.stim').write_text(witness.pop('faulted_circuit'))
    save(folder / 'physical_witness.json', witness)
    assert upper >= exact['lower_bound']
    estimate = estimate_four_fault_table(circuit)
    record = dict(configuration=configuration, status='bounds_verified',
                  lower_bound=exact['lower_bound'], upper_bound=upper,
                  exact_distance=upper if upper == exact['lower_bound'] else None,
                  ideal_validation=dict(shots=128, zero_detectors=True, zero_observables=True,
                                        gauge_detectors_rejected=True),
                  counts=dict(qubits=circuit.num_qubits, measurements=circuit.num_measurements,
                              detectors=circuit.num_detectors, observables=circuit.num_observables),
                  heuristic=heuristic, four_fault_preflight=estimate)
    save(folder / 'result.json', record)
    print(json.dumps(dict(stage='bounds_complete', folder=str(folder), lower=record['lower_bound'],
                          upper=upper, effects=estimate['distinct_fault_effects'])), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--heuristic-seconds', type=float, default=12)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    code = load()
    source = json.loads((ROOT / 'schedules_v1/report.json').read_text())
    provenance = inventory()
    report = dict(status='running', scope='first milestone, exact lower searches through three; no five-fault search',
                  provenance=provenance, input_provenance=code['provenance'], screen=[], cases={})
    started = time.monotonic()
    selected = None
    for item in source['schedules']:
        schedule = json.loads((ROOT / 'schedules_v1' / item['file']).read_text())
        verify(code, schedule)
        result = assess(code, schedule, output / 'screen' / item['schedule_id'], basis='X',
                        experiment='bulk', noise='cnot', heuristic_seconds=args.heuristic_seconds)
        score = (result['lower_bound'], result['upper_bound'])
        report['screen'].append(dict(schedule_id=item['schedule_id'], lower=result['lower_bound'], upper=result['upper_bound']))
        if selected is None or score > selected[0]:
            selected = (score, schedule, result)
        save(output / 'report.json', report)
    _, schedule, _ = selected
    save(output / 'selected_schedule.json', schedule)
    report['selected_schedule_id'] = schedule['schedule_id']
    report['selection_rule'] = 'Largest exact bulk-X lower bound, then largest found upper bound; not a proof of optimal fault distance'
    for experiment, noise in (('bulk', 'cnot'), ('memory', 'cnot'), ('memory', 'full'), ('memory', 'full_wait')):
        for basis in ('X', 'Z'):
            name = f'{experiment}_{noise}_{basis}'
            report['cases'][name] = assess(code, schedule, output / name, basis=basis,
                                          experiment=experiment, noise=noise, heuristic_seconds=args.heuristic_seconds)
            save(output / 'report.json', report)
    # No source edits are permitted to silently change a running certificate.
    assert inventory() == provenance
    report.update(status='complete', elapsed_seconds=time.monotonic() - started)
    save(output / 'report.json', report)
    print(json.dumps(dict(status='complete', selected=report['selected_schedule_id'],
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
