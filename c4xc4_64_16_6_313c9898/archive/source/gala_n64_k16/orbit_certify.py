"""Exact low-fault follow-up with independent physical witness reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import stim

from orbit_schedule import binary, faults
from orbit_campaign import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--through-four', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', maximum_faults=4 if args.through_four else 3, cases=[])
    for source in args.folder:
        schedule = json.loads((source/'schedule.json').read_text())
        for basis in ('X', 'Z'):
            folder = args.output/(schedule['schedule_id']+'_'+basis)
            folder.mkdir()
            path = source/f'{basis}.stim'
            c = stim.Circuit.from_file(path)
            preflight = faults.estimate_four_fault_table(c)
            save(folder/'preflight.json', preflight)
            def progress(event):
                save(folder/'progress.json', event)
                print(json.dumps(dict(schedule=schedule['schedule_id'], basis=basis, **event)), flush=True)
            print(json.dumps(dict(stage='exact_start', schedule=schedule['schedule_id'], basis=basis, **preflight)), flush=True)
            start = time.monotonic()
            fn = faults.certify_through_four if args.through_four else faults.certify_through_three
            exact = fn(c, heartbeat_seconds=20, progress=progress)
            witness = None
            if exact['witness']:
                errors = [faults.explain_masks(c, e['detector_mask'], e['observable_mask']) for e in exact['witness']['effects']]
                witness = faults.replay(c, errors)
                (folder/'witness.stim').write_text(witness.pop('faulted_circuit'))
                save(folder/'physical_witness.json', witness)
            item = dict(source=str(source), basis=basis, schedule_id=schedule['schedule_id'],
                        circuit_sha256=binary.digest(str(c).encode()), exact=exact,
                        upper_bound=None if witness is None else witness['fault_count'],
                        wall_seconds=time.monotonic()-start)
            save(folder/'certificate.json', item)
            report['cases'].append(item)
            save(args.output/'report.json', report)
            print(json.dumps(dict(stage='exact_complete', schedule=schedule['schedule_id'], basis=basis,
                                  lower=exact['lower_bound'], upper=item['upper_bound'])), flush=True)
    report['status'] = 'complete'
    report['source_sha256'] = faults.inventory()['source_sha256']
    report['driver_sha256'] = binary.digest(Path(__file__).read_bytes())
    save(args.output/'report.json', report)


if __name__ == '__main__':
    main()
