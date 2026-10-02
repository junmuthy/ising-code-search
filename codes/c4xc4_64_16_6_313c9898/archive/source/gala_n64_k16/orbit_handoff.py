"""Bind final resource summaries to static and circuit certificates."""
import argparse
import json
from pathlib import Path

import stim

from orbit_campaign import save
from orbit_schedule import HERE, binary, circuits, faults, load_module, old_schedule
from sparse_core import permute, save_certificate, space_key

SHORTLIST = [
    'bc5e4e6336735d0eb4bfd1cb7059314667efaf33b3491f574c30cf5f9a5f3e5e',
    '1fad29e8c9bc9e4f3d381f850ec4665de20b46b6aa41ecc677a4cd6b2a1b196d',
    '313c98984982b039f1068bfd6001a4ffceb370b2091c3edbc9d3f8b915aaa028',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = load_module('orbit_handoff_replay', HERE/'fault_distance_milestone1/audit.py',
                        {'model':binary, 'circuit':circuits, 'schedule':old_schedule, 'faults':faults})
    finalists = json.loads((args.root/'all_finalists_v1/report.json').read_text())['candidates']
    report = dict(status='running', shortlist=[next(r for r in finalists if r['id']==key) for key in SHORTLIST], circuits=[])
    # The third code's Z labels already have weight six. Apply the known H
    # fold with its inverse logical permutation to obtain matching X labels.
    folder = args.root/'all_finalists_v1'/SHORTLIST[2]/'independent'
    data = json.loads((folder/'candidate.json').read_text())
    static = json.loads((folder/'audit.json').read_text())
    pi = static['actions']['H']['logical_permutation']
    x = [permute(data['certificate']['z'][pi.index(j)], data['certificate']['fold']) for j in range(16)]
    assert all(v.bit_count()==6 for v in x+data['certificate']['z'])
    modified = dict(data, certificate=dict(data['certificate'], x=x))
    save_certificate(modified, args.output/'alternate_w12_all_weight6_logicals')
    report['alternate_logical_basis'] = dict(code=SHORTLIST[2], x_weights=[6]*16, z_weights=[6]*16,
        path=str(args.output/'alternate_w12_all_weight6_logicals'),
        note='Same logical labels and operations; only X representatives changed modulo stabilizers. Full independent audit rerun.')
    for certificate_root in ('exact_three_v1', 'exact_four_bc5e_v1', 'exact_three_restricted_v1', 'exact_four_restricted_v1'):
        source_report = json.loads((args.root/certificate_root/'report.json').read_text())
        assert source_report['status']=='complete'
        for item in source_report['cases']:
            source = Path(item['source'])
            basis = item['basis']
            code = json.loads((source/'code.json').read_text())
            schedule = json.loads((source/'schedule.json').read_text())
            c = stim.Circuit.from_file(source/f'{basis}.stim')
            old_schedule.verify(code, schedule)
            assert c == circuits.build(code,schedule,basis,noise='full_wait',rounds=3)
            assert binary.digest(str(c).encode()) == item['circuit_sha256']
            key = space_key(code['checks']['X'], code['checks']['Z'])
            static = json.loads((args.root/'all_finalists_v1'/key/'independent/audit.json').read_text())
            side = 'Z' if basis=='X' else 'X'
            support = static['distance'][side]['weight_six_witness']
            assert support and len(support)==6
            gate = side+'_ERROR'
            offset = max(i for i,inst in enumerate(c) if inst.name==gate and [t.value for t in inst.targets_copy()]==list(range(64)))
            om = sum((len(set(support)&set(row))%2)<<j for j,row in enumerate(code['logical_supports'][basis]))
            witness = dict(circuit_sha256=item['circuit_sha256'], fault_count=6, detector_mask=0, observable_mask=om,
                faults=[dict(instruction_offset=offset,gate=gate,target_range=[q,q+1],paulis=[[q,side]]) for q in support])
            replayed = audit.replay_saved(c,witness,schedule['depth'])
            name = certificate_root+'_'+item['schedule_id']+'_'+basis
            save(args.output/(name+'_six_fault_witness.json'), witness)
            exact = item['exact']
            expected = [(w,'unsat') for w in range(1,exact['lower_bound'])]
            observed = [(attempt['fault_count'],attempt['status']) for attempt in exact['attempts']]
            assert observed[:len(expected)] == expected
            upper = 6
            if item['upper_bound'] is not None:
                assert observed[-1][1]=='sat'
                saved = json.loads((args.root/certificate_root/(item['schedule_id']+'_'+basis)/'physical_witness.json').read_text())
                audit.replay_saved(c,saved,schedule['depth'])
                upper = saved['fault_count']
            report['circuits'].append(dict(source=str(source), basis=basis, schedule_id=item['schedule_id'],
                 certificate_root=certificate_root, lower_bound=exact['lower_bound'],upper_bound=upper,
                 cnots=schedule['validation']['cnots'], depth=schedule['depth'],qubits=c.num_qubits,
                 six_fault_upper_replay=replayed,circuit_sha256=item['circuit_sha256']))
    report['status']='complete_audited_handoff'
    report['source_sha256']=binary.digest(Path(__file__).read_bytes())
    save(args.output/'report.json', report)
    print(json.dumps(dict(status=report['status'],circuits=report['circuits'],alternate=report['alternate_logical_basis'])),flush=True)


if __name__=='__main__':
    main()
