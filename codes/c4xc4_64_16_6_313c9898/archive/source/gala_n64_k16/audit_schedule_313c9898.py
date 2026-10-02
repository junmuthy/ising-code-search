"""Independently reconstruct completed 313c9898 schedules and bind certificates."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import stim

from orbit_campaign import save
from orbit_schedule import HERE, binary, circuits, faults, load_module, old_schedule
from schedule_313c9898 import KEY, LOGICALS, pinned_data
from sparse_core import rank, same_space


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    replay = load_module('code313_independent_replay', HERE/'fault_distance_milestone1/audit.py',
                         {'model':binary,'circuit':circuits,'schedule':old_schedule,'faults':faults})
    pinned = pinned_data()
    report = dict(status='running', code=KEY, logical_basis_sha256=binary.digest(LOGICALS.read_bytes()),
                  screens=[], circuits=[], bounds=[], exact_certificates=[], unfinished=[])
    cases = {}
    for path in sorted(args.root.glob('*/report.json')):
        if path.parent == args.output:
            continue
        run = json.loads(path.read_text())
        if 'schedules' not in run and 'maximum_faults' not in run:
            continue
        if run['status'] not in ('complete','complete_bounded_screen'):
            report['unfinished'].append(str(path))
            continue
        if 'schedules' not in run:
            continue
        for source, digest in run.get('source_sha256', {}).items():
            assert binary.digest(Path(source).read_bytes()) == digest
        report['screens'].append(dict(path=str(path), schedules=len(run['schedules']),
                                      attempts=dict(Counter(a['status'] for a in run.get('attempts',[])))))
        for item in run['schedules']:
            folder = path.parent/item['schedule_id']
            code = json.loads((folder/'code.json').read_text())
            schedule = json.loads((folder/'schedule.json').read_text())
            old_schedule.verify(code,schedule)
            assert code['fold']==pinned['certificate']['fold']
            for s,k,l in [('X','hx','x'),('Z','hz','z')]:
                assert rank(code['checks'][s])==24 and same_space(code['checks'][s],pinned[k])
                assert code['logicals'][s]==pinned['certificate'][l]
                assert code['logical_supports'][s]==binary.supports(code['logicals'][s])
                assert code['supports'][s]==binary.supports(code['checks'][s])
                assert code['relations'][s]==binary.relations(code['checks'][s])
            for basis in ('X','Z'):
                c = stim.Circuit.from_file(folder/f'{basis}.stim')
                assert c==circuits.build(code,schedule,basis,rounds=3,noise='full_wait')
                digest = binary.digest(str(c).encode())
                assert digest==item['cases'][basis]['circuit_sha256']
                ideal=c.without_noise()
                ideal.detector_error_model(allow_gauge_detectors=False)
                d,o=ideal.compile_detector_sampler(seed=313).sample(32,separate_observables=True)
                assert not np.any(d) and not np.any(o)
                side='Z' if basis=='X' else 'X'
                support=code['logical_supports'][side][0]
                assert len(support)==6
                gate=side+'_ERROR'
                offset=max(i for i,inst in enumerate(c) if inst.name==gate and [t.value for t in inst.targets_copy()]==list(range(64)))
                witness=dict(circuit_sha256=digest,fault_count=6,detector_mask=0,observable_mask=1,
                             faults=[dict(instruction_offset=offset,gate=gate,target_range=[q,q+1],paulis=[[q,side]]) for q in support])
                replay.replay_saved(c,witness,schedule['depth'])
                name=path.parent.name+'_'+item['schedule_id']+'_'+basis
                save(args.output/(name+'_six_fault_witness.json'),witness)
                upper=6
                heuristic=item['cases'][basis]['heuristic']
                if heuristic.get('witness'):
                    physical=replay.replay_saved(c,heuristic['witness'],schedule['depth'])
                    assert physical['fault_count']==heuristic['upper_bound']
                    upper=min(upper,physical['fault_count'])
                case=dict(source=str(folder),schedule_id=item['schedule_id'],basis=basis,circuit_sha256=digest,
                          cnots=schedule['validation']['cnots'],depth=schedule['depth'],qubits=c.num_qubits,
                          checks_per_type={s:len(rows) for s,rows in code['checks'].items()},
                          check_weights={s:dict(Counter(w.bit_count() for w in rows)) for s,rows in code['checks'].items()},
                          lower_bound=1,upper_bound=upper,certificates=[],six_fault_witness=str(args.output/(name+'_six_fault_witness.json')))
                cases[str(folder),basis]=case
                report['circuits'].append(dict(source=str(folder),basis=basis,sha256=digest,rebuilt=True,ideal_verified=True))
    for path in sorted(args.root.glob('*/report.json')):
        if path.parent == args.output:
            continue
        run=json.loads(path.read_text())
        if run.get('status')!='complete' or 'maximum_faults' not in run:
            continue
        for item in run['cases']:
            case=cases[item['source'],item['basis']]
            assert case['circuit_sha256']==item['circuit_sha256']
            exact=item['exact']
            lower=exact['lower_bound']
            expected=[(w,'unsat') for w in range(1,lower)]
            attempts=[(a['fault_count'],a['status']) for a in exact['attempts']]
            assert attempts[:len(expected)]==expected
            if item['upper_bound'] is not None:
                c=stim.Circuit.from_file(Path(item['source'])/f'{item["basis"]}.stim')
                wpath=path.parent/(item['schedule_id']+'_'+item['basis'])/'physical_witness.json'
                witness=json.loads(wpath.read_text())
                physical=replay.replay_saved(c,witness,case['depth'])
                assert physical['fault_count']==item['upper_bound']
                case['upper_bound']=min(case['upper_bound'],item['upper_bound'])
            case['lower_bound']=max(case['lower_bound'],lower)
            assert case['lower_bound']<=case['upper_bound']
            case['certificates'].append(str(path))
            report['exact_certificates'].append(dict(source=str(path),schedule=item['schedule_id'],basis=item['basis'],lower=lower,upper=item['upper_bound']))
    report['bounds']=list(cases.values())
    report['status']='complete_independent_audit' if not report['unfinished'] else 'partial_independent_audit'
    report['source_sha256']={str(Path(__file__).resolve()):binary.digest(Path(__file__).read_bytes())}
    save(args.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],circuits=len(report['circuits']),certificates=len(report['exact_certificates']),unfinished=report['unfinished'])),flush=True)


if __name__=='__main__':
    main()
