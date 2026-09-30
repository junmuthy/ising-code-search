"""Construct 448-CNOT schedules from two compatible clean subcircuits."""
import argparse
import json
from pathlib import Path

from orbit_campaign import save
from orbit_schedule import assess, binary, materialize, solver
from schedule_313c9898 import OLD, source_schedule
from schedule_313c9898_seeded import seeded_basis
from sparse_core import permute


def small_block(code, seed):
    checks={s:rows[:16] for s,rows in code['checks'].items()}
    partial=dict(code,checks=checks,supports={s:binary.supports(rows) for s,rows in checks.items()})
    seedword=checks['X'][0]
    index={w:i for i,w in enumerate(checks['X'])}
    points=binary.supports([seedword])[0]
    seen=set()
    todo=[list(range(64))]
    edges={}
    while todo:
        p=todo.pop()
        word=permute(seedword,p)
        if word in seen:
            continue
        seen.add(word)
        for j,q in enumerate(points):
            edges[index[word],p[q]]=j
        for action in (code['data']['px'],code['data']['py']):
            todo.append([action[q] for q in p])
    assert seen==set(checks['X'])
    partial['edge_map']=edges
    schedule,info=solver(partial,8,seed=seed,milliseconds=5000)()
    if schedule is None:
        raise RuntimeError(info)
    return schedule,info


def staged(code, selected, reference, small, small_first=True, compact=True):
    lookup={old:16+i for i,old in enumerate(selected)}
    large=[[dict(g,check=lookup[g['check']]) for g in layer if g['check'] in lookup]
           for layer in reference['layers']]
    layers=small['layers']+large if small_first else large+small['layers']
    # ASAP preserves the total order of gates on EACH data and ancilla wire.
    # All reordered gates therefore commute as disjoint operations.
    last_data={}
    last_anc={}
    times={}
    for t,layer in enumerate(layers):
        for g in layer:
            anc=g['type'],g['check']
            q=g['data']
            tick=max(last_data.get(q,-1),last_anc.get(anc,-1))+1 if compact else t
            last_data[q]=last_anc[anc]=tick
            times[g['type'],g['check'],q]=tick
    result=materialize(code,times,max(times.values())+1,'two_clean_blocks_asap' if compact else 'two_clean_blocks')
    result.update(source_schedule_id=reference['schedule_id'],small_schedule_id=small['schedule_id'],
                  retained_original_indices=selected,small_first=small_first,asap_compacted=compact)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--heuristic-seconds',type=float,default=6)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',schedules=[],attempts=[],
                parameters=dict(count=args.count,heuristic_seconds=args.heuristic_seconds),
                source_sha256={str(Path(__file__).resolve()):binary.digest(Path(__file__).read_bytes())},
                contract=dict(rounds=3,noise='full_wait',memory_bases=['X','Z'],observables=16))
    sources=sorted(OLD.glob('redundant_*'))
    for trial in range(args.count):
        seed=20260914+trial//2
        source,reference=source_schedule(sources[(trial//2)%len(sources)])
        code,selected=seeded_basis(source,seed)
        small,info=small_block(code,seed)
        schedule=staged(code,selected,reference,small,small_first=trial%2==0)
        report['attempts'].append(dict(trial=trial,seed=seed,small_block_solver=info,status='sat'))
        report['schedules'].append(assess(code,schedule,args.output/schedule['schedule_id'],args.heuristic_seconds))
        save(args.output/'report.json',report)
    report['status']='complete_bounded_screen'
    save(args.output/'report.json',report)


if __name__=='__main__':
    main()
