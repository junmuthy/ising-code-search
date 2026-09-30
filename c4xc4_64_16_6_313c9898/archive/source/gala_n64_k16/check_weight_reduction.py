"""Exhaustive stabilizer weights and equivalent low-cost check presentations.

The code space and saved logical representatives/permutations never change.
The exact weight optimization is separate from optional heuristic load
balancing. All output is written to a fresh directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gala_n128_k32'))
from algebra import basis, combine, permute, power, rank, reduce, same_space, unpack

SOURCE=HERE/'certified/64_16_6_c4xc4/candidate.json'


def enumerate_space(rows, ceiling=14, chunk=32):
    independent=list(basis(rows).values())
    half=len(independent)//2
    arrays=[]
    for generators in (independent[:half],independent[half:]):
        values=np.zeros(1,dtype=np.uint64)
        for row in generators:
            values=np.concatenate((values,values^np.uint64(row)))
        arrays.append(values)
    histogram=np.zeros(65,dtype=np.int64)
    low=[]
    for start in range(0,len(arrays[1]),chunk):
        words=arrays[1][start:start+chunk,None]^arrays[0][None,:]
        weights=np.bitwise_count(words)
        histogram += np.bincount(weights.ravel(),minlength=65)
        low.extend(map(int,words[(weights>0)&(weights<=ceiling)]))
    low.sort(key=lambda v:(v.bit_count(),v))
    assert int(histogram.sum())==2**len(independent)
    assert len(set(low))==len(low)
    filtration={w:rank([v for v in low if v.bit_count()<=w]) for w in range(ceiling+1)}
    minimum_counts={w:filtration[w]-filtration[w-1] for w in range(1,ceiling+1)
                    if filtration[w]>filtration[w-1]}
    assert filtration[ceiling]==len(independent)
    return {'dimension':len(independent),'enumerated':int(histogram.sum()),
            'weight_histogram':{w:int(v) for w,v in enumerate(histogram) if v},
            'rank_filtration':filtration,'minimum_basis_weight_counts':minimum_counts,
            'minimum_basis_total_weight':sum(w*c for w,c in minimum_counts.items()),
            'minimum_possible_maximum_check_weight':max(minimum_counts),
            'enumeration_basis':independent,'low_weight_rows':low}


def greedy_basis(words, order):
    pivots={}
    selected=[]
    for i in order:
        row=words[int(i)]
        while row:
            pivot=row.bit_length()-1
            if pivot not in pivots:
                pivots[pivot]=row
                selected.append(int(i))
                break
            row ^= pivots[pivot]
        if len(selected)==24: break
    assert len(selected)==24
    return selected


def load_score(selected, mx, mz):
    dx=mx[selected].sum(axis=0)
    dz=mz[selected].sum(axis=0)
    total=dx+dz
    return int(total.max()),int(max(dx.max(),dz.max())),int(total@total)


def balance(words,fold,trials,seed,milp_seconds):
    mx=unpack(words,64)
    mz=unpack([permute(v,fold) for v in words],64)
    rng=np.random.default_rng(seed)
    weight_classes=[[i for i,v in enumerate(words) if v.bit_count()==w]
                    for w in sorted({v.bit_count() for v in words})]
    best=None
    for trial in range(max(1,trials)):
        order=np.concatenate([rng.permutation(c) for c in weight_classes])
        selected=greedy_basis(words,order)
        score=load_score(selected,mx,mz)
        if best is None or score<best[0]: best=(score,selected)
    history=[]
    if milp_seconds>0:
        from scipy.optimize import Bounds, LinearConstraint, milp
        started=time.monotonic()
        count=len(words)
        costs=(mx+mz).T.astype(float)
        low=np.asarray([v.bit_count()==12 for v in words],dtype=float)
        # These counts characterize every minimum-total-weight basis.
        constraints=[np.ones(count),low]
        lower=[24.,18.]
        upper=[24.,18.]
        cuts=[]
        for target in range(10,best[0][0]):
            while time.monotonic()-started<milp_seconds:
                matrix=np.vstack([*constraints,*costs,*[c[0] for c in cuts]])
                lb=np.asarray([*lower,*([-np.inf]*64),*([-np.inf]*len(cuts))])
                ub=np.asarray([*upper,*([target]*64),*[c[1] for c in cuts]])
                result=milp(np.zeros(count),integrality=np.ones(count),
                            bounds=Bounds(np.zeros(count),np.ones(count)),
                            constraints=LinearConstraint(matrix,lb,ub),
                            options={'time_limit':max(.1,min(10.,milp_seconds-(time.monotonic()-started))),
                                     'mip_rel_gap':0})
                record={'target_combined_degree':target,'solver_status':int(result.status),
                        'message':result.message,'rank_cuts':len(cuts)}
                if result.x is None:
                    history.append(record)
                    # Numerical infeasibility is not advertised as an exact proof.
                    break
                selected=np.flatnonzero(result.x>.5).tolist()
                if len(selected)!=24 or sum(words[i].bit_count()==12 for i in selected)!=18:
                    history.append({**record,'rejected':'invalid rounded cardinalities'})
                    break
                pivots=basis([words[i] for i in selected])
                record['selected_rank']=len(pivots)
                history.append(record)
                if len(pivots)==24:
                    score=load_score(selected,mx,mz)
                    if score<best[0]: best=(score,selected)
                    print(json.dumps({'load_balancing_basis':score,'target':target}),flush=True)
                    break
                closure=np.asarray([reduce(v,pivots)==0 for v in words],dtype=float)
                cuts.append((closure,len(pivots)))
            if best[0][0]==10 or time.monotonic()-started>=milp_seconds: break
    return best[1],{'score':best[0],'random_trials':trials,'seed':seed,
                    'milp_seconds_cap':milp_seconds,'milp_history':history,
                    'peak_load_optimality':'proved by average load' if best[0][0]==10 else 'not proved'}


def coordinates(rows, vector):
    pivots={}
    for i,row in enumerate(rows):
        coefficient=1<<i
        while row:
            p=row.bit_length()-1
            if p not in pivots:
                pivots[p]=(row,coefficient)
                break
            row ^= pivots[p][0]
            coefficient ^= pivots[p][1]
    coefficients=0
    residual=vector
    while residual:
        p=residual.bit_length()-1
        if p not in pivots: raise ValueError('vector outside generator span')
        residual ^= pivots[p][0]
        coefficients ^= pivots[p][1]
    assert combine(rows,coefficients)==vector
    return coefficients


def orbit_search(words,px,py,fold):
    remaining=set(words)
    orbits=[]
    generators=(px,py,power(fold,2))
    while remaining:
        seed=min(remaining)
        orbit={seed}
        stack=[seed]
        while stack:
            v=stack.pop()
            for p in generators:
                w=permute(v,p)
                if w not in orbit:
                    orbit.add(w)
                    stack.append(w)
        assert orbit<=remaining
        remaining -= orbit
        orbits.append(sorted(orbit))
    assert len(orbits)<20
    best=None
    feasible=0
    for mask in range(1,1<<len(orbits)):
        rows=[v for i,o in enumerate(orbits) if mask>>i&1 for v in o]
        if rank(rows)!=24: continue
        feasible+=1
        score=(sum(v.bit_count() for v in rows),len(rows),mask)
        if best is None or score<best[0]: best=(score,rows)
    assert best is not None
    return {'maximum_check_weight':14,'physical_generators':['Tx','Ty','fold_squared'],
            'orbit_count':len(orbits),'subsets_examined':2**len(orbits)-1,'spanning_subsets':feasible,
            'orbits':[{'size':len(o),'weight':o[0].bit_count(),'rank':rank(o),'rows':o} for o in orbits],
            'minimum_total_weight_per_sector':best[0][0],'number_checks_per_sector':best[0][1],
            'selected_orbit_mask':best[0][2],'best_x_rows':best[1]}


def main(args):
    destination=Path(args.output)
    if destination.exists(): raise SystemExit(f'Refusing to overwrite {destination}')
    data=json.loads(SOURCE.read_text())
    started=time.monotonic()
    enumerations={side:enumerate_space(data[side]) for side in ('hx','hz')}
    assert enumerations['hx']['weight_histogram']==enumerations['hz']['weight_histogram']
    for side,result in enumerations.items():
        print(json.dumps({'sector':side,'enumerated':result['enumerated'],
                          'minimum_basis':result['minimum_basis_weight_counts'],
                          'total_weight':result['minimum_basis_total_weight']}),flush=True)
    words=enumerations['hx']['low_weight_rows']
    fold=data['certificate']['fold']
    selected,balancing=balance(words,fold,args.trials,args.seed,args.milp_seconds)
    hx=[words[i] for i in selected]
    hz=[permute(v,fold) for v in hx]
    assert same_space(hx,data['hx']) and same_space(hz,data['hz'])
    assert Counter(v.bit_count() for v in hx)=={12:18,14:6}
    orbit=orbit_search(words,data['px'],data['py'],fold)
    change={}
    actions={}
    for side,new in (('hx',hx),('hz',hz)):
        change[side]={'new_from_original':[coordinates(data[side],v) for v in new],
                      'original_from_new':[coordinates(new,v) for v in data[side]]}
        actions[side]={name:[coordinates(new,permute(v,p)) for v in new]
                       for name,p in (('Tx',data['px']),('Ty',data['py']))}
    actions['H']={'x_to_z':[coordinates(hz,permute(v,fold)) for v in hx],
                  'z_to_x':[coordinates(hx,permute(v,fold)) for v in hz]}
    reduced={**data,'hx':hx,'hz':hz,'presentation':'minimum-total-weight independent CSS generators'}
    # Different exact array-based routines recheck the entire code and gates.
    from extract_component import audit
    audited,arrays=audit(reduced,seconds=0)
    report={'status':'exact_check_weight_optimization','source_candidate':str(SOURCE),
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'enumerations':enumerations,'load_balancing':balancing,'symmetry_closed_search':orbit,
            'basis_changes':change,'stabilizer_actions':actions,'audit':audited,
            'elapsed_seconds':time.monotonic()-started,
            'original_total_support':sum(v.bit_count() for v in data['hx']+data['hz']),
            'optimized_total_support':sum(v.bit_count() for v in hx+hz)}
    destination.mkdir(parents=True)
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (destination/'candidate.json').write_text(json.dumps(reduced,indent=2)+'\n')
    np.savez_compressed(destination/'checks.npz',hx=arrays[0],hz=arrays[1])
    (destination/'supports.json').write_text(json.dumps({
        side:[[i for i in range(64) if v>>i&1] for v in rows] for side,rows in (('hx',hx),('hz',hz))},indent=2)+'\n')
    print(json.dumps({'output':str(destination),'original_support':report['original_total_support'],
                      'optimized_support':report['optimized_total_support'],'load_score':balancing['score'],
                      'symmetry_closed_minimum_per_sector':orbit['minimum_total_weight_per_sector']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--trials',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--milp-seconds',type=float,default=60)
    main(parser.parse_args())
