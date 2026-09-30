"""Independent array-based audit of a saved candidate and distance cross-checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import galois
import numpy as np

from algebra import bits, unpack

GF2=galois.GF(2)


def gf_rank(a):
    return int(np.linalg.matrix_rank(GF2(a)))


def in_space(rows, space):
    return gf_rank(np.vstack((rows,space))) == gf_rank(space)


def moved(rows,p):
    return rows[:,np.argsort(p)]


def permutation_order(p):
    seen=set()
    lengths=[]
    for q in range(len(p)):
        if q in seen: continue
        here=q
        length=0
        while here not in seen:
            seen.add(here)
            here=p[here]
            length+=1
        lengths.append(length)
    return math.lcm(*lengths)


def independent_small_errors(check,stabilizers):
    """Enumerate sorted supports, then test stabilizer membership directly.

    This does not use a logical basis or the search verifier's signatures.
    Each weight-5 support occurs exactly once as its first three + last two.
    """
    n=check.shape[1]
    columns=[sum(int(check[i,j]) << i for i in range(len(check))) for j in range(n)]
    singles=defaultdict(list)
    pairs=defaultdict(list)
    zero=[]
    for q,s in enumerate(columns):
        singles[s].append(q)
        if s==0: zero.append((q,))
    for a,b in itertools.combinations(range(n),2):
        s=columns[a]^columns[b]
        pairs[s].append((a,b))
        if s==0: zero.append((a,b))
    for syndrome,bucket in pairs.items():
        for a,b in bucket:
            for c in singles.get(syndrome,()):
                if b<c: zero.append((a,b,c))
            for c,d in bucket:
                if b<c: zero.append((a,b,c,d))
    for a,b,c in itertools.combinations(range(n),3):
        for d,e in pairs.get(columns[a]^columns[b]^columns[c],()):
            if c<d: zero.append((a,b,c,d,e))
    assert len(zero)==len(set(zero))
    for support in zero:
        row=np.zeros((1,n),dtype=np.uint8)
        row[0,list(support)]=1
        if not in_space(row,stabilizers):
            return {'status':'logical_witness','support':support}
    return {'status':'certified','maximum':5,'zero_syndrome_counts':dict(Counter(map(len,zero))),
            'zero_syndrome_total':len(zero),'all_are_stabilizers':True}


def weight_six_witness(check,stabilizers):
    n=check.shape[1]
    columns=[sum(int(check[i,j]) << i for i in range(len(check))) for j in range(n)]
    triples=defaultdict(list)
    for triple in itertools.combinations(range(n),3):
        s=columns[triple[0]]^columns[triple[1]]^columns[triple[2]]
        for other in triples[s]:
            support=sorted(set(triple).symmetric_difference(other))
            if len(support)!=6: continue
            row=np.zeros((1,n),dtype=np.uint8)
            row[0,support]=1
            if not in_space(row,stabilizers): return support
        triples[s].append(triple)
    return None


def milp(check,dual,seconds,maximum=5):
    """One parity suffices because the independently checked grid is transitive.

    Any nontrivial logical pairs oddly with some dual basis element. Translate
    it so that this element becomes logical zero; the weight is unchanged.
    Floating-point MILP infeasibility is a cross-check, not the exact proof.
    """
    import cvxpy as cp
    e=cp.Variable(check.shape[1],boolean=True)
    slack=cp.Variable(len(check),integer=True)
    logical_slack=cp.Variable(integer=True)
    constraints=[check.astype(int)@e==2*slack,slack>=0,
                 slack<=np.sum(check,axis=1)//2,
                 dual[0].astype(int)@e==1+2*logical_slack,logical_slack>=0,
                 logical_slack<=int(np.sum(dual[0]))//2,cp.sum(e)<=maximum]
    problem=cp.Problem(cp.Minimize(cp.sum(e)),constraints)
    start=time.monotonic()
    try:
        value=problem.solve(solver='HIGHS',highs_options={'time_limit':seconds,'threads':1})
    except cp.SolverError as exc:
        return {'status':'solver_error','detail':str(exc),'seconds':time.monotonic()-start}
    result={'status':problem.status,'seconds':time.monotonic()-start,'maximum':maximum,
            'logical_parity_index':0,'symmetry_reduction':'regular logical grid'}
    if problem.status == cp.INFEASIBLE:
        result['crosscheck_pass']=True
    elif e.value is not None:
        vector=np.rint(e.value).astype(np.uint8)
        if np.max(np.abs(e.value-vector))<1e-5 and int(vector.sum())<=maximum and not np.any(check@vector%2) and dual[0]@vector%2:
            result['witness']=np.flatnonzero(vector).tolist()
            result['witness_validated']=True
        else:
            result['crosscheck_pass']=False
    else:
        result['crosscheck_pass']=False
    return result


def audit(data,seconds):
    n=data['n']
    hx,hz=unpack(data['hx'],n),unpack(data['hz'],n)
    cert=data['certificate']
    z,x=unpack(cert['z'],n),unpack(cert['x'],n)
    assert n<=130 and n-gf_rank(hx)-gf_rank(hz)==32
    assert not np.any(hx@hz.T%2)
    assert not np.any(hx@z.T%2) and not np.any(hz@x.T%2)
    assert np.array_equal(z@x.T%2,np.eye(32,dtype=np.uint8))
    actions={}
    for name,p in (('Tx',data['px']),('Ty',data['py'])):
        assert sorted(p)==list(range(n))
        for h in (hx,hz): assert in_space(moved(h,p),h)
        target=[((i+1)%4)*8+j if name=='Tx' else i*8+(j+1)%8 for i in range(4) for j in range(8)]
        assert in_space(moved(z,p)^z[target],hz)
        assert in_space(moved(x,p)^x[target],hx)
        actions[name]={'physical_order':permutation_order(p),'logical_permutation':target}
    if data.get('central') is not None:
        p=data['central']
        assert sorted(p)==list(range(n))
        for h in (hx,hz): assert in_space(moved(h,p),h)
        assert in_space(moved(z,p)^z,hz) and in_space(moved(x,p)^x,hx)
        actions['central']={'physical_order':permutation_order(p),'logical_action':'identity'}
    p=cert['fold']
    assert sorted(p)==list(range(n))
    assert in_space(moved(hx,p),hz) and in_space(moved(hz,p),hx)
    action_zx=moved(z,p)@z.T%2
    action_xz=moved(x,p)@x.T%2
    assert np.array_equal(action_zx,action_xz)
    assert np.all(np.sum(action_zx,axis=0)==1) and np.all(np.sum(action_zx,axis=1)==1)
    target=np.argmax(action_zx,axis=1)
    assert in_space(moved(z,p)^x[target],hx)
    assert in_space(moved(x,p)^z[target],hz)
    actions['H']={'physical_fold_order':permutation_order(p),'logical_permutation':target.tolist(),
                  'zx_matrix':action_zx.tolist(),'xz_matrix':action_xz.tolist()}
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for row in np.vstack((hx,hz)):
        support=np.flatnonzero(row)
        for q in support[1:]: parent[root(int(q))]=root(int(support[0]))
    metrics={'component_sizes':sorted(Counter(root(i) for i in range(n)).values()),
             'check_weight_x':dict(Counter(map(int,hx.sum(axis=1)))),
             'check_weight_z':dict(Counter(map(int,hz.sum(axis=1)))),
             'degree_x':dict(Counter(map(int,hx.sum(axis=0)))),
             'degree_z':dict(Counter(map(int,hz.sum(axis=0)))),
             'logical_weights_x':dict(Counter(map(int,x.sum(axis=1)))),
             'logical_weights_z':dict(Counter(map(int,z.sum(axis=1))))}
    # Displayed Tanner connectivity alone can be manufactured by multiplying
    # checks from disjoint blocks. RREF removes this generator-choice artifact.
    parent[:]=range(n)
    for row in np.vstack((np.asarray(GF2(hx).row_reduce()),np.asarray(GF2(hz).row_reduce()))):
        support=np.flatnonzero(row)
        for q in support[1:]: parent[root(int(q))]=root(int(support[0]))
    metrics['row_space_component_sizes']=sorted(Counter(root(i) for i in range(n)).values())
    print(json.dumps({'algebra':'passed','metrics':metrics,'physical_orders':{k:v.get('physical_order',v.get('physical_fold_order')) for k,v in actions.items()}}),flush=True)
    distance={}
    for sector,check,stabilizers,dual in (('Z',hx,hz,x),('X',hz,hx,z)):
        exact=independent_small_errors(check,stabilizers)
        assert exact['status']=='certified'
        witness=weight_six_witness(check,stabilizers)
        distance[sector]={'exclusion':exact,'weight_six_witness':witness,'lower_bound':6 if witness else 7,
                          'weight_six_search':'complete triple-pair collision enumeration',
                          'upper_bound':6 if witness else None}
        print(json.dumps({'sector':sector,**distance[sector]}),flush=True)
        if seconds>0:
            distance[sector]['milp']=milp(check,dual,seconds)
            assert 'witness' not in distance[sector]['milp']
            print(json.dumps({'sector':sector,'milp':distance[sector]['milp']}),flush=True)
            if witness is None:
                upper=milp(check,dual,seconds,maximum=7)
                distance[sector]['weight_seven_search']=upper
                if 'witness' in upper:
                    assert len(upper['witness'])==7
                    distance[sector]['upper_bound']=7
                print(json.dumps({'sector':sector,'weight_seven_search':upper}),flush=True)
    return {'status':'certified','n':n,'k':32,'rank_x':gf_rank(hx),'rank_z':gf_rank(hz),
            'distance':distance,'actions':actions,'metrics':metrics},(hx,hz,x,z)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('--output',required=True)
    parser.add_argument('--milp-seconds',type=float,default=45)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists(): raise SystemExit(f'Refusing to overwrite {destination}')
    data=json.loads(Path(args.candidate).read_text())
    result,(hx,hz,x,z)=audit(data,args.milp_seconds)
    destination.mkdir(parents=True)
    result['candidate_path']=args.candidate
    result['candidate_sha256']=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest()
    result['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (destination/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'candidate.json').write_text(json.dumps(data,indent=2)+'\n')
    np.savez_compressed(destination/'checks.npz',hx=hx,hz=hz)
    np.savez_compressed(destination/'logical_bases.npz',x=x,z=z)
    (destination/'physical_permutations.json').write_text(json.dumps({
        'tx':data['px'],'ty':data['py'],'central':data.get('central'),'fold':data['certificate']['fold']},indent=2)+'\n')
    print(json.dumps({'status':'certified','output':str(destination)}),flush=True)
