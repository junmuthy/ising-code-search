"""Reconstruct and independently audit a C4 x C4 [[64,16,6]] component.

Uses a saved nonabelian [[128,32,6]] parent; does not mutate that parent or
assume its logical basis restricts correctly to a component.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PREVIOUS=Path(__file__).resolve().parents[1]/'gala_n128_k32'
sys.path.insert(0,str(PREVIOUS))

import numpy as np

from algebra import (bits, canonical_x, combine, compose, pairing, permute,
                     power, quotient_basis, rank, unpack)
from models import Extension
from certify import (GF2, gf_rank, in_space, independent_small_errors, milp,
                     moved, permutation_order, weight_six_witness)


PARENT_ID='1c8a50d5f2a67053b0a438b9a3419073f86265e2c4593b5ac3f7d1372f86a216'
PARENT=PREVIOUS/'runs/extension_products_w16'/f'{PARENT_ID}.json'


def components(rows,n):
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for row in rows:
        support=list(bits(row))
        for q in support[1:]: parent[root(q)]=root(support[0])
    return [[i for i in range(n) if root(i)==r] for r in sorted({root(i) for i in range(n)})]


def reconstruct():
    parent=json.loads(PARENT.read_text())
    group=Extension(*parent['extension'])
    assert parent['extension']==[1,0]
    assert group.checks(parent['a'],parent['b'])==(parent['hx'],parent['hz'])
    component=next(c for c in components(parent['hx']+parent['hz'],128) if 0 in c)
    assert len(component)==64
    index={q:i for i,q in enumerate(component)}
    def project(row):
        return sum(((row>>q)&1)<<i for i,q in enumerate(component))
    def restrict(p):
        assert all(p[q] in index for q in component)
        return [index[p[q]] for q in component]
    hx=[project(row) for row in parent['hx'] if project(row)]
    hz=[project(row) for row in parent['hz'] if project(row)]
    px=restrict(parent['px'])
    py=restrict(power(parent['py'],2))
    central=restrict(parent['central'])
    # Scan found this alternative fold; the parent's saved fold alone fails
    # the simultaneous-basis requirement on the component.
    global_fold=list(group.folds())[15]
    fold_exchanges_components=global_fold[0] not in index
    if fold_exchanges_components:
        global_fold=compose(parent['py'],global_fold)
    fold=restrict(global_fold)
    permutations=[compose(power(px,i),power(py,j)) for i in range(4) for j in range(4)]
    rawz=quotient_basis(hx,hz,64)
    rawx=quotient_basis(hz,hx,64)
    orbit=next([permute(v,p) for p in permutations] for v in sorted(rawz,key=int.bit_count)
               if rank(hz+[permute(v,p) for p in permutations])==rank(hz)+16)
    # Odd-augmentation unit of F2[C4 x C4], found by the preliminary exact
    # common-basis solve. All final actions below are verified independently.
    coefficients=31433
    seed=combine(orbit,coefficients)
    z=[permute(seed,p) for p in permutations]
    x=canonical_x(z,rawx)
    assert pairing(z,x)==[1<<i for i in range(16)]
    return {'n':64,'k':16,'hx':hx,'hz':hz,'px':px,'py':py,'central':central,
            'logical_shape':[4,4],'certificate':{'z':z,'x':x,'fold':fold},
            'parent_id':PARENT_ID,'parent_coordinates':component,
            'extension':parent['extension'],'parent_a':parent['a'],'parent_b':parent['b'],
            'fold_index':15,'compose_fold_with_parent_y':fold_exchanges_components,
            'basis_coefficients':coefficients}


def audit(data,seconds=30):
    n,k=data['n'],data['k']
    assert n<70 and k==16
    hx,hz=unpack(data['hx'],n),unpack(data['hz'],n)
    x,z=unpack(data['certificate']['x'],n),unpack(data['certificate']['z'],n)
    assert gf_rank(hx)==gf_rank(hz)==24 and n-24-24==k
    assert not np.any(hx@hz.T%2)
    assert not np.any(hx@z.T%2) and not np.any(hz@x.T%2)
    assert np.array_equal(z@x.T%2,np.eye(k,dtype=np.uint8))
    actions={}
    for name,p in (('Tx',data['px']),('Ty',data['py'])):
        assert sorted(p)==list(range(n))
        for h in (hx,hz): assert in_space(moved(h,p),h)
        target=[((i+1)%4)*4+j if name=='Tx' else i*4+(j+1)%4 for i in range(4) for j in range(4)]
        assert in_space(moved(z,p)^z[target],hz)
        assert in_space(moved(x,p)^x[target],hx)
        actions[name]={'physical_order':permutation_order(p),'logical_order':4,
                       'logical_permutation':target,
                       'z_action':(moved(z,p)@x.T%2).tolist(),
                       'x_action':(moved(x,p)@z.T%2).tolist()}
    assert compose(data['px'],data['py'])==compose(data['py'],data['px'])
    p=data['central']
    for h in (hx,hz): assert in_space(moved(h,p),h)
    assert in_space(moved(z,p)^z,hz) and in_space(moved(x,p)^x,hx)
    actions['central']={'physical_order':permutation_order(p),'logical_action':'identity'}
    p=data['certificate']['fold']
    assert sorted(p)==list(range(n))
    assert in_space(moved(hx,p),hz) and in_space(moved(hz,p),hx)
    zx=moved(z,p)@z.T%2
    xz=moved(x,p)@x.T%2
    assert np.array_equal(zx,xz)
    assert np.all(zx.sum(axis=0)==1) and np.all(zx.sum(axis=1)==1)
    target=np.argmax(zx,axis=1)
    assert in_space(moved(z,p)^x[target],hx) and in_space(moved(x,p)^z[target],hz)
    actions['H']={'physical_fold_order':permutation_order(p),
                  'logical_permutation':target.tolist(),'zx_action':zx.tolist(),'xz_action':xz.tolist()}
    reduced=np.vstack((np.asarray(GF2(hx).row_reduce()),np.asarray(GF2(hz).row_reduce())))
    reduced_rows=[sum(int(v)<<i for i,v in enumerate(row)) for row in reduced]
    metrics={'displayed_component_sizes':[len(c) for c in components(data['hx']+data['hz'],n)],
             'row_space_component_sizes':[len(c) for c in components(reduced_rows,n)],
             'check_weights_x':dict(Counter(map(int,hx.sum(axis=1)))),
             'check_weights_z':dict(Counter(map(int,hz.sum(axis=1)))),
             'degrees_x':dict(Counter(map(int,hx.sum(axis=0)))),
             'degrees_z':dict(Counter(map(int,hz.sum(axis=0)))),
             'logical_weights_x':dict(Counter(map(int,x.sum(axis=1)))),
             'logical_weights_z':dict(Counter(map(int,z.sum(axis=1))))}
    assert metrics['row_space_component_sizes']==[64]
    print(json.dumps({'algebra':'passed','metrics':metrics}),flush=True)
    distance={}
    for sector,h,stabilizers,dual in (('Z',hx,hz,x),('X',hz,hx,z)):
        lower=independent_small_errors(h,stabilizers)
        assert lower['status']=='certified'
        witness=weight_six_witness(h,stabilizers)
        assert witness is not None
        distance[sector]={'distance':6,'exclusion':lower,'weight_six_witness':witness}
        if seconds>0:
            crosscheck=milp(h,dual,seconds)
            assert 'witness' not in crosscheck
            distance[sector]['milp']=crosscheck
        print(json.dumps({'sector':sector,**distance[sector]}),flush=True)
    return {'status':'certified','n':n,'k':k,'d':6,'rank_x':24,'rank_z':24,
            'actions':actions,'metrics':metrics,'distance':distance},(hx,hz,x,z)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--milp-seconds',type=float,default=30)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists(): raise SystemExit(f'Refusing to overwrite {destination}')
    data=reconstruct()
    result,(hx,hz,x,z)=audit(data,args.milp_seconds)
    destination.mkdir(parents=True)
    result['parent_sha256']=hashlib.sha256(PARENT.read_bytes()).hexdigest()
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__),PREVIOUS/'algebra.py',PREVIOUS/'models.py',PREVIOUS/'certify.py')}
    (destination/'candidate.json').write_text(json.dumps(data,indent=2)+'\n')
    (destination/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(destination/'checks.npz',hx=hx,hz=hz)
    np.savez_compressed(destination/'logical_bases.npz',x=x,z=z)
    (destination/'physical_permutations.json').write_text(json.dumps({
        'tx':data['px'],'ty':data['py'],'central':data['central'],
        'fold':data['certificate']['fold'],'parent_coordinates':data['parent_coordinates']},indent=2)+'\n')
    print(json.dumps({'status':'certified','output':str(destination)}),flush=True)
