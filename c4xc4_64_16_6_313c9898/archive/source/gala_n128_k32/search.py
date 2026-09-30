"""Bounded, reproducible feasibility pilots; never infer distance from sampling."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

from algebra import (bits, combine, commute, hadamard, low_logical, module,
                     nullspace, permute, rank, transpose)
from models import (Extension, affine_fold, decode_mat, gl_checks, lift_gl,
                    mat_commutator, ring_mul, translation)


def linear_kernel(images, width):
    return nullspace(transpose(images,width),len(images))


def random_word(rng, positions, weight):
    return sum(1 << i for i in rng.sample(list(positions),min(weight,len(positions))))


def sparse_vectors(kernel, rng, count, maxweight):
    """Sample the full kernel, then greedily sparsify without leaving it.

    Include sparse basis vectors. This is a heuristic sampler, not a complete
    enumeration of low-weight elements or a uniform sample of commuting pairs.
    """
    if not kernel: return
    seen=set()
    ordered=sorted(kernel,key=int.bit_count)
    for v in ordered:
        if v.bit_count() <= maxweight and v not in seen:
            seen.add(v)
            yield v
            if len(seen) >= max(1,count//4): break
    for attempt in range(count*10):
        if attempt%3 == 0:
            v=combine(kernel,rng.getrandbits(len(kernel)))
        else:
            v=combine(kernel,sum(1<<i for i in rng.sample(range(len(kernel)),min(len(kernel),rng.randint(2,8)))))
        for _ in range(4):
            changed=False
            rng.shuffle(ordered)
            for b in ordered:
                if (v^b).bit_count() < v.bit_count():
                    v ^= b
                    changed=True
            if not changed: break
        if v and v.bit_count() <= maxweight and v not in seen:
            seen.add(v)
            yield v
            if len(seen) >= count: return


def gl_seed(rng, index):
    mode=index%8
    if mode == 0:
        f=[random_word(rng,range(32),rng.randint(1,3)) for _ in range(4)]
    elif mode == 1:
        a,b,c=[random_word(rng,range(32),rng.randint(1,3)) for _ in range(3)]
        f=[a,b,b,c]
    elif mode == 2:
        factor=1 ^ (1 << rng.choice((4,16,20)))
        f=[ring_mul(factor,random_word(rng,range(32),rng.randint(1,3))) for _ in range(4)]
    elif mode == 3:
        f=[random_word(rng,range(32),rng.choice((2,4))) for _ in range(4)]
    elif mode == 4:
        f=[random_word(rng,range(32),rng.choice((2,4))),0,0,random_word(rng,range(32),rng.choice((2,4)))]
    elif mode == 5:
        a,b=[random_word(rng,range(32),rng.choice((2,4))) for _ in range(2)]
        f=[a,b,0,a]
    elif mode == 6:
        subgroup=[i for i in range(32) if i%2 == 0]
        f=[random_word(rng,subgroup,rng.choice((2,4))) for _ in range(4)]
    else:
        a=random_word(rng,range(32),rng.choice((1,2,3)))
        f=[a,a,a,a]
    return f,mode


def gl_folds():
    # A bounded selection of affine normalizers, checked on row spaces later.
    sigmas=((0,1,2,3),(2,3,0,1),(1,0,3,2),(3,2,1,0),(1,2,3,0),(3,0,1,2))
    phis=((1,0,0,1),(-1,0,0,-1),(1,0,0,-1),(-1,0,0,1),(1,0,-2,-1))
    for sigma,phi in itertools.product(sigmas,phis):
        for offsets in (None,[(0,0),(0,1),(0,0),(0,1)],[(0,0),(1,0),(0,0),(1,0)]):
            yield affine_fold(sigma,phi,offsets)


def assess(hx,hz,px,py,folds,central=None):
    info,grid=module(hx,hz,px,py,central)
    if grid is None: return info,None
    # Reject by exact witness before spending time on logical basis synthesis.
    for sector,check,dual in (('Z',hx,grid['x']),('X',hz,grid['z'])):
        witness=low_logical(check,dual,len(px))
        if witness:
            return {**info,'status':'low_distance','sector':sector,'witness':witness},grid
    fold_status=Counter()
    for fold in folds:
        h,logical=hadamard(hx,hz,grid,fold)
        fold_status[h['status']] += 1
        if logical:
            return {**info,'status':'accepted','distance_lower_bound':6,'hadamard':h}, {**logical,'fold':fold}
    return {**info,'status':'h_not_certified','distance_lower_bound':6,'fold_status':dict(fold_status)},grid


def run(args):
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    journal=destination/'candidates.jsonl'
    if journal.exists():
        raise SystemExit(f'Refusing to mix runs: {journal} already exists')
    rng=random.Random(args.seed)
    started=time.monotonic()
    totals=Counter()
    ranks=Counter()
    commutants=Counter()
    seen=set()
    fixed_records=[]
    extensions={(a,b):Extension(a,b) for a,b in itertools.product(range(2),repeat=2)}
    folded={key:list(e.folds()) for key,e in extensions.items()}
    glfolds=list(gl_folds())
    with journal.open('x') as stream:
        for index in range(args.fixed):
            if time.monotonic()-started > args.seconds:
                totals['time_cap_reached'] += 1
                break
            if args.branch == 'gl':
                f,mode=gl_seed(rng,index)
                fixed_rank=rank(lift_gl(f))
                metadata={'branch':'gl','index':index,'mode':mode,'f':f,'fixed_rank':fixed_rank}
                width=128
                first=f
                if fixed_rank > 48:
                    totals['fixed_rank_too_high'] += 1
                    fixed_records.append({**metadata,'status':'rank_obstruction_for_this_f'})
                    continue
                kernel=linear_kernel([mat_commutator(f,decode_mat(1<<i)) for i in range(128)],128)
                px,py,central=translation(1,0),translation(0,1),None
                folds=glfolds
            else:
                key=list(extensions)[index%4]
                e=extensions[key]
                mode=(index//4)%4
                if args.extension_products:
                    mode += 4
                if mode == 0:
                    a=random_word(rng,range(64),rng.choice((2,4,6)))
                elif mode == 1:
                    a=e.polynomial_mul(3,random_word(rng,range(64),rng.choice((1,2,3))))
                elif mode == 2:
                    a=e.polynomial_mul(1^(1<<32),random_word(rng,range(64),rng.choice((1,2,3))))
                elif mode == 3:
                    central_elements=[g for g in range(64) if all(e.mul(g,h)==e.mul(h,g) for h in (16,2))]
                    a=random_word(rng,central_elements,rng.choice((2,4)))
                elif mode in (4,5):
                    g,h=rng.sample(range(1,64),2)
                    a=e.polynomial_mul(1^(1<<g),1^(1<<h))
                    if mode == 5:
                        a=e.polynomial_mul(random_word(rng,range(64),3),a)
                elif mode == 6:
                    g,h,j=rng.sample(range(1,64),3)
                    a=e.polynomial_mul(e.polynomial_mul(1^(1<<g),1^(1<<h)),1^(1<<j))
                else:
                    a=0
                    for _ in range(2):
                        g,h=rng.sample(range(1,64),2)
                        a ^= e.polynomial_mul(1^(1<<g),1^(1<<h))
                fixed_rank=rank(e.lift(a))
                metadata={'branch':'extension','index':index,'extension':key,'mode':mode,'a':a,'fixed_rank':fixed_rank}
                width=64
                first=a
                if fixed_rank > 48:
                    totals['fixed_rank_too_high'] += 1
                    fixed_records.append({**metadata,'status':'rank_obstruction_for_this_a'})
                    continue
                kernel=linear_kernel([e.commutator(a,1<<i) for i in range(64)],64)
                px,py,central=e.right(16),e.right(2),e.right(1)
                folds=folded[key]
            commutants[len(kernel)] += 1
            fixed_records.append({**metadata,'commutant_dimension':len(kernel)})
            for v in sparse_vectors(kernel,rng,args.per_fixed,2*args.maxweight if width == 128 else args.maxweight):
                second=decode_mat(v) if width == 128 else v
                identity=json.dumps([metadata['branch'],metadata.get('extension'),first,second])
                digest=hashlib.sha256(identity.encode()).hexdigest()
                if digest in seen:
                    totals['duplicates'] += 1
                    continue
                seen.add(digest)
                hx,hz=gl_checks(first,second) if width == 128 else e.checks(first,second)
                weights=[r.bit_count() for r in hx+hz]
                if max(weights) > args.maxweight:
                    totals['overweight'] += 1
                    continue
                rx,rz=rank(hx),rank(hz)
                ranks[f'{rx},{rz}'] += 1
                record={**metadata,'g' if width == 128 else 'b':second,
                        'id':digest,'rank_x':rx,'rank_z':rz,'max_check_weight':max(weights)}
                if (rx,rz) != (48,48):
                    info={'status':'wrong_ranks'}
                    certificate=None
                else:
                    info,certificate=assess(hx,hz,px,py,folds,central)
                record.update(info)
                totals[info['status']] += 1
                stream.write(json.dumps(record,sort_keys=True)+'\n')
                if info['status'] == 'accepted':
                    artifact={**record,'hx':hx,'hz':hz,'px':px,'py':py,'central':central,'certificate':certificate}
                    with (destination/f'{digest}.json').open('x') as output:
                        json.dump(artifact,output,indent=2)
                if time.monotonic()-started > args.seconds: break
            if (index+1)%16 == 0:
                stream.flush()
                print(json.dumps({'fixed':index+1,'elapsed':round(time.monotonic()-started,2),'totals':dict(totals)}),flush=True)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    summary={'parameters':vars(args),'elapsed_seconds':time.monotonic()-started,
             'totals':dict(totals),'rank_histogram':dict(ranks),'commutant_dimensions':dict(commutants),
             'fixed_records':fixed_records,'source_sha256':sources,
             'gala_revision':subprocess.check_output(['git','-C','/home/judah_unmuth/gala-code-search','rev-parse','HEAD'],text=True).strip()}
    with (destination/'summary.json').open('x') as output:
        json.dump(summary,output,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('fixed_records','source_sha256')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=('gl','extension'),required=True)
    parser.add_argument('--fixed',type=int,default=128)
    parser.add_argument('--per-fixed',type=int,default=64)
    parser.add_argument('--maxweight',type=int,default=12)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--extension-products',action='store_true',help='Use noncentral factor-product seed families')
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
