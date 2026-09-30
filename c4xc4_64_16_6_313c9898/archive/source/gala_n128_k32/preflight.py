"""Exact capacity bounds for specified seed and displayed-row fold equations."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from algebra import combine, pairing, permute, rank
from models import affine_fold, decode_mat, gl_checks, mat_commutator, translation
from search import assess, linear_kernel, sparse_vectors


def seed_orbit(seed):
    return [permute(seed,translation(i,j)) for i in range(4) for j in range(8)]


def parameter_checks(word):
    return gl_checks(decode_mat(word),decode_mat(word >> 128))


def capacity_space(seed,fold,qx,qz,variables):
    images=[]
    for hx,hz in variables:
        output=sum(((row & seed).bit_count()%2) << i for i,row in enumerate(hx))
        for i in range(64):
            output |= (permute(hx[i],fold)^hz[qx[i]]) << (64+128*i)
            output |= (permute(hz[i],fold)^hx[qz[i]]) << (64+8192+128*i)
        images.append(output)
    kernel=linear_kernel(images,64+2*8192)
    lifted=[parameter_checks(v) for v in kernel]
    return kernel,rank([r for hx,hz in lifted for r in hx]),rank([r for hx,hz in lifted for r in hz])


def run(args):
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'spaces.jsonl'
    if path.exists(): raise SystemExit(f'Refusing to overwrite {path}')
    rng=random.Random(args.seed)
    variables=[parameter_checks(1 << i) for i in range(256)]
    seeds=[sum(1 << q for q in (0,1,32,33,64,96))]
    for weight in (7,8,9,10,6,7,8):
        seeds.append(sum(1 << q for q in rng.sample(range(128),weight)))
    sigmas=((1,2,3,0),(0,1,2,3),(1,0,2,3),(1,2,0,3))
    phis=((1,0,0,1),(-1,0,0,-1),(1,0,-2,-1))
    cases=[]
    geometry=Counter()
    for seed,phi,sigma in itertools.product(seeds,phis,sigmas):
        z=seed_orbit(seed)
        fold=affine_fold(sigma,phi)
        gram_rank=rank(pairing(z,[permute(v,fold) for v in z]))
        geometry[gram_rank] += 1
        if gram_rank != 32: continue
        for checksigma in ((0,1),(1,0)):
            q=affine_fold(checksigma,phi)
            cases.append((seed,fold,q,q,sigma,phi,checksigma))
        if len(cases) >= args.spaces: break
    started=time.monotonic()
    totals=Counter()
    fingerprints=set()
    with path.open('x') as stream:
        for index,(seed,fold,qx,qz,sigma,phi,checksigma) in enumerate(cases[:args.spaces]):
            kernel,rx,rz=capacity_space(seed,fold,qx,qz,variables)
            fingerprint=hashlib.sha256(json.dumps(kernel).encode()).hexdigest()
            duplicate=fingerprint in fingerprints
            fingerprints.add(fingerprint)
            record={'index':index,'seed_support':[i for i in range(128) if seed >> i & 1],
                    'sheet_fold':sigma,'group_map':phi,'check_sheet_map':checksigma,
                    'dimension':len(kernel),'capacity_x':rx,'capacity_z':rz,
                    'space_sha256':fingerprint,'duplicate_linear_space':duplicate,
                    'geometry_pairing_rank':32}
            if min(rx,rz) < 48:
                record['status']='rank_capacity_below_48'
            else:
                sampled=Counter()
                sample_ranks=Counter()
                for word in sparse_vectors(kernel,rng,args.samples,32):
                    f,g=decode_mat(word),decode_mat(word >> 128)
                    if mat_commutator(f,g):
                        sampled['noncommuting'] += 1
                        continue
                    hx,hz=gl_checks(f,g)
                    ranks=(rank(hx),rank(hz))
                    sample_ranks[str(ranks)] += 1
                    if ranks != (48,48):
                        sampled['wrong_ranks'] += 1
                        continue
                    info,certificate=assess(hx,hz,translation(1,0),translation(0,1),[fold])
                    sampled[info['status']] += 1
                    if info['status'] == 'accepted':
                        artifact={**info,'f':f,'g':g,'hx':hx,'hz':hz,'certificate':certificate,
                                  'px':translation(1,0),'py':translation(0,1)}
                        with (destination/f'hit_{index}.json').open('x') as output:
                            json.dump(artifact,output,indent=2)
                record.update(status='capacity_pass',sample_counts=dict(sampled),sample_ranks=dict(sample_ranks))
            totals[record['status']] += 1
            stream.write(json.dumps(record,sort_keys=True)+'\n')
            stream.flush()
            print(json.dumps({'space':index,'dimension':len(kernel),'capacity':[rx,rz],
                              'status':record['status'],'elapsed':round(time.monotonic()-started,2)}),flush=True)
    summary={'parameters':vars(args),'elapsed_seconds':time.monotonic()-started,
             'totals':dict(totals),'distinct_linear_spaces':len(fingerprints),'geometry_rank_histogram':dict(geometry),
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    with (destination/'summary.json').open('x') as stream: json.dump(summary,stream,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_sha256'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spaces',type=int,default=48)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
