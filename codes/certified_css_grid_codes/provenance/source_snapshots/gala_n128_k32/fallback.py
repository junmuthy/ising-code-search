"""Small general equivariant CSS control, outside the square GALA parent.

Start with 48 X and 48 Z Bell-state checks and one free 32-site sheet.
Apply translation-equivariant binary orthogonal transvections to both Pauli
sectors. This preserves the regular logical grid and H exactly but can change
distance. The 16-site check orbit is deliberately not a free R-row.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from algebra import combine, commute, hadamard, low_logical, module, pairing, rank
from models import TRANSLATE, translation
from search import random_word


def initial():
    h=[(1 << g) | (1 << (32+g)) for g in range(32)]
    h += [(1 << (64+g)) | (1 << (64+g+16)) for g in range(16)]
    z=[1 << (96+g) for g in range(32)]
    return h,z


def transvection(h,z,a,b,order):
    entries=[a,a,b,b]
    v=[sum(1 << (32*s+TRANSLATE[g][t]) for s in range(4) for t in range(32)
           if entries[order[s]] >> t & 1) for g in range(32)]
    assert commute(v,v)
    def move(rows):
        return [row ^ combine(v,coeff) for row,coeff in zip(rows,pairing(rows,v))]
    return move(h),move(z)


def run(args):
    rng=random.Random(args.seed)
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    totals=Counter()
    start=time.monotonic()
    with (destination/'candidates.jsonl').open('x') as stream:
        for trial in range(args.trials):
            h,z=initial()
            recipe=[]
            layers=1+trial%args.layers
            for _ in range(layers):
                a=random_word(rng,range(32),rng.choice((1,1,2,3)))
                b=random_word(rng,range(32),rng.choice((1,1,2,3)))
                order=rng.sample(range(4),4)
                h,z=transvection(h,z,a,b,order)
                recipe.append([a,b,order])
            assert rank(h)==48 and commute(h,h)
            assert commute(h,z) and pairing(z,z)==[1 << i for i in range(32)]
            record={'trial':trial,'layers':layers,'recipe':recipe,
                    'max_check_weight':max(v.bit_count() for v in h)}
            if record['max_check_weight'] > args.maxweight:
                status='overweight'
            else:
                witness=low_logical(h,z,128)
                if witness:
                    status='low_distance'
                    record['witness']=witness
                else:
                    info,grid=module(h,h,translation(1,0),translation(0,1))
                    assert info['status']=='regular_grid'
                    hinfo,logical=hadamard(h,h,{'z':z,'x':z,'perms':grid['perms']},list(range(128)))
                    assert hinfo['status']=='logical_hadamard'
                    status='accepted'
                    record.update(n=128,k=32,distance_lower_bound=6,hadamard=hinfo)
                    artifact={**record,'hx':h,'hz':h,'px':translation(1,0),'py':translation(0,1),
                              'certificate':{**logical,'fold':list(range(128))},'construction':'general_equivariant_orthogonal'}
                    with (destination/f'hit_{trial}.json').open('x') as output: json.dump(artifact,output,indent=2)
            record['status']=status
            totals[status] += 1
            stream.write(json.dumps(record)+'\n')
            if (trial+1)%64==0:
                stream.flush()
                print(json.dumps({'trial':trial+1,'totals':dict(totals),'elapsed':time.monotonic()-start}),flush=True)
    summary={'parameters':vars(args),'totals':dict(totals),'elapsed_seconds':time.monotonic()-start,
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    with (destination/'summary.json').open('x') as output: json.dump(summary,output,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=256)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--maxweight',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20260913)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
