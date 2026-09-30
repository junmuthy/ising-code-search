"""Fold-constrained abelian two-block control, including the known component.

Unlike the standard square commuting parent, HZ is P(HX), not automatically
[B^T|A^T]. For the swapping affine folds below, CSS is linear in b at fixed a.
P squared is a common translation, so reverse ZX duality is automatic.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from sparse_core import (HERE, assess, commute, compose, permute, power, rank,
                         same_space, save_certificate, space_key)
from sparse_models import SmallGroup
from sparse_pilot import group_seed
from search import linear_kernel, sparse_vectors
from algebra import bits


GROUP = SmallGroup('c8c4')


def fold_catalog():
    theta_seen = set()
    for p in GROUP.folds():
        theta = tuple(v%32 for v in p[:32])
        if theta in theta_seen or any(theta[theta[g]] != g for g in range(32)):
            continue
        theta_seen.add(theta)
        for c in (0, 16):
            if theta[c] == c:
                yield [32+theta[g] for g in range(32)] + [GROUP.mul(c, theta[g]) for g in range(32)]


def checks(a, b, fold):
    hx = GROUP.checks(a, b)[0]
    return hx, [permute(v, fold) for v in hx]


def equation(a, b, fold):
    hx, hz = checks(a, b, fold)
    return sum(((hx[0]&row).bit_count()%2) << j for j, row in enumerate(hz))


def known_seed():
    original = json.loads((HERE/'certified/64_16_6_c4xc4/candidate.json').read_text())
    ps = [compose(power(original['px'], i), power(original['py'], j)) for i in range(8) for j in range(4)]
    first = {p[0] for p in ps}
    second = next(q for q in range(64) if q not in first)
    coordinates = [p[b] for b in (0, second) for p in ps]
    inverse = [coordinates.index(i) for i in range(64)]
    hx, hz = [[permute(v, inverse) for v in original[side]] for side in ('hx', 'hz')]
    a = sum(1 << GROUP.inverse[g] for g in bits(hx[0]&((1 << 32)-1)))
    b = sum(1 << GROUP.inverse[g] for g in bits(hx[0] >> 32))
    fold = [inverse[original['certificate']['fold'][q]] for q in coordinates]
    xx, zz = checks(a, b, fold)
    assert same_space(hx, xx) and same_space(hz, zz)
    assert equation(a, b, fold) == 0 and commute(xx, zz)
    return a, b, fold, coordinates


def run(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    rng = random.Random(args.seed)
    planted_a, planted_b, planted_fold, _ = known_seed()
    folds = [planted_fold] + [p for p in fold_catalog() if p != planted_fold]
    started = time.monotonic()
    counts, dims, seen = Counter(), Counter(), set()
    px, py, central = GROUP.physical()
    with (destination/'candidates.jsonl').open('x') as stream:
        for index in range(args.fixed):
            if time.monotonic()-started > args.seconds:
                counts['time_cap_reached'] += 1
                break
            # Half the fixed inputs explore the known fold; the remainder
            # sample the full catalog. Include bounded support deletions/swaps
            # around the known a, alongside noncentral product seeds.
            fold = folds[0] if index%2 == 0 else folds[(index//2)%len(folds)]
            if index%4 == 0:
                a = planted_a
                for g in rng.sample(list(bits(a)), rng.choice((0, 2, 4))):
                    a ^= 1 << g
                if rng.randrange(2):
                    g = rng.choice(list(bits(a)))
                    h = rng.choice([q for q in range(32) if not a >> q & 1])
                    a ^= (1 << g) | (1 << h)
            else:
                a = group_seed(GROUP, rng, index%6, args.maxweight)
            if not a or a.bit_count() >= args.maxweight or rank(GROUP.lift(a)) > 24:
                counts['fixed_rejected'] += 1
                continue
            kernel = linear_kernel([equation(a, 1 << i, fold) for i in range(32)], 32)
            dims[len(kernel)] += 1
            for b in sparse_vectors(kernel, rng, args.per_fixed, args.maxweight-a.bit_count()):
                hx, hz = checks(a, b, fold)
                assert commute(hx, hz)
                key = space_key(hx, hz)
                # Same code with a different fold must be considered again.
                if (key, tuple(fold)) in seen:
                    counts['duplicate_code_fold'] += 1
                    continue
                seen.add((key, tuple(fold)))
                info, certificate = assess(hx, hz, px, py, [fold], central)
                record = dict(index=index, a=a, b=b, fold=fold, row_space_id=key,
                              max_check_weight=max(v.bit_count() for v in hx+hz), **info)
                counts[info['status']] += 1
                stream.write(json.dumps(record)+'\n')
                if certificate is not None:
                    data = dict(record, construction='fold_constrained_abelian', n=64, k=16, hx=hx, hz=hz,
                                px=px, py=py, central=central, certificate=certificate, logical_shape=[4, 4])
                    if not (destination/'certified'/key).exists():
                        save_certificate(data, destination/'certified'/key)
                    print(json.dumps({'accepted': key, 'weight': record['max_check_weight']}), flush=True)
                if time.monotonic()-started > args.seconds:
                    break
            if (index+1)%16 == 0:
                stream.flush()
                print(json.dumps(dict(index=index+1, counts=dict(counts), seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(status='complete_bounded_pilot', parameters=vars(args), fold_count=len(folds),
                   totals=dict(counts), kernel_dimensions=dict(dims), unique_code_fold_pairs=len(seen),
                   elapsed_seconds=time.monotonic()-started,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('sparse_*.py')})
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxweight', type=int, default=10)
    parser.add_argument('--fixed', type=int, default=1024)
    parser.add_argument('--per-fixed', type=int, default=128)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
