"""Bounded low-check-weight construction pilots, with append-only journals."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from sparse_core import HERE, assess, commute, low_logical, pairing, rank, save_certificate, space_key
from sparse_models import (SmallGroup, RING, commutator, decode, gl_folds, initial_control,
                           lift_gl, transvection, translations, two_block)
from search import linear_kernel, random_word, sparse_vectors


def group_seed(e, rng, mode, cap):
    if mode == 0:
        return random_word(rng, range(e.order), rng.choice([2, 3, 4]))
    if mode == 1:
        return random_word(rng, range(e.order), rng.choice([2, 4, 6]))
    factors = [1^(1 << rng.randrange(1, e.order)) for _ in range(4)]
    a = e.polynomial_mul(*factors[:2])
    if mode == 2:
        return a
    if mode == 3:
        return e.polynomial_mul(a, factors[2])
    if mode == 4:
        return a^e.polynomial_mul(*factors[2:])
    return e.polynomial_mul(random_word(rng, range(e.order), 3), a)


def gl_seed(rng, mode):
    f = [random_word(rng, range(16), rng.choice((1, 2, 2, 3))) for _ in range(4)]
    if mode == 1:
        f[2] = f[1]
    elif mode == 2:
        factor = 1^(1 << rng.choice((2, 8, 10)))
        f = [RING.polynomial_mul(factor, v) for v in f]
    elif mode == 3:
        f[1] = f[2] = 0
    elif mode == 4:
        f[2], f[3] = 0, f[0]
    elif mode == 5:
        f = [RING.polynomial_mul(1^(1 << rng.randrange(1, 16)), 1^(1 << rng.randrange(1, 16))) for _ in range(4)]
    return f


def run(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    rng = random.Random(args.seed)
    groups = ([SmallGroup('extension', a, b) for a, b in itertools.product(range(2), repeat=2)]
              if args.branch == 'extension' else [SmallGroup('c8c4'), SmallGroup('c4c4c2')])
    folds = {id(e): list(e.folds()) for e in groups} if args.branch in ('extension', 'abelian') else {}
    glfolds = list(gl_folds())
    seen, fixed_seen = set(), set()
    counts, rank_counts, commutants = Counter(), Counter(), Counter()
    started = time.monotonic()
    with (destination/'candidates.jsonl').open('x') as journal:
        for index in range(args.fixed):
            if time.monotonic()-started > args.seconds:
                counts['time_cap_reached'] += 1
                break
            if args.branch == 'control':
                h, z = initial_control()
                recipe = []
                for _ in range(1+index%6):
                    a, b = [random_word(rng, range(16), rng.choice((1, 1, 2, 3))) for _ in range(2)]
                    order = rng.sample(range(4), 4)
                    h, z = transvection(h, z, a, b, order)
                    recipe.append([a, b, order])
                assert rank(h) == 24 and commute(h, h) and pairing(z, z) == [1 << i for i in range(16)]
                candidates = [(h, h, {'recipe': recipe})]
                px, py = translations()
                central, candidates_folds = None, [list(range(64))]
            elif args.branch == 'gl':
                f = gl_seed(rng, index%6)
                fixed_key = tuple(f)
                if fixed_key in fixed_seen:
                    counts['duplicate_fixed'] += 1
                    continue
                fixed_seen.add(fixed_key)
                if rank(lift_gl(f)) > 24:
                    counts['fixed_rank_too_high'] += 1
                    continue
                kernel = linear_kernel([commutator(f, decode(1 << i)) for i in range(64)], 64)
                commutants[len(kernel)] += 1
                candidates = (( *two_block(lift_gl(f), lift_gl(decode(v))), {'f': f, 'g': decode(v)})
                              for v in sparse_vectors(kernel, rng, args.per_fixed, args.maxweight*2))
                px, py = translations()
                central, candidates_folds = None, glfolds
            else:
                e = groups[index%len(groups)]
                a = group_seed(e, rng, index//len(groups)%6, args.maxweight)
                fixed_key = (e.family, e.alpha, e.beta, a)
                if fixed_key in fixed_seen:
                    counts['duplicate_fixed'] += 1
                    continue
                fixed_seen.add(fixed_key)
                if a.bit_count() >= args.maxweight or rank(e.lift(a)) > 24:
                    counts['fixed_weight_or_rank_rejected'] += 1
                    continue
                kernel = linear_kernel([e.polynomial_mul(a, 1 << i)^e.polynomial_mul(1 << i, a)
                                        for i in range(e.order)], e.order)
                commutants[len(kernel)] += 1
                # In abelian algebras the coordinate basis makes greedy descent
                # collapse to monomials. Sample sparse supports directly there.
                if args.branch == 'abelian':
                    partners = [random_word(rng, range(e.order), rng.randint(1, args.maxweight-a.bit_count()))
                                for _ in range(args.per_fixed)]
                else:
                    partners = sparse_vectors(kernel, rng, args.per_fixed, args.maxweight-a.bit_count())
                candidates = ((*e.checks(a, b), {'family': e.family, 'extension': [e.alpha, e.beta], 'a': a, 'b': b}) for b in partners)
                px, py, central = e.physical()
                candidates_folds = folds[id(e)]
            for hx, hz, recipe in candidates:
                maximum = max(v.bit_count() for v in hx+hz)
                if maximum > args.maxweight:
                    counts['overweight'] += 1
                    continue
                key = space_key(hx, hz)
                if key in seen:
                    counts['duplicate_row_spaces'] += 1
                    continue
                seen.add(key)
                rx, rz = rank(hx), rank(hz)
                rank_counts[f'{rx},{rz}'] += 1
                record = dict(index=index, branch=args.branch, row_space_id=key, max_check_weight=maximum, **recipe)
                if (rx, rz) != (24, 24):
                    info, certificate = dict(status='wrong_ranks', rank_x=rx, rank_z=rz), None
                else:
                    info, certificate = assess(hx, hz, px, py, candidates_folds, central)
                record.update(info)
                counts[record['status']] += 1
                journal.write(json.dumps(record)+'\n')
                if certificate is not None:
                    data = dict(record, n=64, k=16, hx=hx, hz=hz, px=px, py=py, central=central,
                                logical_shape=[4, 4], certificate=certificate)
                    save_certificate(data, destination/'certified'/key)
                    print(json.dumps({'accepted': key, 'weight': maximum, 'branch': args.branch}), flush=True)
                if time.monotonic()-started > args.seconds:
                    break
            if (index+1)%32 == 0:
                journal.flush()
                print(json.dumps(dict(index=index+1, totals=dict(counts), seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(status='complete_bounded_pilot', parameters=vars(args), totals=dict(counts),
                   ranks=dict(rank_counts), commutant_dimensions=dict(commutants),
                   unique_fixed_inputs=len(fixed_seen), unique_row_spaces=len(seen), elapsed_seconds=time.monotonic()-started,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('sparse_*.py')})
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'ranks')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('extension', 'abelian', 'gl', 'control'), required=True)
    parser.add_argument('--maxweight', type=int, required=True)
    parser.add_argument('--fixed', type=int, default=1024)
    parser.add_argument('--per-fixed', type=int, default=96)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
