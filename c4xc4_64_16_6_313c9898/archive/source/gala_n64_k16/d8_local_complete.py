"""Complete the specified local neighborhoods using exact sparse enumeration.

Adds a fixed-block distance obstruction and a support meet-in-the-middle path
for large coefficient kernels. Previously completed slices are linked, not
rerun or overwritten. Completeness is only for the explicit configuration list.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from d8_partner_search import (HERE, GROUP, assess_eight, checks, configurations,
                               enumerate_kernel, export_hit, partner_space)
from algebra import basis, bits, commute, nullspace, rank, reduce
from sparse_core import space_key


def fixed_distance_obstruction(fixed, maximum=7):
    """Too many light pure-block kernel vectors for stabilizers to absorb.

    For r=rank(L_fixed) and rank(HZ)=24, the Z stabilizers supported purely
    on this block have dimension 24-r: the other-block projection of HZ is
    a permuted copy of L_fixed. If light vectors in ker(L_fixed) span more
    than 24-r dimensions, at least one is a nontrivial logical <=maximum.
    This proof requires the block-swapping fold used throughout this family.
    """
    matrix = GROUP.lift(fixed)
    r = rank(matrix)
    if r > 24:
        return dict(status='fixed_rank_obstruction', fixed_rank=r)
    kernel = nullspace(matrix, 32)
    light = [v for v in kernel if v.bit_count() <= maximum]
    if rank(light) <= 24-r and len(kernel) <= 20:
        light, _ = enumerate_kernel(kernel, maximum)
        light = [v for v in light if v]
    selected, pivots = [], {}
    for v in light:
        residual = reduce(v, pivots)
        if residual:
            pivots[residual.bit_length()-1] = residual
            selected.append(v)
        if len(selected) > 24-r:
            assert commute(matrix, selected)
            assert all(v.bit_count() <= maximum for v in selected)
            return dict(status='fixed_block_short_logical_obstruction', fixed_rank=r,
                        stabilizer_intersection_dimension=24-r,
                        light_kernel_rank=len(selected), light_kernel_rows=selected,
                        logical_weight_upper_bound=maximum)
    return None


def sparse_supports(images, ceiling):
    """All zero-syndrome coefficient words of weight <=ceiling, via 16+16."""
    buckets = []
    for offset in (0, 16):
        syndromes = [0]
        for image in images[offset:offset+16]:
            syndromes += [v ^ image for v in syndromes]
        by_syndrome = defaultdict(lambda: defaultdict(list))
        for mask, syndrome in enumerate(syndromes):
            weight = mask.bit_count()
            if weight <= ceiling:
                by_syndrome[syndrome][weight].append(mask)
        buckets.append(by_syndrome)
    words = []
    for syndrome, left in buckets[0].items():
        right = buckets[1].get(syndrome, {})
        for w, lows in left.items():
            for u, highs in right.items():
                if w+u <= ceiling:
                    words.extend(lo | (hi << 16) for lo in lows for hi in highs)
    assert len(words) == len(set(words))
    words.sort(key=lambda v: (v.bit_count(), v))
    return words, dict(method='complete_16_plus_16_support_meet_in_middle',
                       half_words_enumerated=2*2**16, eligible_partners=len(words),
                       partner_weight_ceiling=ceiling,
                       eligible_weight_histogram=dict(Counter(v.bit_count() for v in words)))


def run(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    prior = {}
    for folder in args.covered:
        p = Path(folder)
        summary = json.loads((p/'summary.json').read_text())
        assert summary['parameters']['maxweight'] >= args.maxweight
        for line in (p/'slices.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record['status'] in ('complete_slice', 'fixed_rank_obstruction'):
                prior[record['slice_id']] = str(p/'slices.jsonl')
    started = time.monotonic()
    configs = configurations(args.phase)
    totals, statuses, cache, seen = Counter(), Counter(), [], {}
    processed = 0
    with (destination/'slices.jsonl').open('x') as slices, (destination/'candidates.jsonl').open('x') as journal:
        for index, c in enumerate(configs):
            if time.monotonic()-started > args.seconds:
                break
            sid = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
            record = dict(c, index=index, slice_id=sid)
            assert all(p >= 32 for p in c['fold'][:32]) and all(p < 32 for p in c['fold'][32:])
            obstruction = fixed_distance_obstruction(c['fixed']) if sid not in prior else None
            if sid in prior:
                record.update(status='covered_by_prior_certificate', prior=prior[sid])
            elif obstruction:
                record.update(obstruction)
            else:
                space, kernel = partner_space(c['fixed'], c['fold'], c['fixed_side'])
                record.update(space)
                assert kernel is not None
                cap = args.maxweight-c['fixed_weight']
                if cap < 0:
                    record['status'] = 'fixed_weight_obstruction'
                else:
                    if len(kernel) <= 22:
                        eligible, info = enumerate_kernel(kernel, cap)
                        info['method'] = 'complete_kernel_enumeration'
                    else:
                        eligible, info = sparse_supports(space['equation_images'], cap)
                    record.update(info)
                    record['status'] = 'complete_slice'
                    counts = Counter()
                    for v in eligible:
                        if time.monotonic()-started > args.seconds:
                            record['status'] = 'partial_slice_time_cap'
                            break
                        a, b = (c['fixed'], v) if c['fixed_side'] == 'a' else (v, c['fixed'])
                        hx, hz = checks(a, b, c['fold'])
                        code_id = space_key(hx, hz)
                        key = code_id, tuple(c['fold'])
                        entry = dict(slice_id=sid, code_id=code_id, a=a, b=b, max_check_weight=a.bit_count()+b.bit_count())
                        if key in seen:
                            entry.update(status='duplicate_code_fold', reference=seen[key])
                        else:
                            result, certificate = assess_eight(hx, hz, c['fold'], cache)
                            entry.update(result)
                            seen[key] = dict(slice_id=sid, a=a, b=b, status=result['status'])
                            if 'witness' in result:
                                word = sum(1 << q for q in result['witness'])
                                assert commute(hx, [word]) and reduce(word, basis(hz))
                                if word not in cache:
                                    cache.append(word)
                                    cache = cache[-256:]
                            if certificate is not None:
                                px, py, central = GROUP.physical()
                                data = dict(entry, n=64, k=16, hx=hx, hz=hz, px=px, py=py, central=central,
                                            logical_shape=[4, 4], certificate=certificate, source_seed=c['seed_id'])
                                target = destination/'certified'/f'{code_id}_{sid[:8]}'
                                export_hit(data, target)
                                print(json.dumps(dict(hit=str(target), weight=entry['max_check_weight'])), flush=True)
                        journal.write(json.dumps(entry)+'\n')
                        counts[entry['status']] += 1
                        totals[entry['status']] += 1
                    record['candidate_counts'] = dict(counts)
                    record['processed_eligible_partners'] = sum(counts.values())
                    if record['status'] == 'complete_slice':
                        assert sum(counts.values()) == len(eligible)
            slices.write(json.dumps(record)+'\n')
            statuses[record['status']] += 1
            processed += 1
            slices.flush()
            journal.flush()
            if index%16 == 0 or record['status'] not in ('covered_by_prior_certificate', 'fixed_rank_obstruction'):
                print(json.dumps(dict(index=index+1, configurations=len(configs), status=record['status'],
                                      eligible=record.get('eligible_partners'), counts=record.get('candidate_counts'),
                                      seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(status='complete_explicit_neighborhood' if processed == len(configs) and not statuses['partial_slice_time_cap'] else 'bounded_scope',
                   parameters=vars(args), configurations=len(configs), configurations_processed=processed,
                   slice_counts=dict(statuses), candidate_counts=dict(totals), unique_code_fold_pairs=len(seen),
                   elapsed_seconds=time.monotonic()-started,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), HERE/'d8_partner_search.py')})
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fixed', 'neighborhood', 'folds'), required=True)
    parser.add_argument('--maxweight', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--covered', action='append', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
