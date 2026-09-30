"""Exact sparse-partner slices and bounded neighborhoods around d=8 seeds.

Old search drivers/certificates are untouched. A rank-24 fixed left block
allows the full-rank condition to be imposed linearly on the partner block.
Every completed slice enumerates its entire coefficient kernel before
filtering by actual lifted check weight. No sampled absence is called a proof.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from sparse_core import (HERE, basis, combine, common_h, commute, grid_module,
                         permute, rank, reduce, same_space, save_certificate, space_key)
from sparse_twisted import GROUP, checks, equation, fold_catalog
from sparse_distance_upgrade import weight_seven_search
from algebra import bits, low_logical, nullspace, transpose
from search import linear_kernel


SEEDS = ('71e1114cdaa26f7c2bc3cbd2098452d03aee37bfcc62016f59367dc784c7625e',
         '1ceb6dd0cf189fb3a5e0fcc9d52f1f9020a6ab8f2dd560d298fc3a46244b92bc')


def seed_data(key):
    path = HERE/'sparse_campaign/twisted_w12_v1/certified'/key/'candidate.json'
    return json.loads(path.read_text())


def partner_space(a, fold, fixed_side='a'):
    """CSS and, when rank(fixed)=24, exact full-code rank as linear equations.

    If rank(fixed)<24 the containment condition would be too restrictive:
    retain only CSS equations and check the combined rank on actual samples.
    The neighborhood scheduler can defer a large space instead of truncating
    it silently. Rank(fixed)>24 is an exact obstruction for this fixed block.
    """
    matrix = GROUP.lift(a)
    fixed_rank = rank(matrix)
    if fixed_rank > 24:
        return dict(status='fixed_rank_obstruction', fixed_rank=fixed_rank), None
    left = nullspace(transpose(matrix, 32), 32) if fixed_rank == 24 else []
    images = []
    for i in range(32):
        variable = GROUP.lift(1 << i)
        obstruction = sum(combine(variable, u) << (32*j) for j, u in enumerate(left))
        css = equation(a, 1 << i, fold) if fixed_side == 'a' else equation(1 << i, a, fold)
        images.append(css | (obstruction << 32))
    kernel = linear_kernel(images, 32+32*len(left))
    return dict(status='linear_space', fixed_rank=fixed_rank, kernel_dimension=len(kernel),
                kernel=kernel, equation_images=images,
                rank_constraint='exact_rank_24' if left else 'combined_rank_checked_after_enumeration'), kernel


def enumerate_kernel(kernel, ceiling):
    """Complete uint32 enumeration, including a histogram of discarded words."""
    values = np.zeros(1, dtype=np.uint32)
    for v in kernel:
        values = np.concatenate((values, values ^ np.uint32(v)))
    assert len(values) == 1 << len(kernel)
    weights = np.bitwise_count(values)
    hist = np.bincount(weights, minlength=33)
    low = sorted(map(int, values[weights <= ceiling]), key=lambda v: (v.bit_count(), v))
    return low, dict(enumerated=len(values), weight_histogram={w: int(c) for w, c in enumerate(hist) if c},
                     eligible_partners=len(low), partner_weight_ceiling=ceiling)


def short_error(check, stabilizers, n=64):
    """Complete search at weights <=4 using exact stabilizer membership."""
    columns = transpose(check, n)
    pivots = basis(stabilizers)
    singles, pairs = defaultdict(list), defaultdict(list)
    for q, syndrome in enumerate(columns):
        if syndrome == 0 and reduce(1 << q, pivots):
            return [q]
        singles[syndrome].append(q)
    for a, b in itertools.combinations(range(n), 2):
        word = (1 << a) | (1 << b)
        syndrome = columns[a] ^ columns[b]
        if syndrome == 0 and reduce(word, pivots):
            return [a, b]
        for c in singles.get(syndrome, ()):
            if c != a and c != b and reduce(word ^ (1 << c), pivots):
                return sorted((a, b, c))
        for prior in pairs[syndrome]:
            combined = prior ^ word
            if combined.bit_count() == 4 and reduce(combined, pivots):
                return list(bits(combined))
        pairs[syndrome].append(word)
    return None


def six_error(check, stabilizers, n=64):
    columns = transpose(check, n)
    pivots = basis(stabilizers)
    triples = defaultdict(list)
    for a, b, c in itertools.combinations(range(n), 3):
        word = (1 << a) | (1 << b) | (1 << c)
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        for prior in triples[syndrome]:
            combined = prior ^ word
            if combined.bit_count() == 6 and reduce(combined, pivots):
                return list(bits(combined))
        triples[syndrome].append(word)
    return None


def assess_eight(hx, hz, fold, cached=()):
    if rank(hx) != 24 or rank(hz) != 24:
        return dict(status='wrong_combined_rank'), None
    assert commute(hx, hz)
    stabilizers = basis(hz)
    for word in cached:
        if commute(hx, [word]) and reduce(word, stabilizers):
            return dict(status='short_logical_cached', sector='Z', witness=list(bits(word))), None
    witness = short_error(hx, hz)
    if witness is not None:
        return dict(status='short_logical', sector='Z', witness=witness), None
    px, py, central = GROUP.physical()
    info, grid = grid_module(hx, hz, px, py, central)
    if grid is None:
        return info, None
    witness = low_logical(hx, grid['x'], 64)
    if witness is not None:
        return dict(info, status='logical_weight_five', sector='Z', witness=witness), None
    witness = six_error(hx, hz)
    if witness is not None:
        return dict(info, status='logical_weight_six', sector='Z', witness=witness), None
    seven = weight_seven_search(hx, hz)
    if seven['status'] == 'logical_witness':
        return dict(info, status='logical_weight_seven', sector='Z', witness=seven['support']), None
    hi, logical = common_h(hx, hz, grid, fold)
    if logical is None:
        return dict(info, status='h_not_certified', distance_z_lower_bound=8, hadamard=hi), None
    # The verified two-way ZX fold proves equality of X/Z distances. The
    # exported independent audit nevertheless rechecks both sectors directly.
    return dict(info, status='accepted', distance_lower_bound=8, hadamard=hi), {**logical, 'fold': fold}


def export_hit(data, destination):
    audit = save_certificate(data, destination)
    for side, h, s in (('Z', data['hx'], data['hz']), ('X', data['hz'], data['hx'])):
        assert audit['distance'][side]['lower_bound'] >= 7
        seven = weight_seven_search(h, s)
        assert seven['status'] == 'excluded'
        audit['distance'][side].update(lower_bound=8, weight_seven_exclusion=seven)
    (Path(destination)/'full_audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def configurations(phase):
    out = []
    seen = set()
    catalog = list(fold_catalog())
    for key in SEEDS:
        data = seed_data(key)
        original_fold = data['certificate']['fold']
        for side in ('a', 'b'):
            original = data[side]
            choices = [(original, 'unchanged')]
            if phase != 'fixed':
                occupied = list(bits(original))
                empty = [i for i in range(32) if not original >> i & 1]
                choices += [(original ^ (1 << i) ^ (1 << j), 'one_support_swap') for i in occupied for j in empty]
                choices += [(original ^ (1 << i) ^ (1 << j), 'two_support_deletions') for i, j in itertools.combinations(occupied, 2)]
            for fixed, mutation in choices:
                folds = [original_fold]
                if phase == 'folds':
                    if mutation != 'unchanged':
                        continue
                    folds = catalog
                if phase == 'neighborhood' and mutation == 'unchanged':
                    continue
                for fold in folds:
                    identity = (side, fixed, tuple(fold))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    out.append(dict(seed_id=key, fixed_side=side, fixed=fixed, fold=fold,
                                    mutation=mutation, fixed_weight=fixed.bit_count()))
    # Favor changing the smaller coefficient and both seeds before expensive
    # larger-support neighborhoods. Stable ordering makes coverage reproducible.
    out.sort(key=lambda c: (c['fixed_weight'], c['mutation'], c['seed_id'], c['fixed_side'], c['fixed'], c['fold']))
    return out


def run(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    started = time.monotonic()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), HERE/'sparse_core.py', HERE/'sparse_twisted.py', HERE/'sparse_models.py', HERE/'sparse_distance_upgrade.py')}
    configs = configurations(args.phase)
    cache, seen = [], {}
    totals, slices, counters = Counter(), [], Counter()
    with (destination/'candidates.jsonl').open('x') as journal, (destination/'slices.jsonl').open('x') as slice_journal:
        for number, config in enumerate(configs):
            if time.monotonic()-started > args.seconds or counters['completed_slices'] >= args.max_slices:
                counters['campaign_cap_reached'] += 1
                break
            fixed, fold = config['fixed'], config['fold']
            sid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            record = dict(config, slice_id=sid, index=number)
            ceiling = args.maxweight-config['fixed_weight']
            if ceiling < 0:
                record['status'] = 'fixed_weight_obstruction'
            else:
                space, kernel = partner_space(fixed, fold, config['fixed_side'])
                record.update(space)
                if kernel is not None:
                    if len(kernel) > args.max_dimension:
                        record['status'] = 'deferred_large_kernel'
                    else:
                        eligible, summary = enumerate_kernel(kernel, ceiling)
                        record.update(summary)
                        counts = Counter()
                        record['status'] = 'complete_slice'
                        for variable in eligible:
                            if time.monotonic()-started > args.seconds:
                                record['status'] = 'partial_slice_time_cap'
                                break
                            a, b = (fixed, variable) if config['fixed_side'] == 'a' else (variable, fixed)
                            hx, hz = checks(a, b, fold)
                            assert max(v.bit_count() for v in hx+hz) <= args.maxweight
                            code_id = space_key(hx, hz)
                            code_fold = (code_id, tuple(fold))
                            candidate = dict(slice_id=sid, a=a, b=b, code_id=code_id, max_check_weight=a.bit_count()+b.bit_count())
                            if code_fold in seen:
                                candidate.update(status='duplicate_code_fold', reference=seen[code_fold])
                            else:
                                result, certificate = assess_eight(hx, hz, fold, cache)
                                candidate.update(result)
                                seen[code_fold] = dict(slice_id=sid, a=a, b=b, status=result['status'])
                                if 'witness' in result:
                                    word = sum(1 << q for q in result['witness'])
                                    assert commute(hx, [word]) and reduce(word, basis(hz))
                                    if word not in cache:
                                        cache.append(word)
                                        cache = cache[-256:]
                                if certificate is not None:
                                    px, py, central = GROUP.physical()
                                    data = dict(candidate, n=64, k=16, hx=hx, hz=hz, px=px, py=py, central=central,
                                                logical_shape=[4, 4], certificate=certificate,
                                                construction='fold_constrained_abelian', source_seed=config['seed_id'])
                                    target = destination/'certified'/f'{code_id}_{sid[:8]}'
                                    export_hit(data, target)
                                    print(json.dumps(dict(hit=str(target), weight=candidate['max_check_weight'])), flush=True)
                            journal.write(json.dumps(candidate)+'\n')
                            counts[candidate['status']] += 1
                            totals[candidate['status']] += 1
                        record['candidate_counts'] = dict(counts)
                        record['processed_eligible_partners'] = sum(counts.values())
                        if record['status'] == 'complete_slice':
                            assert record['processed_eligible_partners'] == record['eligible_partners']
                            counters['completed_slices'] += 1
            counters[record['status']] += 1
            slices.append(record)
            slice_journal.write(json.dumps(record)+'\n')
            slice_journal.flush()
            journal.flush()
            print(json.dumps(dict(slice=number+1, slices_available=len(configs), fixed_weight=config['fixed_weight'],
                                  status=record['status'], kernel_dimension=record.get('kernel_dimension'),
                                  eligible=record.get('eligible_partners'), counts=record.get('candidate_counts'),
                                  seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(status='complete_configured_scope' if len(slices) == len(configs) and not counters['partial_slice_time_cap'] else 'bounded_scope',
                   parameters=vars(args), configurations_available=len(configs), configurations_processed=len(slices),
                   slice_counts=dict(counters), candidate_counts=dict(totals), unique_code_fold_pairs=len(seen),
                   elapsed_seconds=time.monotonic()-started, source_sha256=source_hashes,
                   coverage_note='Each complete_slice covers all partners within its weight ceiling; deferred/partial slices do not.')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fixed', 'neighborhood', 'folds'), required=True)
    parser.add_argument('--maxweight', type=int, default=10)
    parser.add_argument('--max-dimension', type=int, default=24)
    parser.add_argument('--max-slices', type=int, default=10000)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
