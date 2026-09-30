"""Exact joint coefficient/fold classification for the folded C8 x C4 family.

Distance floor is explicit (six or eight). Fresh, checkpointed run directories
preserve older searches. Coordinate normalizations always conjugate the fold.
"""
import argparse
from collections import Counter, OrderedDict
import hashlib
import itertools
import json
from pathlib import Path
import time

from d8_partner_search import (HERE, GROUP, SEEDS, assess_eight, checks,
                               enumerate_kernel, export_hit, partner_space,
                               seed_data, short_error)
from d8_local_complete import fixed_distance_obstruction, sparse_supports
from d8_shape_pilot import canonical_translation
from sparse_core import (common_h, grid_module, save_certificate, space_key)
from sparse_twisted import fold_catalog
from sparse_distance_upgrade import weight_seven_search
from algebra import (basis, bits, commute, compose, low_logical, permute, rank,
                     reduce, same_space)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def translate(word, shift):
    return sum(1 << GROUP.mul(g, shift) for g in bits(word))


def automorphisms():
    return sorted({tuple(v % 32 for v in p[:32]) for p in GROUP.folds()})


def affine_permutation(alpha, shifts=(0, 0)):
    return [32*s+GROUP.mul(alpha[g], shifts[s]) for s in range(2) for g in range(32)]


def transform_pair(a, b, fold, alpha, shifts=(0, 0)):
    """Transport coefficients and fold under old-to-new physical coordinates."""
    q = affine_permutation(alpha, shifts)
    inverse = [q.index(i) for i in range(64)]
    aa = translate(permute(a, alpha), GROUP.inverse[shifts[0]])
    bb = translate(permute(b, alpha), GROUP.inverse[shifts[1]])
    pp = compose(q, compose(fold, inverse))
    return aa, bb, pp, q


def affine_folds():
    """All common-involution affine swapping folds with square 1 or Tx^4."""
    for theta in automorphisms():
        if any(theta[theta[g]] != g for g in range(32)):
            continue
        for central in (0, 16):
            assert theta[central] == central
            for t0 in range(32):
                t1 = GROUP.mul(central, GROUP.inverse[theta[t0]])
                yield [32+GROUP.mul(theta[g], t0) for g in range(32)] + [
                    GROUP.mul(theta[g], t1) for g in range(32)]


def gauge_fold(a, b, fold, fixed_side):
    """Remove the first offset by moving only the variable-coefficient sheet."""
    t0 = fold[0] % 32
    theta = [GROUP.mul(p % 32, GROUP.inverse[t0]) for p in fold[:32]]
    shifts = (0, GROUP.inverse[t0]) if fixed_side == 'a' else (theta[t0], 0)
    result = transform_pair(a, b, fold, list(range(32)), shifts)
    assert result[2][0] == 32
    assert result[0 if fixed_side == 'a' else 1] == (a if fixed_side == 'a' else b)
    return result


def make_manifest(minimum_distance):
    """Full weight-four shape classification, with recoverable normalizers."""
    auts = automorphisms()
    classes = {canonical_translation(1 | sum(1 << g for g in triple))
               for triple in itertools.combinations(range(1, 32), 3)}
    proofs = {w: fixed_distance_obstruction(w, maximum=minimum_distance-1)
              for w in sorted(classes)}
    survivors = {w for w, proof in proofs.items() if proof is None}
    remaining, orbits, normalizers = survivors.copy(), [], {}
    while remaining:
        representative = min(remaining)
        orbit = set()
        for alpha in auts:
            moved = permute(representative, alpha)
            word = canonical_translation(moved)
            orbit.add(word)
            t = next(t for t in range(32) if translate(moved, t) == word)
            inverse = [alpha.index(i) for i in range(32)]
            # word = translate(alpha(representative), t); the inverse
            # automorphism and a physical shift inverse(t) recover rep.
            shift = inverse[t]
            assert translate(permute(word, inverse), GROUP.inverse[shift]) == representative
            normalizers.setdefault(str(word), dict(representative=representative,
                                                   automorphism=inverse, fixed_sheet_shift=shift))
        assert orbit <= remaining
        remaining -= orbit
        orbits.append(dict(representative=representative, translated_classes=sorted(orbit)))
    small = {word: fixed_distance_obstruction(word, maximum=minimum_distance-1)
             for word in [0]+[1 | (1 << g) for g in range(1, 32)]}
    assert all(small.values())
    # In this commutative exponent-eight algebra, f^8 = epsilon(f)*1.
    # Thus any odd coefficient is a unit, with inverse f^7 and lift rank 32.
    assert all(GROUP.mul(i, j) == GROUP.mul(j, i) for i in range(32) for j in range(32))
    for g in range(32):
        v = 0
        for _ in range(8):
            v = GROUP.mul(v, g)
        assert v == 0
    folds = list(fold_catalog())
    return dict(minimum_distance=minimum_distance, weight=4, anchored_supports=4495,
                translation_classes=len(classes), group_automorphisms=len(auts),
                surviving_translation_classes=len(survivors), shape_classes=len(orbits),
                orbits=orbits, normalizers=normalizers,
                obstructions={w: p for w, p in proofs.items() if p is not None},
                zero_and_weight_two_obstructions=small,
                odd_weight_obstruction='f^8=epsilon(f)*1, so odd f has inverse f^7 and rank 32',
                allowed_positive_weight_splits=[[4, 4], [4, 6], [6, 4]],
                folds=folds, affine_fold_count=3072,
                normalized_fold_count=len(folds), fixed_sides=['a', 'b'],
                raw_slices=len(orbits)*len(folds)*2,
                scope='Two-sheet C8xC4; common involutive affine swapping fold; P^2 in {1,Tx^4}; displayed weight <=10')


def normalize_pair(a, b, fold, fixed_side, manifest):
    word = a if fixed_side == 'a' else b
    canonical = canonical_translation(word)
    t = next(t for t in range(32) if translate(word, t) == canonical)
    shifts = [0, 0]
    index = 0 if fixed_side == 'a' else 1
    shifts[index] = GROUP.inverse[t]
    a, b, fold, q1 = transform_pair(a, b, fold, list(range(32)), shifts)
    normalizer = manifest['normalizers'][str(canonical)]
    shifts[index] = normalizer['fixed_sheet_shift']
    a, b, fold, q2 = transform_pair(a, b, fold, normalizer['automorphism'], shifts)
    a, b, fold, q3 = gauge_fold(a, b, fold, fixed_side)
    assert (a if fixed_side == 'a' else b) == normalizer['representative']
    assert fold in manifest['folds']
    return a, b, fold, compose(q3, compose(q2, q1))


def configurations(manifest):
    for orbit in manifest['orbits']:
        for fold_index, fold in enumerate(manifest['folds']):
            for side in ('a', 'b'):
                yield dict(fixed=orbit['representative'], fixed_side=side,
                           fixed_weight=4, fold=fold, fold_index=fold_index)


def config_key(config):
    return digest({k: config[k] for k in ('fixed', 'fixed_side', 'fold')})


def assess_six(hx, hz, fold, cached=()):
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
    hi, logical = common_h(hx, hz, grid, fold)
    if logical is None:
        return dict(info, status='h_not_certified', distance_z_lower_bound=6, hadamard=hi), None
    return dict(info, status='accepted', distance_lower_bound=6, hadamard=hi), {**logical, 'fold': fold}


def export_six(data, destination):
    """Independent audit, opportunistic exact upgrade, and check optimization."""
    from check_weight_reduction import enumerate_space
    audit = save_certificate(data, destination)
    for side, h, s in (('Z', data['hx'], data['hz']), ('X', data['hz'], data['hx'])):
        distance = audit['distance'][side]
        if distance['lower_bound'] >= 7:
            seven = weight_seven_search(h, s)
            distance['weight_seven_search'] = seven
            distance['lower_bound'] = 8 if seven['status'] == 'excluded' else 7
            if seven['status'] == 'logical_witness':
                distance['upper_bound'] = 7
    (destination/'full_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    ceiling = max(v.bit_count() for v in data['hx']+data['hz'])
    weights = {side: enumerate_space(data[side], ceiling=ceiling) for side in ('hx', 'hz')}
    (destination/'weights.json').write_text(json.dumps(weights, indent=2)+'\n')
    return audit


def load_prior(folders, minimum_distance, maxweight):
    prior = {}
    # Older d=8 runs can be reused at d=6 only if ALL their evaluations were
    # rejected for reasons already incompatible with d>=6. No d=6/7 witness
    # may be carried into a lowered-distance campaign as a rejection.
    universal = {'short_logical', 'short_logical_cached', 'wrong_logical_relations',
                 'wrong_combined_rank', 'duplicate_code_fold', 'logical_weight_five',
                 'norm_zero', 'not_automorphism', 'h_not_certified'}
    for folder in folders:
        folder = Path(folder).resolve()
        summary = json.loads((folder/'summary.json').read_text())
        if summary['parameters']['maxweight'] < maxweight:
            continue
        previous_distance = summary['parameters'].get('min_distance', 8)
        if previous_distance > minimum_distance and not set(summary['candidate_counts']) <= universal:
            continue
        for line in (folder/'slices.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['status'] == 'complete_slice':
                assert row['eligible_partners'] == row['processed_eligible_partners']
                if row['candidate_counts'].get('accepted', 0):
                    continue
                prior[config_key(row)] = dict(prior=str(folder/'slices.jsonl'),
                                              prior_slice_id=row['slice_id'])
    return prior


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = make_manifest(args.min_distance)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    configs = list(configurations(manifest))
    prior = load_prior(args.covered, args.min_distance, args.maxweight)
    started = time.monotonic()
    slices, totals = Counter(), Counter()
    cache, seen, enumeration_cache = [], OrderedDict(), OrderedDict()
    assessed = processed = 0
    assess = assess_six if args.min_distance == 6 else assess_eight
    print(json.dumps(dict(configurations=len(configs), prior_slices=len(prior),
                          minimum_distance=args.min_distance)), flush=True)
    with (output/'slices.jsonl').open('x') as slice_stream, (output/'candidates.jsonl').open('x') as journal:
        for index, config in enumerate(configs):
            if time.monotonic()-started >= args.seconds or assessed >= args.max_slices:
                break
            sid = config_key(config)
            record = dict(config, index=index, slice_id=sid)
            if sid in prior:
                record.update(status='covered_by_prior_certificate', **prior[sid])
            else:
                space, kernel = partner_space(config['fixed'], config['fold'], config['fixed_side'])
                assert kernel is not None
                record.update(space)
                cap = args.maxweight-4
                ek = tuple(sorted(basis(kernel).values())), cap
                if ek not in enumeration_cache:
                    if len(kernel) <= 22:
                        eligible, info = enumerate_kernel(kernel, cap)
                        info['method'] = 'complete_kernel_enumeration'
                    else:
                        eligible, info = sparse_supports(space['equation_images'], cap)
                    enumeration_cache[ek] = eligible, info
                    if len(enumeration_cache) > 8:
                        enumeration_cache.popitem(last=False)
                eligible, info = enumeration_cache[ek]
                enumeration_cache.move_to_end(ek)
                record.update(info)
                record['status'] = 'complete_slice'
                counts = Counter()
                for v in eligible:
                    if time.monotonic()-started >= args.seconds:
                        record['status'] = 'partial_slice_time_cap'
                        break
                    a, b = (config['fixed'], v) if config['fixed_side'] == 'a' else (v, config['fixed'])
                    hx, hz = checks(a, b, config['fold'])
                    code_id = space_key(hx, hz)
                    key = code_id, config['fold_index']
                    entry = dict(slice_id=sid, a=a, b=b, code_id=code_id,
                                 max_check_weight=a.bit_count()+b.bit_count())
                    if key in seen:
                        entry.update(status='duplicate_code_fold', reference=seen[key])
                        seen.move_to_end(key)
                    else:
                        result, certificate = assess(hx, hz, config['fold'], cache)
                        entry.update(result)
                        seen[key] = dict(slice_id=sid, a=a, b=b, status=result['status'])
                        if len(seen) > 100000:
                            seen.popitem(last=False)
                        if 'witness' in result:
                            word = sum(1 << q for q in result['witness'])
                            assert commute(hx, [word]) and reduce(word, basis(hz))
                            assert word.bit_count() < args.min_distance
                            if word not in cache:
                                cache.append(word)
                                cache = cache[-256:]
                        if certificate is not None:
                            px, py, central = GROUP.physical()
                            data = dict(entry, n=64, k=16, hx=hx, hz=hz, px=px, py=py,
                                        central=central, logical_shape=[4, 4], certificate=certificate,
                                        construction='joint_fold_constrained_abelian')
                            target = output/'certified'/f'{code_id}_{sid[:8]}'
                            if not target.exists():
                                audit = export_six(data, target) if args.min_distance == 6 else export_hit(data, target)
                            entry['certificate_path'] = str(target)
                            print(json.dumps(dict(hit=str(target), weight=entry['max_check_weight'])), flush=True)
                    journal.write(json.dumps(entry)+'\n')
                    counts[entry['status']] += 1
                    totals[entry['status']] += 1
                record.update(candidate_counts=dict(counts), processed_eligible_partners=sum(counts.values()))
                if record['status'] == 'complete_slice':
                    assert record['eligible_partners'] == record['processed_eligible_partners']
                    assessed += 1
            slice_stream.write(json.dumps(record)+'\n')
            slice_stream.flush()
            journal.flush()
            slices[record['status']] += 1
            processed += 1
            if index % 16 == 0 or record['status'] != 'covered_by_prior_certificate':
                print(json.dumps(dict(index=index+1, total=len(configs), status=record['status'],
                                      eligible=record.get('eligible_partners'),
                                      candidate_counts=record.get('candidate_counts'),
                                      seconds=round(time.monotonic()-started, 2))), flush=True)
    complete = processed == len(configs) and not slices['partial_slice_time_cap']
    summary = dict(status='complete_joint_scope' if complete else 'bounded_scope',
                   parameters=vars(args), configurations=len(configs), configurations_processed=processed,
                   slice_counts=dict(slices), candidate_counts=dict(totals),
                   elapsed_seconds=time.monotonic()-started,
                   duplicate_cache_limit=100000, manifest_sha256=digest(manifest),
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), HERE/'d8_partner_search.py', HERE/'d8_local_complete.py',
                                   HERE/'sparse_core.py', HERE/'sparse_twisted.py', HERE/'sparse_models.py')})
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-distance', type=int, choices=(6, 8), default=6)
    parser.add_argument('--maxweight', type=int, choices=(8, 10), default=10)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--max-slices', type=int, default=10000)
    parser.add_argument('--covered', action='append', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
