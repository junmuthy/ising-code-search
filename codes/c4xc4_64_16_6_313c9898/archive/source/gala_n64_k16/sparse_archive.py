"""Audit components of all 93 saved distance-six parent presentations."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

from sparse_core import (HERE, assess, basis, canonical_x, combine, common_h,
                         compose, grid_module, pairing, permute, power,
                         quotient_basis, rank, rref, save_certificate, space_key)
from extract_component import components
from check_weight_reduction import enumerate_space
from models import Extension


JOURNAL = HERE.parent/'gala_n128_k32/runs/extension_products_w16/candidates.jsonl'


def component_data(record, groups):
    group = groups[tuple(record['extension'])]
    hx, hz = group.checks(record['a'], record['b'])
    comps = components(hx+hz, 128)
    if sorted(map(len, comps)) != [64, 64]:
        return None
    component = next(c for c in comps if 0 in c)
    index = {q: i for i, q in enumerate(component)}

    def project(v):
        return sum(((v >> q) & 1) << i for i, q in enumerate(component))

    def restrict(p):
        if any(p[q] not in index for q in component):
            return None
        return [index[p[q]] for q in component]

    checks = [[project(v) for v in rows if project(v)] for rows in (hx, hz)]
    rights = [(g, restrict(group.right(g))) for g in range(64)]
    preserving = [(g, p) for g, p in rights if p is not None]
    # Record an explicit permutation equivalence between the two components.
    exchange = next(g for g, p in rights if p is None)
    assert {group.right(exchange)[q] for q in component} == set(next(c for c in comps if c is not component))
    folds = []
    seen = set()
    for i, p in enumerate(group.folds()):
        for g in (0, exchange):
            fixed = restrict(compose(group.right(g), p))
            if fixed is not None and tuple(fixed) not in seen:
                seen.add(tuple(fixed))
                folds.append((i, g, fixed))
    return dict(n=64, k=16, hx=checks[0], hz=checks[1],
                central=restrict(group.right(1)), parent_id=record['id'],
                parent_coordinates=component, extension=record['extension'],
                parent_a=record['a'], parent_b=record['b'],
                component_exchange_right_element=exchange), preserving, folds


def choose_grid(data, rights):
    hx, hz = data['hx'], data['hz']
    if rank(hx) != 24 or rank(hz) != 24:
        return {'status': 'wrong_component_ranks'}, None
    z = quotient_basis(hx, hz, 64)
    x = canonical_x(z, quotient_basis(hz, hx, 64))
    identity = [1 << i for i in range(16)]
    unique = {}
    for g, p in rights:
        action = pairing([permute(v, p) for v in z], x)
        a2 = [combine(action, v) for v in action]
        a4 = [combine(a2, v) for v in a2]
        if a4 == identity and a2 != identity:
            unique.setdefault(tuple(action), (g, p))
    statuses = Counter()
    for (g, px), (h, py) in itertools.combinations(unique.values(), 2):
        info, grid = grid_module(hx, hz, px, py, data['central'])
        statuses[info['status']] += 1
        if grid is not None:
            data.update(px=px, py=py, logical_shape=[4, 4], translation_right_elements=[g, h])
            return {'status': 'regular_grid', 'pair_statuses': dict(statuses)}, grid
    return {'status': 'no_regular_grid_in_preserving_right_action', 'pair_statuses': dict(statuses),
            'distinct_order_four_logical_actions': len(unique)}, None


def run(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    records = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    records = [r for r in records if r['status'] in ('accepted', 'h_not_certified')]
    records.sort(key=lambda r: (r['max_check_weight'], r['id']))
    groups = {key: Extension(*key) for key in itertools.product(range(2), repeat=2)}
    seen = {}
    totals = Counter()
    started = time.monotonic()
    with (destination/'components.jsonl').open('x') as stream:
        for index, record in enumerate(records):
            extracted = component_data(record, groups)
            result = dict(parent_id=record['id'], parent_status=record['status'], parent_check_weight=record['max_check_weight'])
            if extracted is None:
                result['status'] = 'different_component_sizes'
            else:
                data, rights, folds = extracted
                key = space_key(data['hx'], data['hz'])
                result['row_space_id'] = key
                if key in seen:
                    result.update(status='duplicate_fixed_coordinate_spaces', duplicate_of=seen[key])
                else:
                    seen[key] = record['id']
                    info, grid = choose_grid(data, rights)
                    result.update(info)
                    if grid is not None:
                        fstatus = Counter()
                        for fold_index, offset, p in folds:
                            hi, logical = common_h(data['hx'], data['hz'], grid, p)
                            fstatus[hi['status']] += 1
                            if logical is not None:
                                data.update(certificate={**logical, 'fold': p}, fold_index=fold_index,
                                            fold_right_offset=offset, hadamard=hi)
                                result.update(status='accepted', hadamard=hi)
                                break
                        else:
                            result['status'] = 'h_not_certified'
                        result['fold_statuses'] = dict(fstatus)
                    # Enumerate weights even for failed gate cases; preserve useful near misses.
                    enum = {side: enumerate_space(data[side], ceiling=record['max_check_weight']) for side in ('hx', 'hz')}
                    result['weight_optimum'] = {side: {k: v for k, v in e.items() if k not in ('low_weight_rows', 'enumeration_basis')}
                                                for side, e in enum.items()}
                    (destination/f'{key}.json').write_text(json.dumps(data, indent=2)+'\n')
                    if result['status'] == 'accepted':
                        audited = save_certificate(data, destination/'certified'/key)
                        result['distance'] = audited['distance']
            totals[result['status']] += 1
            stream.write(json.dumps(result)+'\n')
            stream.flush()
            print(json.dumps(dict(index=index+1, total=len(records), weight=record['max_check_weight'],
                                  status=result['status'], totals=dict(totals), seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(status='complete', parent_records=len(records), unique_fixed_coordinate_spaces=len(seen),
                   totals=dict(totals), elapsed_seconds=time.monotonic()-started,
                   source_journal=str(JOURNAL), source_sha256=hashlib.sha256(JOURNAL.read_bytes()).hexdigest(),
                   scope='one representative of each explicitly right-translation-equivalent component pair; bounded inherited fold catalog; all preserving right translations')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
