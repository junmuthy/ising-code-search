"""Exact fixed-code weight audits and independent exports for new orbit hits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from check_weight_reduction import enumerate_space, coordinates
from sparse_core import HERE, permute, same_space, save_certificate, space_key
from sparse_finalize import balanced_basis
from orbit_extension import module_generators
from orbit_campaign import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-cyclic', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, seen = [], set()
    for path in sorted(args.root.glob('*/certified/*/candidate.json')):
        if 'cyclic' in str(path) and not args.include_cyclic:
            continue
        data = json.loads(path.read_text())
        key = space_key(data['hx'], data['hz'])
        if key in seen:
            continue
        seen.add(key)
        folder = args.output/key
        folder.mkdir()
        enum = {s: enumerate_space(data[s], ceiling=12) for s in ('hx', 'hz')}
        assert enum['hx']['weight_histogram'] == enum['hz']['weight_histogram']
        save(folder/'weight_enumeration.json', enum)
        hx, loads = balanced_basis(enum['hx']['low_weight_rows'], data['certificate']['fold'], trials=512, seed=20260914)
        hz = [permute(v, data['certificate']['fold']) for v in hx]
        assert same_space(hx, data['hx']) and same_space(hz, data['hz'])
        optimized = dict(data, hx=hx, hz=hz, presentation='minimum_total_weight_independent')
        audit = save_certificate(optimized, folder/'independent')
        assert audit['metrics']['row_space_component_sizes'] == [64]
        original_audit = save_certificate(data, folder/'original')
        assert original_audit['metrics']['row_space_component_sizes'] == [64]
        save(folder/'basis_changes.json', {s: dict(new_from_original=[coordinates(data[s], v) for v in rows],
              original_from_new=[coordinates(rows, v) for v in data[s]]) for s, rows in (('hx', hx), ('hz', hz))})
        record = dict(id=key, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      minimum_maximum_check_weight=enum['hx']['minimum_possible_maximum_check_weight'],
                      minimum_total_support=sum(e['minimum_basis_total_weight'] for e in enum.values()),
                      checks_per_type=dict(Counter(map(int.bit_count, hx))), load=loads,
                      distance=audit['distance'], fold_order=audit['actions']['H']['physical_order'],
                      physical_translation_orders=[audit['actions'][s]['physical_order'] for s in ('Tx', 'Ty')],
                      minimum_orbit_generators=module_generators(hx, [data['px'], data['py']]))
        records.append(record)
        save(args.output/'report.json', dict(status='running', candidates=records))
        print(json.dumps(record), flush=True)
    save(args.output/'report.json', dict(status='complete', candidates=records,
          note='Distinct fixed-coordinate row spaces; not a complete qubit-permutation equivalence classification.',
          source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                         (Path(__file__).resolve(), HERE/'check_weight_reduction.py', HERE/'sparse_finalize.py', HERE/'sparse_core.py')}))


if __name__ == '__main__':
    main()
