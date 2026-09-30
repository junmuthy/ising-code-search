"""Rank certified low-weight hits and export exact check-weight certificates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from sparse_core import HERE, permute, rank, same_space, save_certificate, unpack
from check_weight_reduction import (coordinates, enumerate_space, greedy_basis,
                                    load_score, orbit_search)


def balanced_basis(words, fold, trials=3000, seed=20260926):
    rng = np.random.default_rng(seed)
    mx = unpack(words, 64)
    mz = unpack([permute(v, fold) for v in words], 64)
    classes = [[i for i, v in enumerate(words) if v.bit_count() == w]
               for w in sorted({v.bit_count() for v in words})]
    best = None
    for _ in range(trials):
        selected = greedy_basis(words, np.concatenate([rng.permutation(c) for c in classes]))
        score = load_score(selected, mx, mz)
        if best is None or score < best[0]:
            best = score, selected
    return [words[i] for i in best[1]], dict(score=best[0], trials=trials, seed=seed,
                                            peak_load_optimality='not proved')


def run(root, destination):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    records, candidates = [], []
    for path in sorted(root.glob('*_v1/certified/*/candidate.json')):
        data = json.loads(path.read_text())
        if max(v.bit_count() for v in data['hx']+data['hz']) > 12:
            continue
        enum = {s: enumerate_space(data[s], ceiling=12) for s in ('hx', 'hz')}
        assert enum['hx']['weight_histogram'] == enum['hz']['weight_histogram']
        key = path.parent.name
        (destination/f'weights_{key}.json').write_text(json.dumps(enum, indent=2)+'\n')
        maximum = max(e['minimum_possible_maximum_check_weight'] for e in enum.values())
        total = sum(e['minimum_basis_total_weight'] for e in enum.values())
        record = dict(id=key, source_candidate=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      minimum_maximum_check_weight=maximum, minimum_total_support=total,
                      minimum_basis_weights=enum['hx']['minimum_basis_weight_counts'])
        records.append(record)
        candidates.append((maximum, total, key, data, enum))
    assert candidates
    optimum = min((v[0], v[1]) for v in candidates)
    best = None
    load_trials = []
    for maximum, total, key, data, enum in candidates:
        if (maximum, total) != optimum:
            continue
        hx, loads = balanced_basis(enum['hx']['low_weight_rows'], data['certificate']['fold'])
        hz = [permute(v, data['certificate']['fold']) for v in hx]
        assert same_space(hx, data['hx']) and same_space(hz, data['hz'])
        assert Counter(v.bit_count() for v in hx) == enum['hx']['minimum_basis_weight_counts']
        load_trials.append(dict(id=key, **loads))
        choice = (*loads['score'], key)
        if best is None or choice < best[0]:
            best = choice, data, enum, hx, hz, key
    _, original, enum, hx, hz, key = best
    reduced = dict(original, hx=hx, hz=hz, presentation='minimum-total-weight independent CSS generators')
    independent = save_certificate(reduced, destination/'best_independent', seconds=15)
    symmetric = save_certificate(original, destination/'best_symmetric', seconds=0)
    change = {s: dict(new_from_original=[coordinates(original[s], v) for v in rows],
                      original_from_new=[coordinates(rows, v) for v in original[s]])
              for s, rows in (('hx', hx), ('hz', hz))}
    actions = {s: {name: [coordinates(rows, permute(v, p)) for v in rows]
                   for name, p in (('Tx', original['px']), ('Ty', original['py']))}
               for s, rows in (('hx', hx), ('hz', hz))}
    actions['H'] = dict(x_to_z=[coordinates(hz, permute(v, original['certificate']['fold'])) for v in hx],
                        z_to_x=[coordinates(hx, permute(v, original['certificate']['fold'])) for v in hz])
    orbits = {}
    for s in ('hx', 'hz'):
        orbits[s] = orbit_search(enum[s]['low_weight_rows'], original['px'], original['py'], original['certificate']['fold'])
        orbits[s]['maximum_check_weight'] = 12
        orbits[s]['row_sector'] = s
    (destination/'best_independent/supports.json').write_text(json.dumps({s: [[i for i in range(64) if v >> i & 1] for v in rows]
                                                                     for s, rows in (('hx', hx), ('hz', hz))}, indent=2)+'\n')
    report = dict(status='certified_weight_reduction_hit', candidates=records, selected_id=key,
                   optimum_among_hits=optimum, load_balancing=load_trials, basis_changes=change,
                   stabilizer_actions=actions, symmetry_closed_search=orbits,
                   independent_audit=independent, symmetric_audit=symmetric,
                   note='Exact weight optima apply to each saved code; candidate list is not classified up to permutations.')
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(selected_id=key, optimum=optimum, candidate_count=len(candidates), load=best[0],
                          symmetry_closed_support=sum(o['minimum_total_weight_per_sector'] for o in orbits.values()))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.root, args.output)
