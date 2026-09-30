"""Exact equivariant logical-support optimization of the fixed distance-eight code.

No check matrix is changed. Logical translation orbits identify equivalent
seed cosets, and the saved ZX fold identifies X costs with Z costs.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sparse_core import (HERE, basis, canonical_x, combine, compose, pairing,
                         permute, permutation_matrix, power, rank, reduce, unpack)
from certify import gf_rank, in_space, moved, permutation_order
from distance8_logical_handoff import simplify


def logical_shifts():
    return [[4*((i+a)%4)+(j+b)%4 for i in range(4) for j in range(4)]
            for a in range(4) for b in range(4)]


def unit_orbits():
    shifts = logical_shifts()
    remaining = {c for c in range(1 << 16) if c.bit_count()%2}
    result, canonical = [], {}
    while remaining:
        seed = min(remaining)
        orbit = sorted({permute(seed, p) for p in shifts})
        assert len(orbit) == 16 and set(orbit) <= remaining
        remaining.difference_update(orbit)
        result.append(dict(seed=seed, members=orbit))
        canonical.update((c, seed) for c in orbit)
    assert len(result) == 2048 and len(canonical) == 32768
    return result, canonical


def physical_shifts(data):
    return [compose(power(data['px'], i), power(data['py'], j))
            for i in range(4) for j in range(4)]


def seed_basis(data, seed):
    z = [permute(combine(data['certificate']['z'], seed), p) for p in physical_shifts(data)]
    x = canonical_x(z, data['certificate']['x'])
    assert pairing(x, z) == [1 << i for i in range(16)]
    return x, z


def h_action(x, z, fold):
    zx = pairing([permute(v, fold) for v in z], z)
    xz = pairing([permute(v, fold) for v in x], x)
    if zx == xz and permutation_matrix(zx):
        return [v.bit_length()-1 for v in zx]
    return None


def make_catalog(data):
    orbits, canonical = unit_orbits()
    old_x = data['certificate']['x']
    fold = data['certificate']['fold']
    result = []
    for orbit in orbits:
        x, z = seed_basis(data, orbit['seed'])
        # P takes this dual X seed to a Z coset with exactly the same minimum.
        partner = pairing([permute(x[0], fold)], old_x)[0]
        assert partner in canonical
        result.append(dict(**orbit, dual_Z_seed=canonical[partner],
                           saved_hadamard_permutation=h_action(x, z, fold)))
    return result, canonical


class CosetEnumerator:
    def __init__(self, stabilizers):
        self.generators = list(basis(stabilizers).values())
        self.halves = []
        for part in (self.generators[:len(self.generators)//2], self.generators[len(self.generators)//2:]):
            values = np.zeros(1, dtype=np.uint64)
            for row in part:
                values = np.concatenate((values, values ^ np.uint64(row)))
            self.halves.append(values)

    def scan(self, representative, collect_weight=None, chunk=32):
        histogram = np.zeros(65, dtype=np.int64)
        minimum, witness, collected = 65, None, []
        for start in range(0, len(self.halves[1]), chunk):
            words = self.halves[1][start:start+chunk, None] ^ self.halves[0][None, :] ^ np.uint64(representative)
            weights = np.bitwise_count(words)
            histogram += np.bincount(weights.ravel(), minlength=65)
            best = int(weights.min())
            if best <= minimum:
                candidate = int(words[weights == best].min())
                if best < minimum or candidate < witness:
                    minimum, witness = best, candidate
            if collect_weight is not None:
                collected.extend(map(int, words[weights == collect_weight]))
        assert int(histogram.sum()) == 2**len(self.generators)
        assert reduce(witness ^ representative, basis(self.generators)) == 0
        return dict(minimum_weight=minimum, minimum_word=witness,
                    enumerated=int(histogram.sum()), histogram={i: int(c) for i, c in enumerate(histogram) if c}), sorted(collected)


def audit_basis(data, original, expected_weight):
    """Independent array-based audit of the unchanged code and new logical basis."""
    assert data['hx'] == original['hx'] and data['hz'] == original['hz']
    hx, hz = unpack(data['hx'], 64), unpack(data['hz'], 64)
    x, z = unpack(data['certificate']['x'], 64), unpack(data['certificate']['z'], 64)
    assert gf_rank(hx) == gf_rank(hz) == 24
    assert not np.any(hx @ hz.T % 2)
    assert not np.any(x @ hz.T % 2) and not np.any(z @ hx.T % 2)
    assert np.array_equal(x @ z.T % 2, np.eye(16, dtype=np.uint8))
    assert np.all(x.sum(axis=1) == expected_weight) and np.all(z.sum(axis=1) == expected_weight)
    ones = np.ones((1, 64), dtype=np.uint8)
    assert in_space(ones, hx) and in_space(ones, hz)
    actions = {}
    for name, key, target in (('Tx', 'px', logical_shifts()[4]), ('Ty', 'py', logical_shifts()[1])):
        p = data[key]
        for rows, checks in ((x, hx), (z, hz)):
            assert in_space(moved(checks, p), checks)
            assert in_space(moved(rows, p) ^ rows[target], checks)
        actions[name] = dict(physical_order=permutation_order(p), logical_permutation=target)
    for rows, checks in ((x, hx), (z, hz)):
        assert in_space(moved(rows, data['central']) ^ rows, checks)
    p = data['certificate']['fold']
    assert in_space(moved(hx, p), hz) and in_space(moved(hz, p), hx)
    zx, xz = moved(z, p) @ z.T % 2, moved(x, p) @ x.T % 2
    assert np.array_equal(zx, xz)
    assert np.all(zx.sum(axis=0) == 1) and np.all(zx.sum(axis=1) == 1)
    sigma = np.argmax(zx, axis=1).tolist()
    assert sigma == data['hadamard']['logical_permutation']
    assert in_space(moved(z, p) ^ x[sigma], hx)
    assert in_space(moved(x, p) ^ z[sigma], hz)
    combined = np.vstack((x, z))
    overlaps = combined @ combined.T
    np.fill_diagonal(overlaps, 0)
    actions['H'] = dict(physical_order=permutation_order(p), logical_permutation=sigma)
    return dict(status='verified_unchanged_code_new_logical_basis', n=64, k=16,
                exact_distance=8, distance_provenance='Unchanged checks; inherited exact distance certificate',
                x_weights=dict(Counter(map(int, x.sum(axis=1)))),
                z_weights=dict(Counter(map(int, z.sum(axis=1)))),
                total_logical_support=int(combined.sum()),
                combined_qubit_loads=list(map(int, combined.sum(axis=0))),
                maximum_pairwise_support_overlap=int(overlaps.max()), actions=actions)


def export_basis(data, seed, word, fold, destination, z_words=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    raw_x, raw_z = seed_basis(data, seed)
    sigma = h_action(raw_x, raw_z, fold)
    assert sigma is not None
    z = ([permute(word, p) for p in physical_shifts(data)] if z_words is None else list(z_words))
    assert len(z) == 16 and all(reduce(a^b, basis(data['hz'])) == 0 for a, b in zip(z, raw_z))
    # P maps Z_i to X_sigma(i). The ordering works even for non-involutive P.
    x = [0]*16
    for i in range(16):
        x[sigma[i]] = permute(z[i], fold)
    candidate = deepcopy(data)
    candidate['certificate'] = dict(x=x, z=z, fold=fold)
    candidate['hadamard'] = dict(status='logical_hadamard', logical_permutation=sigma,
                                 basis_coefficients=seed)
    candidate['logical_representatives'] = 'Minimum-support equivariant logical basis; code unchanged'
    candidate['logical_basis_change'] = dict(
        Z_from_original_Z=pairing(z, data['certificate']['x']),
        X_from_original_X=pairing(x, data['certificate']['z']))
    report = audit_basis(candidate, data, word.bit_count())
    destination.mkdir(parents=True)
    (destination/'candidate.json').write_text(json.dumps(candidate, indent=2)+'\n')
    (destination/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(destination/'logical_bases.npz', x=unpack(x, 64), z=unpack(z, 64))
    (destination/'logical_supports.json').write_text(json.dumps({s: [[q for q in range(64) if w >> q & 1]
        for w in rows] for s, rows in (('X', x), ('Z', z))}, indent=2)+'\n')
    return report


def run(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    started = time.monotonic()
    data, provenance, _ = simplify()
    catalog, _ = make_catalog(data)
    (destination/'catalog.json').write_text(json.dumps(catalog, indent=2)+'\n')
    eligible = [r for r in catalog if r['saved_hadamard_permutation'] is not None]
    print(json.dumps(dict(stage='saved_fold', unit_orbits=len(catalog), eligible_orbits=len(eligible))), flush=True)
    engine = CosetEnumerator(data['hz'])
    values, hits = {}, []
    with (destination/'cosets.jsonl').open('x') as journal:
        for index, record in enumerate(eligible):
            seed = record['seed']
            result, _ = engine.scan(combine(data['certificate']['z'], seed))
            entry = dict(seed=seed, **result)
            values[seed] = entry
            journal.write(json.dumps(entry)+'\n')
            journal.flush()
            if result['minimum_weight'] <= 10:
                assert result['minimum_weight'] == 10, 'Unexpected weight-eight counterexample needs investigation'
                hits.append(seed)
                if len(hits) == 1:
                    export_basis(data, seed, result['minimum_word'], data['certificate']['fold'], destination/'first_hit')
                    print(json.dumps(dict(first_weight_ten_hit=seed)), flush=True)
            if (index+1)%32 == 0 or index+1 == len(eligible):
                print(json.dumps(dict(processed=index+1, total=len(eligible), hits=len(hits),
                                      seconds=round(time.monotonic()-started, 2))), flush=True)
    result = dict(status='saved_fold_weight_ten_found' if hits else 'saved_fold_weight_ten_excluded',
                  unit_orbits=len(catalog), saved_fold_eligible_orbits=len(eligible),
                  enumerated_cosets=len(values), hit_seeds=hits,
                  minimum_distribution=dict(Counter(r['minimum_weight'] for r in values.values())),
                  source_candidate=provenance['candidate_source'],
                  source_candidate_sha256=provenance['candidate_source_sha256'],
                  source_distance_audit=provenance['distance'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    (destination/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
