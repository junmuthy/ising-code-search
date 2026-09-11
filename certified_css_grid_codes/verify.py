"""Verify the portable bundles using only Python and NumPy.

Default: hashes, construction, arrays, CSS algebra, all logical actions and
distance upper witnesses. --distance additionally proves the lower bounds by
complete sorted-support syndrome matching. No original qLDPC paths are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

if not __debug__:
    raise RuntimeError('Verification requires assertions; do not run Python with -O')

ROOT = Path(__file__).resolve().parent


def pivots(rows):
    result = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            if p not in result:
                result[p] = row
                break
            row ^= result[p]
    return result


def residual(row, space):
    for p in sorted(space, reverse=True):
        if row >> p & 1:
            row ^= space[p]
    return row


def same_space(a, b):
    pa, pb = pivots(a), pivots(b)
    return len(pa) == len(pb) and all(residual(v, pb) == 0 for v in a)


def transpose(rows, width):
    return [sum(((v >> q)&1) << i for i, v in enumerate(rows)) for q in range(width)]


def move(word, permutation):
    return sum(1 << permutation[q] for q in range(len(permutation)) if word >> q & 1)


def compose(p, q):
    return [p[q[i]] for i in range(len(p))]


def order(p):
    assert sorted(p) == list(range(len(p)))
    unseen, cycles = set(range(len(p))), []
    while unseen:
        start = q = min(unseen)
        length = 0
        while q in unseen:
            unseen.remove(q)
            q = p[q]
            length += 1
        assert q == start
        cycles.append(length)
    return math.lcm(*cycles)


def array(rows, width):
    return np.asarray([[(v >> q)&1 for q in range(width)] for v in rows], dtype=np.uint8)


def components(rows, width):
    parent = list(range(width))
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    for word in rows:
        support = [q for q in range(width) if word >> q & 1]
        for q in support[1:]:
            parent[find(q)] = find(support[0])
    return sorted(Counter(find(q) for q in range(width)).values())


def reconstruct(data, kind):
    """Rebuild the checks from the archived algebraic recipe."""
    if kind in ('folded_c8xc4', 'nonabelian_extension'):
        size = 32 if kind == 'folded_c8xc4' else 64
        def multiply(a, b):
            if size == 32:
                i, j = divmod(a, 4)
                p, q = divmod(b, 4)
                return 4*((i+p)%8)+(j+q)%4
            alpha, beta = data['extension']
            i, j, t = a//16, (a//2)%8, a%2
            p, q, r = b//16, (b//2)%8, b%2
            central = (t+r+j*p+alpha*((i+p)//4)+beta*((j+q)//8))%2
            return 2*(8*((i+p)%4)+(j+q)%8)+central
        def lift(word):
            rows = [0]*size
            for g in range(size):
                if word >> g & 1:
                    for h in range(size):
                        rows[multiply(g, h)] ^= 1 << h
            return rows
        a, b = lift(data['a']), lift(data['b'])
        hx = [aa | bb << size for aa, bb in zip(a, b)]
        if size == 32:
            hz = [move(v, data['certificate']['fold']) for v in hx]
        else:
            hz = [bb | aa << size for aa, bb in zip(transpose(a, size), transpose(b, size))]
        return hx, hz
    assert kind == 'equivariant_orthogonal'
    rows = [(1 << g) | (1 << (32+g)) for g in range(32)]
    rows += [(1 << (64+g)) | (1 << (80+g)) for g in range(16)]
    for a, b, ordering in data['recipe']:
        entries = [a, a, b, b]
        translated = [sum(1 << (32*s+8*((g//8+t//8)%4)+(g%8+t%8)%8)
                          for s in range(4) for t in range(32) if entries[ordering[s]] >> t & 1)
                      for g in range(32)]
        updated = []
        for row in rows:
            new = row
            for v in translated:
                if (row & v).bit_count()%2:
                    new ^= v
            updated.append(new)
        rows = updated
    return rows, rows.copy()


def exclude_through(checks, stabilizers, width, maximum):
    """Every support has one split into its smallest a and largest b sites."""
    columns, space = transpose(checks, width), pivots(stabilizers)
    indices, counts = {}, {}
    for weight in range(1, maximum+1):
        left_size = weight//2
        if left_size not in indices:
            index = defaultdict(list)
            for support in itertools.combinations(range(width), left_size):
                syndrome = 0
                for q in support:
                    syndrome ^= columns[q]
                index[syndrome].append(support)
            indices[left_size] = index
        zero = 0
        for right in itertools.combinations(range(width), weight-left_size):
            syndrome = 0
            for q in right:
                syndrome ^= columns[q]
            for left in indices[left_size].get(syndrome, ()):
                if left and left[-1] >= right[0]:
                    continue
                support = left+right
                word = sum(1 << q for q in support)
                assert word.bit_count() == weight
                zero += 1
                if residual(word, space):
                    raise AssertionError(f'Logical below claimed distance: {support}')
        counts[weight] = zero
    return dict(maximum=maximum, zero_syndrome_counts=counts, all_are_stabilizers=True)


def verify_manifest(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for name, record in manifest['files'].items():
        path = (root/name).resolve()
        assert path.is_relative_to(root) and path.is_file() and not (root/name).is_symlink()
        blob = path.read_bytes()
        assert len(blob) == record['bytes'], name
        assert hashlib.sha256(blob).hexdigest() == record['sha256'], name
    return len(manifest['files'])


def verify_code(folder, exact_distance=False):
    folder = Path(folder)
    started = time.monotonic()
    meta = json.loads((folder/'metadata.json').read_text())
    data = json.loads((folder/'candidate.json').read_text())
    n, k, d = meta['n'], meta['k'], meta['distance']
    assert (data['n'], data['k']) == (n, k)
    hx, hz = data['hx'], data['hz']
    x, z = data['certificate']['x'], data['certificate']['z']
    assert len(x) == len(z) == k
    assert all(isinstance(v, int) and 0 <= v < 1 << n for v in hx+hz+x+z)
    assert len(pivots(hx)) == len(pivots(hz)) == (n-k)//2
    ax, az, lx, lz = (array(rows, n) for rows in (hx, hz, x, z))
    assert not np.any(ax @ az.T % 2)
    assert not np.any(ax @ lz.T % 2) and not np.any(az @ lx.T % 2)
    assert np.array_equal(lx @ lz.T % 2, np.eye(k, dtype=np.uint8))
    with np.load(folder/'checks.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['hx'], ax) and np.array_equal(saved['hz'], az)
    with np.load(folder/'logical_bases.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['x'], lx) and np.array_equal(saved['z'], lz)
    support_data = json.loads((folder/'supports.json').read_text())
    for key, rows in (('hx', hx), ('hz', hz), ('x', x), ('z', z)):
        assert support_data[key] == [[q for q in range(n) if w >> q & 1] for w in rows]
    rebuilt_x, rebuilt_z = reconstruct(data, meta['construction'])
    assert same_space(rebuilt_x, hx) and same_space(rebuilt_z, hz)
    shape = meta['logical_shape']
    assert shape[0]*shape[1] == k
    sx, sz = pivots(hx), pivots(hz)
    physical = json.loads((folder/'physical_permutations.json').read_text())
    for key in ('px', 'py'):
        assert physical[key] == data[key]
    assert physical['fold'] == data['certificate']['fold']
    actions = {}
    for key, axis in (('px', 0), ('py', 1)):
        p = data[key]
        actions[key] = order(p)
        target = [shape[1]*((i+1)%shape[0])+j if axis == 0 else shape[1]*i+(j+1)%shape[1]
                  for i in range(shape[0]) for j in range(shape[1])]
        for checks, logical, space in ((hx, x, sx), (hz, z, sz)):
            assert same_space([move(w, p) for w in checks], checks)
            assert all(residual(move(w, p)^logical[t], space) == 0 for w, t in zip(logical, target))
    if data.get('central') is not None:
        p = data['central']
        assert physical['central'] == p
        actions['central'] = order(p)
        for checks, logical, space in ((hx, x, sx), (hz, z, sz)):
            assert same_space([move(w, p) for w in checks], checks)
            assert all(residual(move(w, p)^w, space) == 0 for w in logical)
    p, target = physical['fold'], data['hadamard']['logical_permutation']
    actions['fold'] = order(p)
    assert sorted(target) == list(range(k))
    assert same_space([move(w, p) for w in hx], hz)
    assert same_space([move(w, p) for w in hz], hx)
    assert all(residual(move(w, p)^z[t], sz) == 0 for w, t in zip(x, target))
    assert all(residual(move(w, p)^x[t], sx) == 0 for w, t in zip(z, target))
    # RREF each sector separately so the direct-sum test is row-basis invariant.
    def rref(rows):
        b = pivots(rows)
        for bit in sorted(b):
            for other in b:
                if other > bit and b[other] >> bit & 1:
                    b[other] ^= b[bit]
        return list(b.values())
    parts = components(rref(hx)+rref(hz), n)
    assert parts == meta['component_sizes']
    distance = {}
    for side, checks, stabilizers in (('X', hz, hx), ('Z', hx, hz)):
        support = meta['distance_witnesses'][side]
        assert len(set(support)) == len(support) == d and all(0 <= q < n for q in support)
        word = sum(1 << q for q in support)
        assert all((w & word).bit_count()%2 == 0 for w in checks)
        assert residual(word, pivots(stabilizers))
        distance[side] = dict(upper_bound=d, witness=support)
        if exact_distance:
            distance[side]['lower_bound_certificate'] = exclude_through(checks, stabilizers, n, d-1)
            distance[side]['lower_bound'] = d
    loads = list(map(int, np.vstack((lx, lz)).sum(axis=0)))
    if n == 64:
        assert all(w.bit_count() == 10 for w in x+z) and loads == [5]*64
    return dict(code=folder.name, status='exact_distance_verified' if exact_distance else 'algebra_and_upper_witnesses_verified',
                n=n, k=k, claimed_distance=d, distance=distance, physical_orders=actions,
                components=parts, check_weights={s: dict(Counter(map(int.bit_count, h))) for s, h in (('X', hx), ('Z', hz))},
                logical_weights={s: dict(Counter(map(int.bit_count, h))) for s, h in (('X', x), ('Z', z))},
                total_logical_support=sum(loads), peak_logical_load=max(loads), elapsed_seconds=time.monotonic()-started)


def run(root=ROOT, exact_distance=False):
    root = Path(root)
    files = verify_manifest(root)
    index = json.loads((root/'index.json').read_text())
    results = []
    for entry in index['codes']:
        result = verify_code(root/entry['directory'], exact_distance)
        results.append(result)
        print(json.dumps(result), flush=True)
    return dict(status='verified', manifest_files=files, exact_distance_recomputed=exact_distance, codes=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance', action='store_true')
    parser.add_argument('--output', help='Optional fresh JSON report path, preferably outside this package')
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    report = run(exact_distance=args.distance)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
