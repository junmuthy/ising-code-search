"""Exact C4 x C4 module, common-H basis search, and independent auditing.

No old certificates or fixed-size verifiers are modified. Integer rows use
bit q for qubit q; permutations map old coordinates to new coordinates.
"""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gala_n128_k32'))
from algebra import (basis, canonical_x, combine, commute, compose, low_logical,
                     pairing, permute, permutation_matrix, power, quotient_basis,
                     rank, reduce, same_space, unpack)


def rref(rows):
    b = basis(rows)
    for p in sorted(b):
        for q in b:
            if q > p and b[q] >> p & 1:
                b[q] ^= b[p]
    return [b[p] for p in sorted(b, reverse=True)]


def space_key(hx, hz):
    return hashlib.sha256(json.dumps([rref(hx), rref(hz)]).encode()).hexdigest()


def grid_module(hx, hz, px, py, central=None):
    n = len(px)
    rx, rz = rank(hx), rank(hz)
    info = dict(n=n, k=n-rx-rz, rank_x=rx, rank_z=rz)
    if n >= 70 or info['k'] != 16 or rx != rz:
        return {**info, 'status': 'wrong_ranks'}, None
    assert all(0 <= v < 1 << n for v in hx+hz)
    if not commute(hx, hz):
        return {**info, 'status': 'noncommuting'}, None
    for p in (px, py) + (() if central is None else (central,)):
        assert sorted(p) == list(range(n))
        if any(not same_space([permute(v, p) for v in h], h) for h in (hx, hz)):
            return {**info, 'status': 'not_automorphism'}, None
    zraw, xraw = quotient_basis(hx, hz, n), quotient_basis(hz, hx, n)
    bz, bx = basis(hz), basis(hx)
    identity = list(range(n))
    tests = [(power(px, 4), identity), (power(py, 4), identity),
             (compose(px, py), compose(py, px))]
    if central is not None:
        tests.append((central, identity))
    for p, q in tests:
        if any(reduce(permute(v, p)^permute(v, q), b)
               for rows, b in ((zraw, bz), (xraw, bx)) for v in rows):
            return {**info, 'status': 'wrong_logical_relations'}, None
    perms = [compose(power(px, i), power(py, j)) for i in range(4) for j in range(4)]
    for seed in sorted(zraw, key=int.bit_count):
        z = [permute(seed, p) for p in perms]
        if reduce(combine(z, (1 << 16)-1), bz):
            assert rank(hz+z) == rz+16
            x = canonical_x(z, xraw)
            for p, axis in ((px, 0), (py, 1)):
                for i, j in itertools.product(range(4), repeat=2):
                    target = ((i+1)%4)*4+j if axis == 0 else i*4+(j+1)%4
                    assert not reduce(permute(z[i*4+j], p)^z[target], bz)
                    assert not reduce(permute(x[i*4+j], p)^x[target], bx)
            return {**info, 'status': 'regular_grid'}, dict(z=z, x=x, perms=perms)
    return {**info, 'status': 'norm_zero'}, None


def common_h(hx, hz, grid, fold):
    """Exhaust every translation-compatible basis (32768 units) if needed.

    First-row folded pairing is only a necessary filter. Every surviving basis
    is checked against BOTH complete logical action matrices. Therefore a
    negative result is exact for this fold and equivariant basis class.
    """
    if any(not same_space([permute(v, fold) for v in h], target)
           for h, target in ((hx, hz), (hz, hx))):
        return {'status': 'not_zx_fold'}, None

    def evaluate(z, x):
        zx = pairing([permute(v, fold) for v in z], z)
        xz = pairing([permute(v, fold) for v in x], x)
        if zx == xz and permutation_matrix(zx):
            return [v.bit_length()-1 for v in zx]

    z, x = grid['z'], grid['x']
    direct = evaluate(z, x)
    if direct is not None:
        return {'status': 'logical_hadamard', 'logical_permutation': direct}, dict(z=z, x=x)
    units = np.asarray([c for c in range(1, 1 << 16) if c.bit_count()%2], dtype=np.uint16)
    variables = [(units >> i) & 1 for i in range(16)]
    first = np.zeros(len(units), dtype=np.uint16)
    for j, translation in enumerate(grid['perms']):
        moved = [permute(permute(v, translation), fold) for v in z]
        quadratic = pairing(z, moved)
        result = np.zeros(len(units), dtype=np.uint16)
        for i in range(16):
            if quadratic[i] >> i & 1:
                result ^= variables[i]
            for k in range(i+1, 16):
                if ((quadratic[i] >> k) ^ (quadratic[k] >> i)) & 1:
                    result ^= variables[i] & variables[k]
        first |= result << j
    survivors = units[np.bitwise_count(first) == 1]
    for c in survivors:
        seed = combine(z, int(c))
        newz = [permute(seed, p) for p in grid['perms']]
        newx = canonical_x(newz, x)
        result = evaluate(newz, newx)
        if result is not None:
            return {'status': 'logical_hadamard', 'logical_permutation': result,
                    'basis_coefficients': int(c)}, dict(z=newz, x=newx)
    return {'status': 'equivariant_basis_exhausted', 'units': 32768,
            'first_row_survivors': len(survivors)}, None


def assess(hx, hz, px, py, folds, central=None):
    info, grid = grid_module(hx, hz, px, py, central)
    if grid is None:
        return info, None
    for side, h, dual in (('Z', hx, grid['x']), ('X', hz, grid['z'])):
        witness = low_logical(h, dual, len(px))
        if witness is not None:
            return {**info, 'status': 'low_distance', 'sector': side, 'witness': witness}, None
    statuses = Counter()
    for fold in folds:
        hinfo, logical = common_h(hx, hz, grid, fold)
        statuses[hinfo['status']] += 1
        if logical is not None:
            return {**info, 'status': 'accepted', 'distance_lower_bound': 6, 'hadamard': hinfo}, {**logical, 'fold': fold}
    return {**info, 'status': 'h_not_certified', 'distance_lower_bound': 6,
            'fold_statuses': dict(statuses)}, None


def independent_audit(data, seconds=0):
    from certify import GF2, gf_rank, in_space, independent_small_errors, milp, moved, permutation_order, weight_six_witness
    from extract_component import components
    n = data['n']
    hx, hz = unpack(data['hx'], n), unpack(data['hz'], n)
    x, z = unpack(data['certificate']['x'], n), unpack(data['certificate']['z'], n)
    assert n < 70 and data['k'] == n-gf_rank(hx)-gf_rank(hz) == 16
    assert gf_rank(hx) == gf_rank(hz) == (n-16)//2
    assert not np.any(hx@hz.T % 2)
    assert not np.any(hx@z.T % 2) and not np.any(hz@x.T % 2)
    assert np.array_equal(z@x.T % 2, np.eye(16, dtype=np.uint8))
    actions = {}
    for name, p in (('Tx', data['px']), ('Ty', data['py'])):
        assert sorted(p) == list(range(n))
        target = [((i+1)%4)*4+j if name == 'Tx' else i*4+(j+1)%4
                  for i in range(4) for j in range(4)]
        for h in (hx, hz):
            assert in_space(moved(h, p), h)
        assert in_space(moved(z, p)^z[target], hz)
        assert in_space(moved(x, p)^x[target], hx)
        actions[name] = dict(physical_order=permutation_order(p), logical_permutation=target,
                             z_action=(moved(z, p)@x.T%2).tolist(), x_action=(moved(x, p)@z.T%2).tolist())
    if data.get('central') is not None:
        p = data['central']
        assert sorted(p) == list(range(n))
        assert in_space(moved(z, p)^z, hz) and in_space(moved(x, p)^x, hx)
        for h in (hx, hz):
            assert in_space(moved(h, p), h)
        actions['central'] = dict(physical_order=permutation_order(p), logical_action='identity')
    p = data['certificate']['fold']
    assert sorted(p) == list(range(n))
    assert in_space(moved(hx, p), hz) and in_space(moved(hz, p), hx)
    zx, xz = moved(z, p)@z.T%2, moved(x, p)@x.T%2
    assert np.array_equal(zx, xz)
    assert np.all(zx.sum(axis=0) == 1) and np.all(zx.sum(axis=1) == 1)
    target = np.argmax(zx, axis=1)
    assert in_space(moved(z, p)^x[target], hx) and in_space(moved(x, p)^z[target], hz)
    actions['H'] = dict(physical_order=permutation_order(p), logical_permutation=target.tolist(), zx_action=zx.tolist(), xz_action=xz.tolist())
    metrics = dict(row_space_component_sizes=sorted(map(len, components(rref(data['hx'])+rref(data['hz']), n))),
                   physical_translations_commute=compose(data['px'], data['py']) == compose(data['py'], data['px']))
    for name, rows in (('hx', hx), ('hz', hz), ('x', x), ('z', z)):
        metrics[name+'_weights'] = dict(Counter(map(int, rows.sum(axis=1))))
    for name, rows in (('hx', hx), ('hz', hz), ('combined', np.vstack((hx, hz)))):
        metrics[name+'_degrees'] = dict(Counter(map(int, rows.sum(axis=0))))
    distance = {}
    for side, h, stabilizers, dual in (('Z', hx, hz, x), ('X', hz, hx, z)):
        lower = independent_small_errors(h, stabilizers)
        assert lower['status'] == 'certified'
        witness = weight_six_witness(h, stabilizers)
        distance[side] = dict(exclusion=lower, lower_bound=6 if witness else 7,
                              upper_bound=6 if witness else None, weight_six_witness=witness)
        if seconds:
            distance[side]['milp'] = milp(h, dual, seconds)
            assert 'witness' not in distance[side]['milp']
    return dict(status='certified', n=n, k=16, rank_x=gf_rank(hx), rank_z=gf_rank(hz),
                actions=actions, distance=distance, metrics=metrics), (hx, hz, x, z)


def save_certificate(data, destination, seconds=0):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    audit, arrays = independent_audit(data, seconds)
    audit['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in HERE.glob('sparse_*.py')}
    destination.mkdir(parents=True)
    (destination/'candidate.json').write_text(json.dumps(data, indent=2)+'\n')
    (destination/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    np.savez_compressed(destination/'checks.npz', hx=arrays[0], hz=arrays[1])
    np.savez_compressed(destination/'logical_bases.npz', x=arrays[2], z=arrays[3])
    (destination/'physical_permutations.json').write_text(json.dumps({
        'tx': data['px'], 'ty': data['py'], 'central': data.get('central'),
        'fold': data['certificate']['fold']}, indent=2)+'\n')
    return audit
