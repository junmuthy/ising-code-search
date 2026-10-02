"""Small exact binary routines; integers encode rows, bit q is qubit q."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


def bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def basis(rows):
    pivots = {}
    for row in rows:
        row = int(row)
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row
                break
            row ^= pivots[p]
    return pivots


def reduce(row, pivots):
    for p in sorted(pivots, reverse=True):
        if row >> p & 1:
            row ^= pivots[p]
    return row


def rank(rows):
    return len(basis(rows))


def nullspace(rows, width):
    pivots = basis(rows)
    out = []
    for free in range(width):
        if free in pivots:
            continue
        v = 1 << free
        for p in sorted(pivots):
            if (v & pivots[p]).bit_count() % 2:
                v ^= 1 << p
        out.append(v)
    return out


def combine(rows, coefficients):
    out = 0
    for i in bits(coefficients):
        out ^= rows[i]
    return out


def pack(matrix):
    return [int.from_bytes(row.tobytes(), 'little') for row in
            np.packbits(np.asarray(matrix, dtype=np.uint8), axis=1, bitorder='little')]


def unpack(rows, width):
    return np.asarray([[(r >> j) & 1 for j in range(width)] for r in rows], dtype=np.uint8).reshape(len(rows),width)


def transpose(rows, width):
    out = [0] * width
    for i, row in enumerate(rows):
        for j in bits(row):
            out[j] ^= 1 << i
    return out


def permute(v, permutation):
    out = 0
    for j in bits(v):
        out |= 1 << int(permutation[j])
    return out


def compose(p, q):
    """Apply q, then p."""
    return [p[q[j]] for j in range(len(p))]


def power(p, exponent):
    out = list(range(len(p)))
    for _ in range(exponent):
        out = compose(p, out)
    return out


def same_space(left, right):
    return rank(left) == rank(right) == rank(list(left) + list(right))


def quotient_basis(checks, stabilizers, n):
    pivots = basis(stabilizers)
    out = []
    for row in nullspace(checks, n):
        residual = reduce(row, pivots)
        if residual:
            pivots[residual.bit_length() - 1] = residual
            out.append(row)
    return out


def pairing(left, right):
    return [sum(((a & b).bit_count() % 2) << j for j, b in enumerate(right)) for a in left]


def inverse(rows):
    n = len(rows)
    aug = [row | (1 << (n+i)) for i, row in enumerate(rows)]
    for j in range(n):
        p = next((i for i in range(j, n) if aug[i] >> j & 1), None)
        if p is None:
            raise ValueError('singular binary matrix')
        aug[j], aug[p] = aug[p], aug[j]
        for i in range(n):
            if i != j and aug[i] >> j & 1:
                aug[i] ^= aug[j]
    return [row >> n for row in aug]


def canonical_x(z, xraw):
    inv_transpose = transpose(inverse(pairing(z, xraw)), len(z))
    return [combine(xraw, row) for row in inv_transpose]


def commute(hx, hz):
    return all((x & z).bit_count() % 2 == 0 for x in hx for z in hz)


def module(hx, hz, px, py, central=None):
    n = len(px)
    if any(v < 0 or v.bit_length() > n for v in hx+hz):
        raise ValueError('check matrix contains an out-of-range qubit')
    for p in (px, py) + (() if central is None else (central,)):
        if sorted(p) != list(range(n)):
            raise ValueError('invalid physical permutation')
    rz, rx = rank(hz), rank(hx)
    info = {'n': n, 'k': n-rx-rz, 'rank_x': rx, 'rank_z': rz}
    if not commute(hx, hz):
        return {**info, 'status': 'noncommuting'}, None
    for p in (px, py) + (() if central is None else (central,)):
        if not same_space([permute(v, p) for v in hx], hx) or not same_space([permute(v, p) for v in hz], hz):
            return {**info, 'status': 'not_automorphism'}, None
    if info['k'] != 32:
        return {**info, 'status': 'wrong_k'}, None
    zraw = quotient_basis(hx, hz, n)
    xraw = quotient_basis(hz, hx, n)
    bz, bx = basis(hz), basis(hx)
    assert len(zraw) == len(xraw) == 32
    tests = [(power(px, 4), list(range(n))), (power(py, 8), list(range(n))),
             (compose(px, py), compose(py, px))]
    if central is not None:
        tests.append((central, list(range(n))))
    for p, q in tests:
        if any(reduce(permute(v, p) ^ permute(v, q), b) for rows, b in ((zraw, bz), (xraw, bx)) for v in rows):
            return {**info, 'status': 'wrong_logical_relations'}, None
    perms = [compose(power(px, i), power(py, j)) for i in range(4) for j in range(8)]
    seed = None
    for v in sorted(zraw, key=int.bit_count):
        norm = 0
        for p in perms:
            norm ^= permute(v, p)
        if reduce(norm, bz):
            seed = v
            break
    if seed is None:
        return {**info, 'status': 'norm_zero'}, None
    z = [permute(seed, p) for p in perms]
    assert rank(hz + z) == rz + 32
    x = canonical_x(z, xraw)
    assert pairing(z, x) == [1 << i for i in range(32)]
    for p, axis in ((px, 0), (py, 1)):
        for i in range(4):
            for j in range(8):
                source = i*8+j
                target = ((i+1)%4)*8+j if axis == 0 else i*8+(j+1)%8
                assert not reduce(permute(z[source], p) ^ z[target], bz)
                assert not reduce(permute(x[source], p) ^ x[target], bx)
    return {**info, 'status': 'regular_grid', 'orbit_rank': 32}, {'z': z, 'x': x, 'perms': perms}


def permutation_matrix(rows):
    return all(row.bit_count() == 1 for row in rows) and len(set(rows)) == len(rows)


def hadamard(hx, hz, grid, fold, *, solve_seconds=2):
    if sorted(fold) != list(range(len(fold))):
        raise ValueError('invalid Hadamard fold permutation')
    if not same_space([permute(v, fold) for v in hx], hz) or not same_space([permute(v, fold) for v in hz], hx):
        return {'status': 'not_zx_fold'}, None

    def evaluate(z, x):
        action_zx = pairing([permute(v, fold) for v in z], z)
        action_xz = pairing([permute(v, fold) for v in x], x)
        if action_zx == action_xz and permutation_matrix(action_zx):
            return {'status': 'logical_hadamard', 'logical_permutation': [r.bit_length()-1 for r in action_zx]}, {'z': z, 'x': x}
        return None

    z, x = grid['z'], grid['x']
    direct = evaluate(z, x)
    if direct:
        return direct
    if solve_seconds <= 0:
        return {'status': 'basis_unresolved'}, None
    import z3
    aa = [z3.Bool(f'a{i}') for i in range(32)]
    solver = z3.Solver()
    solver.set(timeout=int(solve_seconds*1000))

    def xor(items):
        items = list(items)
        if not items:
            return z3.BoolVal(False)
        acc = items[0]
        for item in items[1:]:
            acc = z3.Xor(acc, item)
        return acc

    solver.add(xor(aa))  # Units in F2[C4 x C8] have odd augmentation.
    first_row = []
    for translation in grid['perms']:
        moved = [permute(permute(v, translation), fold) for v in z]
        quadratic = pairing(z, moved)
        terms = [aa[i] for i in range(32) if quadratic[i] >> i & 1]
        terms += [z3.And(aa[i], aa[j]) for i in range(32) for j in range(i+1,32)
                  if ((quadratic[i] >> j) ^ (quadratic[j] >> i)) & 1]
        first_row.append(xor(terms))
    solver.add(z3.PbEq([(v,1) for v in first_row], 1))
    for _ in range(4):
        status = solver.check()
        if status != z3.sat:
            return {'status': 'equivariant_basis_unsat' if status == z3.unsat else 'basis_timeout'}, None
        model = solver.model()
        coeff = sum(int(z3.is_true(model.eval(a, model_completion=True))) << i for i,a in enumerate(aa))
        seed = combine(z, coeff)
        newz = [permute(seed,p) for p in grid['perms']]
        newx = canonical_x(newz, x)
        result = evaluate(newz, newx)
        if result:
            result[0]['basis_coefficients'] = coeff
            return result
        solver.add(z3.Or([a != bool(coeff >> i & 1) for i,a in enumerate(aa)]))
    return {'status': 'basis_unresolved'}, None


def low_logical(check, conjugate, n, maximum=5):
    """Complete exclusion through maximum <= 5, including degeneracy."""
    if maximum not in range(1, 6):
        raise ValueError('maximum must be between one and five')
    syndromes, logicals = transpose(check, n), transpose(conjugate, n)
    singles = defaultdict(list)
    for q, syndrome in enumerate(syndromes):
        if syndrome == 0 and logicals[q]:
            return [q]
        singles[syndrome].append(q)
    if maximum == 1:
        return None
    pairs = defaultdict(list)
    for a,b in itertools.combinations(range(n),2):
        s, l = syndromes[a]^syndromes[b], logicals[a]^logicals[b]
        if s == 0 and l:
            return [a,b]
        pairs[s].append((a,b,l,(1<<a)|(1<<b)))
    if maximum == 2:
        return None
    for s, bucket in pairs.items():
        for a,b,l,_ in bucket:
            for c in singles.get(s, ()):
                if c > b and l ^ logicals[c]:
                    return [a,b,c]
        if maximum >= 4:
            for index, (a,b,l,mask) in enumerate(bucket):
                for c,d,l2,mask2 in bucket[index+1:]:
                    if not mask & mask2 and l ^ l2:
                        return sorted([a,b,c,d])
    if maximum < 5:
        return None
    for a,b,c in itertools.combinations(range(n),3):
        s = syndromes[a]^syndromes[b]^syndromes[c]
        l = logicals[a]^logicals[b]^logicals[c]
        mask = (1<<a)|(1<<b)|(1<<c)
        for d,e,l2,mask2 in pairs.get(s, ()):
            if not mask & mask2 and l ^ l2:
                return sorted([a,b,c,d,e])
    return None
