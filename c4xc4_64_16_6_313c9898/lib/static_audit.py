"""Original independent array audit, with local helper imports."""
from collections import Counter
import numpy as np
from algebra import basis,compose,unpack

def rref(rows):
    b = basis(rows)
    for p in sorted(b):
        for q in b:
            if q > p and b[q] >> p & 1:
                b[q] ^= b[p]
    return [b[p] for p in sorted(b, reverse=True)]


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

