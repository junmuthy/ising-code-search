"""Original exact edge-coverage/collision/backaction verifier."""

def verify(code, record):
    times = {}
    for t, layer in enumerate(record['layers']):
        assert len({g['data'] for g in layer}) == len(layer)
        assert len({(g['type'], g['check']) for g in layer}) == len(layer)
        for g in layer:
            edge = g['type'], g['check'], g['data']
            assert edge not in times
            times[edge] = t
    expected = {(s, r, q) for s in ('X', 'Z') for r, row in enumerate(code['supports'][s]) for q in row}
    assert set(times) == expected and len(record['layers']) == record['depth']
    for xr, xs in enumerate(code['supports']['X']):
        for zr, zs in enumerate(code['supports']['Z']):
            assert sum(times['X', xr, q] < times['Z', zr, q] for q in set(xs) & set(zs)) % 2 == 0
    return dict(collision_free=True, exact_edge_coverage=True, clean_backaction=True,
                cnots=len(times), layer_sizes=list(map(len, record['layers'])),
                minimum_depth_lower_bound=max(map(len, code['supports']['X'] + code['supports']['Z'])))

