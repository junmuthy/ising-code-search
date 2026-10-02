"""Joint shape-stabilizer/fold orbits, with explicit physical transport maps."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

from joint_fold_search import (GROUP, automorphisms, configurations, config_key,
                               gauge_fold, make_manifest, transform_pair, translate)
from algebra import compose, permute


def inverse(p):
    return [p.index(i) for i in range(len(p))]


def transport(a, b, fold, q):
    """Transport coefficients by an affine, possibly sheet-swapping map."""
    alpha = [GROUP.mul(q[g] % 32, GROUP.inverse[q[0] % 32]) for g in range(32)]
    out = [0, 0]
    for s, word in enumerate((a, b)):
        shift = q[32*s] % 32
        target = q[32*s] // 32
        assert all(q[32*s+g] // 32 == target and
                   GROUP.mul(q[32*s+g] % 32, GROUP.inverse[shift]) == alpha[g]
                   for g in range(32))
        out[target] = translate(permute(word, alpha), GROUP.inverse[shift])
    return *out, compose(q, compose(fold, inverse(q)))


def build_orbits(manifest):
    configs = list(configurations(manifest))
    lookup = {config_key(c): i for i, c in enumerate(configs)}
    auts = automorphisms()
    stabilizers = {}
    for orbit in manifest['orbits']:
        word = orbit['representative']
        stabilizers[word] = [(alpha, shift) for alpha in auts for shift in range(32)
                             if translate(permute(word, alpha), GROUP.inverse[shift]) == word]
    remaining = set(range(len(configs)))
    results = []
    swap = list(range(32, 64))+list(range(32))
    while remaining:
        root = min(remaining)
        maps = {root: list(range(64))}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            c = configs[node]
            a, b = (c['fixed'], 0) if c['fixed_side'] == 'a' else (0, c['fixed'])
            moves = []
            for alpha, shift in stabilizers[c['fixed']]:
                shifts = (shift, 0) if c['fixed_side'] == 'a' else (0, shift)
                aa, bb, pp, q1 = transform_pair(a, b, c['fold'], alpha, shifts)
                aa, bb, pp, q2 = gauge_fold(aa, bb, pp, c['fixed_side'])
                moves.append((c['fixed_side'], pp, compose(q2, q1)))
            side = 'b' if c['fixed_side'] == 'a' else 'a'
            aa, bb, pp = transport(a, b, c['fold'], swap)
            aa, bb, pp, gauge = gauge_fold(aa, bb, pp, side)
            moves.append((side, pp, compose(gauge, swap)))
            for side, fold, q in moves:
                target = lookup[config_key(dict(fixed=c['fixed'], fixed_side=side, fold=fold))]
                if target not in maps:
                    assert target in remaining
                    maps[target] = compose(q, maps[node])
                    queue.append(target)
        remaining -= set(maps)
        # Revalidate each composed transport, not just its elementary edges.
        root_config = configs[root]
        a, b = ((root_config['fixed'], 0) if root_config['fixed_side'] == 'a'
                else (0, root_config['fixed']))
        for node, q in maps.items():
            aa, bb, pp = transport(a, b, root_config['fold'], q)
            c = configs[node]
            assert pp == c['fold']
            assert (aa, bb) == ((c['fixed'], 0) if c['fixed_side'] == 'a' else (0, c['fixed']))
        results.append(dict(representative=root, members=sorted(maps), root_to_member=maps))
    return dict(configurations=len(configs), orbits=len(results), orbit_records=results,
                configuration_keys=[config_key(c) for c in configs],
                stabilizer_sizes={w: len(s) for w, s in stabilizers.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-distance', type=int, choices=(6, 8), default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = build_orbits(make_manifest(args.min_distance))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('orbit_records', 'configuration_keys')}), flush=True)
