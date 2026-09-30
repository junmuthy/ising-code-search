"""Resolve distance-seven-or-better finalists without changing old audits."""
from collections import defaultdict
import itertools
import json

from sparse_core import HERE, basis, commute, reduce, unpack
from algebra import transpose
from certify import milp


def weight_seven_search(check, stabilizers, n=64):
    columns = transpose(check, n)
    pivots = basis(stabilizers)
    triples = defaultdict(list)
    for a, b, c in itertools.combinations(range(n), 3):
        triples[columns[a]^columns[b]^columns[c]].append((a, b, c))
    zero = 0
    for a, b, c, d in itertools.combinations(range(n), 4):
        for e, f, g in triples.get(columns[a]^columns[b]^columns[c]^columns[d], ()):
            if g >= a:
                continue
            support = [e, f, g, a, b, c, d]
            word = sum(1 << q for q in support)
            assert commute(check, [word])
            zero += 1
            if reduce(word, pivots):
                return dict(status='logical_witness', support=support, zero_syndrome_seen=zero)
    return dict(status='excluded', maximum=7, zero_syndrome_weight_seven=zero,
                method='complete sorted triple/quadruple syndrome matching with stabilizer membership')


def run():
    root = HERE/'sparse_campaign/twisted_w12_v1/certified'
    output = HERE/'sparse_campaign/distance_upgrades_v1'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    for path in sorted(root.glob('*/candidate.json')):
        data = json.loads(path.read_text())
        old = json.loads((path.parent/'audit.json').read_text())
        if old['distance']['X']['lower_bound'] != 7:
            continue
        result = {}
        for side, h, s, dual in (('Z', 'hx', 'hz', 'x'), ('X', 'hz', 'hx', 'z')):
            seven = weight_seven_search(data[h], data[s])
            if seven['status'] == 'logical_witness':
                result[side] = dict(lower_bound=7, upper_bound=7, exact_search=seven)
            else:
                eight = milp(unpack(data[h], 64), unpack(data['certificate'][dual], 64), 20, maximum=8)
                if 'witness' in eight:
                    word = sum(1 << q for q in eight['witness'])
                    assert len(eight['witness']) == 8 and commute(data[h], [word]) and reduce(word, basis(data[s]))
                result[side] = dict(lower_bound=8, upper_bound=8 if 'witness' in eight else None,
                                    exact_search=seven, weight_eight_search=eight)
            print(json.dumps(dict(id=path.parent.name, side=side, result=result[side])), flush=True)
        (output/f'{path.parent.name}.json').write_text(json.dumps(dict(candidate=str(path), distance=result), indent=2)+'\n')


if __name__ == '__main__':
    run()
