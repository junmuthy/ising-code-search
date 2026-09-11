"""Simplify representatives of the existing code, without searching new codes.

Keep the saved logical labels and Z representatives. With the involutive
Hadamard fold P and its logical permutation sigma, set X_i=P(Z_sigma(i)).
Check the change independently with array-based GF(2) calculations.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from sparse_core import HERE, permute, unpack
from certify import gf_rank, in_space, moved, permutation_order


SOURCE = HERE/'sparse_campaign/distance8_handoff_v1'


def simplify(source=SOURCE):
    source = Path(source).resolve()
    candidate_path = source/'independent/candidate.json'
    audit_path = source/'full_audit.json'
    original = json.loads(candidate_path.read_text())
    old_audit = json.loads(audit_path.read_text())
    data = deepcopy(original)
    p = data['certificate']['fold']
    sigma = data['hadamard']['logical_permutation']
    assert permutation_order(p) == 2
    assert sorted(sigma) == list(range(16))
    assert all(sigma[sigma[i]] == i for i in range(16))
    data['certificate']['x'] = [permute(data['certificate']['z'][sigma[i]], p)
                              for i in range(16)]
    hx, hz = unpack(data['hx'], 64), unpack(data['hz'], 64)
    x, z = unpack(data['certificate']['x'], 64), unpack(data['certificate']['z'], 64)
    old_x = unpack(original['certificate']['x'], 64)
    assert gf_rank(hx) == gf_rank(hz) == 24
    assert not np.any(hx @ hz.T % 2)
    assert not np.any(x @ hz.T % 2) and not np.any(z @ hx.T % 2)
    assert np.array_equal(x @ z.T % 2, np.eye(16, dtype=int))
    assert in_space(old_x ^ x, hx)
    assert np.array_equal(moved(x, p), z[sigma])
    assert np.array_equal(moved(z, p), x[sigma])
    assert in_space(moved(hx, p), hz) and in_space(moved(hz, p), hx)
    actions = {}
    for name, key in (('Tx', 'px'), ('Ty', 'py')):
        q = data[key]
        target = old_audit['actions'][name]['logical_permutation']
        for rows, stabilizers in ((x, hx), (z, hz)):
            assert in_space(moved(rows, q) ^ rows[target], stabilizers)
            assert in_space(moved(stabilizers, q), stabilizers)
        actions[name] = dict(physical_order=permutation_order(q), logical_permutation=target)
    for rows, stabilizers in ((x, hx), (z, hz)):
        assert in_space(moved(rows, data['central']) ^ rows, stabilizers)
    assert np.all(x.sum(axis=1) == 12) and np.all(z.sum(axis=1) == 12)
    witnesses = {}
    for side, rows, dual, checks, stabilizers in (('X', x, z, hz, hx), ('Z', z, x, hx, hz)):
        support = old_audit['distance'][side]['weight_eight_witness']
        word = np.zeros((1, 64), dtype=np.uint8)
        word[0, support] = 1
        assert word.sum() == 8 and not np.any(checks @ word.T % 2)
        assert not in_space(word, stabilizers)
        labels = (word @ dual.T % 2)[0]
        assert in_space(word ^ (labels @ rows % 2), stabilizers)
        witnesses[side] = dict(physical_support=support,
                              logical_grid_labels=[list(divmod(i, 4)) for i in np.flatnonzero(labels).tolist()],
                              individual_logical=len(np.flatnonzero(labels)) == 1)
    data['logical_representatives'] = 'Hadamard-matched weight-12 X/Z representatives; original logical labels unchanged'
    report = dict(status='verified_same_code_same_logical_labels', n=64, k=16,
                  distance=dict(value=8, provenance='Unchanged check matrices; inherited exact distance certificate',
                                source=str(audit_path), source_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest()),
                  candidate_source=str(candidate_path),
                  candidate_source_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                  old_x_weights=dict(Counter(v.bit_count() for v in original['certificate']['x'])),
                  x_weights=dict(Counter(map(int, x.sum(axis=1)))),
                  z_weights=dict(Counter(map(int, z.sum(axis=1)))),
                  logical_weights_optimal=False, translations=actions,
                  logical_hadamard_permutation=sigma,
                  hadamard_pairs=[[i, sigma[i]] for i in range(16) if i < sigma[i]],
                  saved_distance_witnesses=witnesses,
                  logical_support_total_before=int(old_x.sum()+z.sum()),
                  logical_support_total_after=int(x.sum()+z.sum()),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return data, report, (hx, hz, x, z)


def export(destination, source=SOURCE):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    data, report, arrays = simplify(source)
    destination.mkdir(parents=True)
    (destination/'candidate.json').write_text(json.dumps(data, indent=2)+'\n')
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    hx, hz, x, z = arrays
    np.savez(destination/'logical_bases.npz', x=x, z=z)
    np.savez(destination/'checks.npz', hx=hx, hz=hz)
    supports = {s: [np.flatnonzero(row).tolist() for row in rows] for s, rows in (('X', x), ('Z', z))}
    (destination/'logical_supports.json').write_text(json.dumps(supports, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output)), flush=True)
