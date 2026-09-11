"""Array-based audit of the complete logical-basis filter and final handoff."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sparse_core import permute, unpack
from certify import GF2, in_space, moved
from distance8_logical_handoff import simplify
from distance8_weight10_search import audit_basis, export_basis, logical_shifts


def audit_catalog(data, catalog):
    """Use independent 16-by-16 GF(2) inverse and action matrices."""
    shifts = logical_shifts()
    canonical, observed = {}, set()
    for record in catalog:
        seed = record['seed']
        members = {permute(seed, p) for p in shifts}
        assert len(members) == 16 and min(members) == seed
        assert not observed & members
        assert members == set(record['members'])
        observed.update(members)
        canonical.update((m, seed) for m in members)
    assert len(catalog) == 2048
    assert observed == {c for c in range(65536) if c.bit_count()%2}
    x, z = unpack(data['certificate']['x'], 64), unpack(data['certificate']['z'], 64)
    p = data['certificate']['fold']
    fzx, fxz = moved(z, p) @ z.T % 2, moved(x, p) @ x.T % 2
    compatible = 0
    for record in catalog:
        c = unpack([permute(record['seed'], q) for q in shifts], 16)
        d = np.asarray(np.linalg.inv(GF2(c)).T, dtype=np.uint8)
        assert np.array_equal(d @ c.T % 2, np.eye(16, dtype=np.uint8))
        zx, xz = c @ fzx @ c.T % 2, d @ fxz @ d.T % 2
        good = (np.array_equal(zx, xz) and np.all(zx.sum(axis=1) == 1)
                and np.all(zx.sum(axis=0) == 1))
        expected = np.argmax(zx, axis=1).tolist() if good else None
        assert record['saved_hadamard_permutation'] == expected
        # This independently checks the cached X-to-Z minimum-cost map.
        partner_bits = d[0] @ fxz % 2
        partner = sum(int(v) << i for i, v in enumerate(partner_bits))
        assert record['dual_Z_seed'] == canonical[partner]
        if good:
            assert canonical[partner] == record['seed']
            compatible += 1
    return dict(unit_classes=32768, translation_orbits=2048,
                array_checked_basis_changes=len(catalog), compatible_orbits=compatible,
                compatible_unit_classes=16*compatible,
                member_coverage='Changing the seed by a logical translation relabels both canonical bases together; monomial H is invariant')


def run(handoff, destination):
    handoff, destination = Path(handoff).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    data, provenance, _ = simplify()
    report = json.loads((handoff/'report.json').read_text())
    source = Path(report['source'])
    assert report['source_candidate_sha256'] == provenance['candidate_source_sha256']
    for filename, field in (('summary.json', 'source_summary_sha256'),
                            ('cosets.jsonl', 'source_cosets_sha256'),
                            ('catalog.json', 'source_catalog_sha256')):
        assert hashlib.sha256((source/filename).read_bytes()).hexdigest() == report[field]
    distance = report['source_distance_audit']
    assert hashlib.sha256(Path(distance['source']).read_bytes()).hexdigest() == distance['source_sha256']
    old = report['weight_eight_obstruction']
    blob = Path(old['path']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == old['sha256']
    proof = json.loads(blob)
    assert proof['all_weight_eight_canonical_grids'] == 0
    assert proof['source_candidate_sha256'] == provenance['candidate_source_sha256']
    earlier_catalog = Path(proof['source_catalog'])
    assert hashlib.sha256((earlier_catalog/'report.json').read_bytes()).hexdigest() == proof['source_catalog_sha256']
    assert hashlib.sha256((earlier_catalog/'weight_eight_words.json').read_bytes()).hexdigest() == proof['source_words_sha256']
    catalog_result = audit_catalog(data, json.loads((source/'catalog.json').read_text()))
    assert catalog_result['compatible_orbits'] == report['cosets_reenumerated'] == 256
    best = json.loads((handoff/'best/candidate.json').read_text())
    checked = audit_basis(best, data, 10)
    assert checked['combined_qubit_loads'] == [5]*64
    assert checked['total_logical_support'] == 320
    expected_change = [sum(1 << (4*i+(j+t)%4) for t in range(3))
                       for i in range(4) for j in range(4)]
    assert best['logical_basis_change'] == dict(Z_from_original_Z=expected_change,
                                                X_from_original_X=expected_change)
    assert checked['actions']['H']['logical_permutation'] == [4*i+(2*i+3-j)%4
                                                             for i in range(4) for j in range(4)]
    # All-qubit opposite-type stabilizers exclude odd physical support.
    for key in ('hx', 'hz'):
        assert in_space(np.ones((1, 64), dtype=np.uint8), unpack(data[key], 64))
    destination.mkdir(parents=True)
    # Preserve the peak-load winner, and expose the overlap trade-off as a
    # separate handoff rather than overwriting its chosen representatives.
    alternative = min(report['ranking'], key=lambda r: (r['score'][1], r['score'][0],
                                                       r['score'][2], r['seed'].bit_count(), r['seed']))
    tradeoff = export_basis(data, alternative['seed'], alternative['z_words'][0],
        data['certificate']['fold'], destination/'lower_overlap', z_words=alternative['z_words'])
    result = dict(status='independently_verified_weight_ten_logical_handoff',
                  catalog=catalog_result, audit=checked,
                  minimum_maximum_logical_weight=10,
                  support_optimality_scope='Pure canonical CSS logical grids with the saved C4 x C4 translations and common-basis H',
                  total_support=320, best_possible_peak_combined_load=5,
                  maximum_pairwise_overlap=checked['maximum_pairwise_support_overlap'],
                  alternative=dict(seed=alternative['seed'], peak_load=max(tradeoff['combined_qubit_loads']),
                                   maximum_pairwise_overlap=tradeoff['maximum_pairwise_support_overlap'],
                                   overlap_optimality='not proved'),
                  handoff_report_sha256=hashlib.sha256((handoff/'report.json').read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256((handoff/'best/candidate.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination/'full_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'audit'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.handoff, args.output)
