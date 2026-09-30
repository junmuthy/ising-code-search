"""Export the distance-eight alternative with an involutive Hadamard fold."""
import json

from sparse_core import HERE, permute, save_certificate, same_space, basis, reduce, commute
from sparse_finalize import balanced_basis
from sparse_distance_upgrade import weight_seven_search
from check_weight_reduction import coordinates, enumerate_space, orbit_search


def run():
    key = '71e1114cdaa26f7c2bc3cbd2098452d03aee37bfcc62016f59367dc784c7625e'
    root = HERE/'sparse_campaign'
    destination = root/'distance8_handoff_v1'
    if destination.exists():
        raise FileExistsError(destination)
    data = json.loads((root/'twisted_w12_v1/certified'/key/'candidate.json').read_text())
    upgraded = json.loads((root/'distance_upgrades_v1'/f'{key}.json').read_text())
    assert all(v['lower_bound'] == v['upper_bound'] == 8 for v in upgraded['distance'].values())
    enum = {s: enumerate_space(data[s], ceiling=12) for s in ('hx', 'hz')}
    hx, loads = balanced_basis(enum['hx']['low_weight_rows'], data['certificate']['fold'])
    hz = [permute(v, data['certificate']['fold']) for v in hx]
    assert same_space(hx, data['hx']) and same_space(hz, data['hz'])
    optimized = dict(data, hx=hx, hz=hz, distance_exact=8, presentation='minimum-total-weight independent CSS generators')
    audit = save_certificate(optimized, destination/'independent')
    save_certificate(dict(data, distance_exact=8), destination/'symmetric')
    # A stronger supplemental audit retains the preliminary full-code audit
    # and adds exact exclusion through seven and verified weight-eight errors.
    for side, h, s in (('Z', hx, hz), ('X', hz, hx)):
        exact = weight_seven_search(h, s)
        assert exact['status'] == 'excluded'
        witness = upgraded['distance'][side]['weight_eight_search']['witness']
        word = sum(1 << q for q in witness)
        assert len(witness) == 8 and commute(h, [word]) and reduce(word, basis(s))
        audit['distance'][side].update(lower_bound=8, upper_bound=8, weight_seven_exclusion=exact,
                                        weight_eight_witness=witness)
    audit['d'] = 8
    (destination/'full_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    orbits = {}
    for side in ('hx', 'hz'):
        o = orbit_search(enum[side]['low_weight_rows'], data['px'], data['py'], data['certificate']['fold'])
        o['maximum_check_weight'] = 12
        orbits[side] = o
    changes = {s: dict(new_from_original=[coordinates(data[s], v) for v in rows],
                       original_from_new=[coordinates(rows, v) for v in data[s]])
               for s, rows in (('hx', hx), ('hz', hz))}
    report = dict(selected_id=key, rationale='distance eight and order-two physical Hadamard fold',
                   total_support=576, load_balancing=loads, enumerations=enum,
                   symmetry_closed_search=orbits, basis_changes=changes)
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (destination/'independent/supports.json').write_text(json.dumps({s: [[q for q in range(64) if v >> q & 1] for v in rows]
                                                                  for s, rows in (('hx', hx), ('hz', hz))}, indent=2)+'\n')
    print(json.dumps(dict(id=key, d=8, total_support=576, loads=loads)), flush=True)


if __name__ == '__main__':
    run()
