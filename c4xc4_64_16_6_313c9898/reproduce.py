"""Portable reconstruction and certification of the fixed 313c9898 code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lib'))

import numpy as np
import stim

from algebra import permute, same_space, unpack
from circuit import build
from model import supports, relations
from replay import replay_saved
from schedule import verify


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest():
    # The compact Git bundle excludes bulk intermediate runs, not runtime inputs.
    name = 'MANIFEST.reproduction.json'
    manifest = read(name if (ROOT / name).exists() else 'MANIFEST.json')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert path.is_file(), entry['path']
        assert path.stat().st_size == entry['bytes'], entry['path']
        assert digest(path) == entry['sha256'], entry['path']
    return len(manifest['files'])


def reconstruct():
    """Regenerate matrices from C8 x C4 coefficients, and layers from 12 colors."""
    meta = read('construction.json')
    def q(s, i, j):
        return 32*s + 4*(i % 8) + j % 4
    def lift(word, i, j):
        return sum(1 << q(0, i-g//4, j-g%4) for g in range(32) if word >> g & 1)
    hx = [lift(meta['a'], i, j) | (lift(meta['b'], i, j) << 32)
          for i in range(8) for j in range(4)]
    fold = [q(1-s, 3*i+4*s, -j) for s in range(2) for i in range(8) for j in range(4)]
    hz = [permute(w, fold) for w in hx]
    original = read('original/code.json')
    assert original['checks'] == dict(X=hx, Z=hz)
    data = read('preferred/code/candidate.json')
    assert data['px'] == [q(s,i+1,j) for s in range(2) for i in range(8) for j in range(4)]
    assert data['py'] == [q(s,i,j+1) for s in range(2) for i in range(8) for j in range(4)]
    assert data['central'] == [q(s,i+4,j) for s in range(2) for i in range(8) for j in range(4)]
    assert data['certificate']['fold'] == fold
    selected = meta['retained_check_indices']
    checks = {s: [rows[i] for i in selected[s]] for s, rows in (('X',hx),('Z',hz))}
    assert checks == dict(X=data['hx'],Z=data['hz'])
    assert same_space(hx,checks['X']) and same_space(hz,checks['Z'])
    logicals = dict(X=data['certificate']['x'],Z=data['certificate']['z'])
    code = dict(data=data,checks=checks,fold=fold,logicals=logicals,
                supports={s:supports(rows) for s,rows in checks.items()},
                logical_supports={s:supports(rows) for s,rows in logicals.items()},
                relations={s:relations(rows) for s,rows in checks.items()})
    lookups = {s:{old:new for new,old in enumerate(selected[s])} for s in ('X','Z')}
    times = {}
    for r in range(32):
        for item in meta['x_seed_edge_times']:
            old, tick = item['data'], item['tick']
            moved = q(old//32, (old%32)//4+r//4, old%4+r%4)
            if r in lookups['X']:
                times['X',lookups['X'][r],moved] = tick
            if r in lookups['Z']:
                times['Z',lookups['Z'][r],fold[moved]] = 11-tick
    layers = [[] for _ in range(12)]
    for (kind,row,qubit),tick in sorted(times.items()):
        layers[tick].append(dict(type=kind,check=row,data=qubit))
    schedule = read('preferred/schedule.json')
    assert schedule['layers'] == layers
    verify(code,schedule)
    assert sum(map(len,layers)) == 576
    # All npz matrices are checked against independently reconstructed rows.
    with np.load(ROOT/'preferred/code/checks.npz') as saved:
        assert np.array_equal(saved['hx'],unpack(checks['X'],64))
        assert np.array_equal(saved['hz'],unpack(checks['Z'],64))
    with np.load(ROOT/'preferred/code/logical_bases.npz') as saved:
        assert np.array_equal(saved['x'],unpack(logicals['X'],64))
        assert np.array_equal(saved['z'],unpack(logicals['Z'],64))
    return data,code,schedule


def validate_static(data):
    from static_audit import independent_audit
    result,_ = independent_audit(data)
    assert result['n']==64 and result['k']==16
    assert all(result['distance'][s]['lower_bound']==result['distance'][s]['upper_bound']==6 for s in ('X','Z'))
    assert result['metrics']['x_weights']==result['metrics']['z_weights']=={6:16}
    assert result['metrics']['row_space_component_sizes']==[64]
    alternate = read('minimum_weight_checks/candidate.json')
    for k in ('hx','hz'):
        assert same_space(data[k],alternate[k])
        assert sorted(w.bit_count() for w in alternate[k])==[8]*16+[12]*8
    return result


def validate_circuit(code,schedule,basis):
    c = build(code,schedule,basis,rounds=3,noise='full_wait')
    assert c == stim.Circuit.from_file(ROOT/f'preferred/{basis}.stim')
    ideal = c.without_noise()
    ideal.detector_error_model(allow_gauge_detectors=False)
    d,o = ideal.compile_detector_sampler(seed=313).sample(32,separate_observables=True)
    assert not np.any(d) and not np.any(o)
    witness = read(f'preferred/{basis}_six_fault_witness.json')
    replay_saved(c,witness,12)
    circuit_hash=hashlib.sha256(str(c).encode()).hexdigest()
    for engine in ('compact','reference'):
        certificate=read(f'certificates/{engine}/{basis}/certificate.json')
        assert certificate['circuit_sha256']==circuit_hash
        assert certificate['exact']['lower_bound']==5
        assert [(v['fault_count'],v['status']) for v in certificate['exact']['attempts']]==[(i,'unsat') for i in range(1,5)]
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exact-four',action='store_true',help='rerun all one-through-four fault exclusions, not just check saved certificates')
    parser.add_argument('--engine',choices=['compact','reference'],default='compact')
    parser.add_argument('--output',type=Path,help='fresh directory for optional regenerated artifacts')
    args=parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
    count=check_manifest()
    data,code,schedule=reconstruct()
    static=validate_static(data)
    result=dict(status='verified',manifest_files=count,static=static,circuits=[],
                fault_distance_claim='saved certificate verified; not recomputed' if not args.exact_four else 'fresh exact exclusion through four',
                engine=args.engine if args.exact_four else None)
    if args.output:
        (args.output/'candidate.json').write_text(json.dumps(data,indent=2)+'\n')
        (args.output/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    for basis in ('X','Z'):
        c=validate_circuit(code,schedule,basis)
        item=dict(basis=basis,qubits=c.num_qubits,observables=c.num_observables,
                  cnot_depth=12,cnots_per_round=576,lower_bound=5,upper_bound=6)
        if args.exact_four:
            from n32_k4_d6_reference_code.schedule_fault_search.screening import certify_through_three,certify_through_four
            from n32_k4_d6_reference_code.stim_fault_distance.fault_distance import detector_error_model,extract_fault_effects
            def progress(event):
                print(json.dumps(dict(basis=basis,**event)),flush=True)
                if args.output:
                    (args.output/f'{basis}_progress.json').write_text(json.dumps(event,indent=2)+'\n')
            if args.engine=='reference':
                exact=certify_through_four(c,heartbeat_seconds=20,progress=progress)
            else:
                from compact_fault4 import find_four
                exact=certify_through_three(c,heartbeat_seconds=20,progress=progress)
                assert exact['lower_bound']==4
                effects=extract_fault_effects(detector_error_model(c))
                witness,stats=find_four(effects,progress=progress)
                assert witness is None
                exact=dict(exact,lower_bound=5,compact_search=stats,
                           attempts=exact['attempts']+[dict(fault_count=4,status='unsat',method=stats['method'])])
            assert exact['lower_bound']==5
            item['fresh_certificate']=exact
        if args.output:
            c.to_file(args.output/f'{basis}.stim')
        result['circuits'].append(item)
        print(json.dumps(dict(stage='verified',basis=basis,lower=5,upper=6)),flush=True)
    if args.output:
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='verified',files=count,code='[[64,16,6]]',schedule='30f5d7baeecc8a92',
                         fresh_fault_exclusion=args.exact_four)),flush=True)


if __name__=='__main__':
    main()
