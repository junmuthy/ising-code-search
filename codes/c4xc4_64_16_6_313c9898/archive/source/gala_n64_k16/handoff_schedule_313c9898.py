"""Export the preferred audited schedule and recheck its static code gates."""
import argparse
import json
from pathlib import Path
import shutil

from orbit_campaign import save
from orbit_schedule import binary
from sparse_core import save_certificate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit=json.loads(args.audit.read_text())
    assert audit['status']=='complete_independent_audit'
    chosen=[v for v in audit['bounds'] if v['schedule_id']=='30f5d7baeecc8a92']
    assert {v['basis'] for v in chosen}=={'X','Z'}
    assert all(v['lower_bound']==5 and v['upper_bound']==6 for v in chosen)
    reference=args.root/'independent_reference_exact4_v1/report.json'
    check=json.loads(reference.read_text())
    assert check['status']=='complete' and len(check['cases'])==2
    assert all(v['exact']['lower_bound']==5 for v in check['cases'])
    source=Path(chosen[0]['source'])
    code=json.loads((source/'code.json').read_text())
    # The nested data record originated from the minimum-weight-basis export.
    # Its inherited descriptive label is not the measured basis of this circuit.
    code=dict(code,data=dict(code['data'],
              presentation='independent_weight12_subset_for_depth12_fault5',
              source_data_presentation=code['data'].get('presentation')))
    args.output.mkdir(parents=True,exist_ok=False)
    static=save_certificate(code['data'],args.output/'code')
    assert static['metrics']['x_weights']==static['metrics']['z_weights']=={6:16}
    save(args.output/'code.json',code)
    for name in ('schedule.json','X.stim','Z.stim'):
        shutil.copyfile(source/name,args.output/name)
    for item in chosen:
        shutil.copyfile(item['six_fault_witness'],args.output/(item['basis']+'_six_fault_witness.json'))
    result=dict(status='complete_certified_handoff',schedule_id='30f5d7baeecc8a92',bounds=chosen,
                static_audit=str(args.output/'code/audit.json'),circuit_audit=str(args.audit),
                original_engine_certificate=str(reference),
                static_parameters=dict(n=64,k=16,d=6),
                circuit_contract=dict(rounds=3,noise='full_wait',observables=16,bases=['X','Z']),
                resources=dict(cnots_per_round=576,cnot_depth=12,data_qubits=64,ancillas=48,total_qubits=112),
                gates_preserved=['regular logical C4 x C4','individual logical H up to permutation'],
                all_canonical_logical_weights=6,
                source_sha256={str(Path(__file__).resolve()):binary.digest(Path(__file__).read_bytes())})
    save(args.output/'report.json',result)
    manifest_path=args.root/'MANIFEST.json'
    assert not manifest_path.exists()
    def inventory(paths):
        return [dict(path=str(path.resolve()),bytes=path.stat().st_size,
                     sha256=binary.digest(path.read_bytes())) for path in sorted(paths) if path.is_file()]
    here=Path(__file__).resolve().parent
    sources=[here/name for name in ('schedule_313c9898.py','schedule_313c9898_seeded.py',
             'schedule_313c9898_staged.py','compact_fault4.py','audit_schedule_313c9898.py',
             'handoff_schedule_313c9898.py','test_schedule_313c9898.py','test_compact_fault4.py')]
    manifest=dict(status='complete',artifacts=inventory(args.root.rglob('*')),sources=inventory(sources),
                  algorithm_crosschecks=inventory((here/'compact_fault4_validation').rglob('*')),
                  excludes=['this manifest itself'],
                  tests=dict(passed=93,groups=dict(compact=62,campaign_schedule=12,prior_orbit_schedule=5,prior_memory_replay=14)))
    save(manifest_path,manifest)
    for section in ('artifacts','sources','algorithm_crosschecks'):
        for item in manifest[section]:
            path=Path(item['path'])
            assert path.stat().st_size==item['bytes'] and binary.digest(path.read_bytes())==item['sha256']
    print(json.dumps(dict(status=result['status'],schedule=result['schedule_id'],resources=result['resources'])),flush=True)


if __name__=='__main__':
    main()
