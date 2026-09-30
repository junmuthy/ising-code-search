"""Assemble a portable, provenance-preserving bundle in a NEW directory.

Function extraction is a mechanical relocation: original function bodies
and decorators are kept byte-for-byte, with explicit portable import headers.
"""
import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE/'schedule_313c9898_campaign'
KEY='313c98984982b039f1068bfd6001a4ffceb370b2091c3edbc9d3f8b915aaa028'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def make_manifest(root):
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'MANIFEST.json']
    write_json(root/'MANIFEST.json',dict(status='complete',files=files,excludes=['MANIFEST.json itself']))
    return len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gala',type=Path,default=Path('/home/judah_unmuth/gala-code-search'))
    args=p.parse_args()
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    provenance=dict(code_id=KEY,python=sys.version,dependencies={},copied_sources=[],extracted_sources=[],
                    qldpc_revision=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip(),
                    gala_revision=subprocess.check_output(['git','-C',str(args.gala),'rev-parse','HEAD'],text=True).strip(),
                    note='Source snapshots are preserved. Reproduction uses only bundled local modules and data.')
    for package in ('numpy','stim','galois','numba','llvmlite','z3-solver','pytest'):
        provenance['dependencies'][package]=metadata.version(package)
    def copy(source,target):
        target=root/target
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        provenance['copied_sources'].append(dict(source=str(source),destination=str(target.relative_to(root)),sha256=sha(source)))
    def extract(source,target,names,header):
        text=source.read_text()
        lines=text.splitlines(keepends=True)
        nodes={node.name:node for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        blocks=[]
        for name in names:
            node=nodes[name]
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
            blocks.append(''.join(lines[start-1:node.end_lineno]))
        destination=root/target
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(header+'\n\n'+'\n\n'.join(blocks)+'\n')
        provenance['extracted_sources'].append(dict(source=str(source),source_sha256=sha(source),
                   destination=target,functions=names,body_sha256=[hashlib.sha256(b.encode()).hexdigest() for b in blocks],
                   transformation='unchanged function bodies/decorators with portable import header'))
    # Preserve every scientific output from the completed bounded campaign.
    shutil.copytree(CAMPAIGN,root/'archive/campaign')
    shutil.copytree(HERE/'compact_fault4_validation',root/'archive/compact_fault4_validation')
    shutil.copytree(CAMPAIGN/'handoff_v1',root/'preferred')
    shutil.copytree(HERE/'orbit_campaign/handoff_v1/alternate_w12_all_weight6_logicals',root/'minimum_weight_checks')
    for name in ('code.json','schedule.json','X.stim','Z.stim','result.json'):
        copy(CAMPAIGN/'baseline_v1/cd91029936d21197'/name,'original/'+name)
    for engine,folder in [('compact','compact_exact4_v1'),('reference','independent_reference_exact4_v1')]:
        copy(CAMPAIGN/folder/'report.json',f'certificates/{engine}/report.json')
        for basis in ('X','Z'):
            source=CAMPAIGN/folder/f'30f5d7baeecc8a92_{basis}'
            shutil.copytree(source,root/f'certificates/{engine}/{basis}')
    copy(HERE/'orbit_campaign/all_finalists_v1'/KEY/'weight_enumeration.json','certificates/stabilizer_weight_enumeration.json')
    # Archive the original scripts, separately from the portable entry points.
    original_sources=list(HERE.glob('*.py'))+list((HERE.parent/'gala_n128_k32').glob('*.py'))
    for child in ('fault_distance_milestone1','distance6_fault_search'):
        original_sources+=list((HERE/child).glob('*.py'))
    for source in original_sources:
        copy(source,'archive/source/'+str(source.relative_to(HERE.parent)))
    templates=HERE/'portable_313c9898'
    for source in templates.rglob('*'):
        if source.is_file():
            copy(source,str(source.relative_to(templates)))
    copy(HERE/'test_compact_fault4.py','tests/test_compact_fault4.py')
    copy(HERE.parent/'gala_n128_k32/algebra.py','lib/algebra.py')
    copy(HERE.parent/'gala_n128_k32/certify.py','lib/certify.py')
    copy(HERE/'distance6_fault_search/circuit.py','lib/circuit.py')
    extract(HERE/'fault_distance_milestone1/model.py','lib/model.py',
            ['digest','save','pivots','residual','same_space','move','supports','array','relations'],
            '"""Portable binary helpers; original function bodies unchanged."""\nimport hashlib\nimport json\nfrom pathlib import Path\nimport numpy as np')
    extract(HERE/'distance6_fault_search/schedule.py','lib/schedule.py',['verify'],
            '"""Original exact edge-coverage/collision/backaction verifier."""')
    extract(HERE/'fault_distance_milestone1/audit.py','lib/replay.py',['replay_saved'],
            '"""Original physical witness replay against the unmodified reference."""\nimport numpy as np\nimport stim\nfrom model import digest\nfrom circuit import NOISE')
    extract(HERE/'sparse_core.py','lib/static_audit.py',['rref','independent_audit'],
            '"""Original independent array audit, with local helper imports."""\nfrom collections import Counter\nimport numpy as np\nfrom algebra import basis,compose,unpack')
    extract(HERE/'extract_component.py','lib/extract_component.py',['components'],
            '"""Original row-space connectivity helper."""\nfrom algebra import bits')
    extract(HERE/'compact_fault4.py','lib/compact_fault4.py',['scan_pairs','find_four'],
            '"""Original collision-checked compact exact four-fault search."""\nimport random\nimport time\nimport numba\nimport numpy as np')
    for relative in ('stim_fault_distance/fault_distance.py','schedule_fault_search/screening.py'):
        copy(args.gala/'n32_k4_d6_reference_code'/relative,'lib/n32_k4_d6_reference_code/'+relative)
    # Python namespace packages intentionally keep the original engine imports.
    candidate=json.loads((root/'preferred/code/candidate.json').read_text())
    parent=json.loads((root/'original/schedule.json').read_text())
    preferred=json.loads((root/'preferred/schedule.json').read_text())
    times=[dict(data=g['data'],tick=t) for t,layer in enumerate(parent['layers']) for g in layer
           if g['type']=='X' and g['check']==0]
    assert len(times)==12
    write_json(root/'construction.json',dict(code_id=KEY,group='C8 x C4',a=candidate['a'],b=candidate['b'],
               coordinate_convention='q = 32*s + 4*i + j',physical_fold_order=4,
               parent_schedule_id=parent['schedule_id'],preferred_schedule_id=preferred['schedule_id'],
               x_seed_edge_times=times,retained_check_indices=preferred['retained_check_indices']))
    (root/'requirements.txt').write_text('\n'.join(f'{name}=={version}' for name,version in provenance['dependencies'].items())+'\n')
    write_json(root/'PROVENANCE.json',provenance)
    write_json(root/'REPRODUCTION_VALIDATION.json',dict(status='pending_portability_test'))
    count=make_manifest(root)
    print(json.dumps(dict(status='assembled',output=str(root),files=count)),flush=True)


if __name__=='__main__':
    main()
