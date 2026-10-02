"""Exact four-fault search using compact, collision-checked pair keys.

A deterministic GF(2)-linear projection puts equal full syndromes in the
same bucket. Every bucket collision is checked with ALL detector bits.
Projection collisions affect speed only, never completeness or soundness.
One uint64 stores a projected syndrome and two effect indices. This replaces
the older 36/44-byte structured records, without changing the fault model.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numba
import numpy as np
import stim

from orbit_campaign import save
from orbit_schedule import binary, faults


@numba.njit(cache=False)
def scan_pairs(keys, detectors, observables, index_bits):
    """Full-signature comparisons inside projected-syndrome buckets.

Requires unique single-effect detector masks. With no two-fault mechanism,
deduplicated effects have that property. Equal pair syndromes then imply
that different pairs are disjoint: a shared endpoint would force identical
single-effect syndromes at their other endpoints.
"""
    mask = np.uint64((1 << index_bits)-1)
    shift = 2*index_bits
    start = 0
    comparisons = 0
    while start < len(keys):
        end = start+1
        fingerprint = keys[start] >> shift
        while end < len(keys) and keys[end] >> shift == fingerprint:
            end += 1
        if end == start+1:
            start = end
            continue
        # One representative per distinct FULL syndrome in this bucket.
        representatives = np.empty(end-start, dtype=np.int64)
        count = 0
        for pos in range(start,end):
            left = int((keys[pos] >> index_bits) & mask)
            right = int(keys[pos] & mask)
            matched = False
            for slot in range(count):
                previous = keys[representatives[slot]]
                a = int((previous >> index_bits) & mask)
                b = int(previous & mask)
                comparisons += 1
                equal = True
                for chunk in range(detectors.shape[1]):
                    if (detectors[left,chunk] ^ detectors[right,chunk]) != (detectors[a,chunk] ^ detectors[b,chunk]):
                        equal = False
                        break
                if equal:
                    matched = True
                    if (observables[left] ^ observables[right]) != (observables[a] ^ observables[b]):
                        assert left != a and left != b and right != a and right != b
                        return np.array([left,right,a,b],dtype=np.int64), comparisons
                    break
            if not matched:
                representatives[count] = pos
                count += 1
        start = end
    return np.empty(0,dtype=np.int64), comparisons


def find_four(effects, *, progress=None, projection_bits=None):
    emit = progress or (lambda event: None)
    n = len(effects)
    if n < 4:
        return None, dict(pair_count=n*(n-1)//2, table_bytes=0)
    if len({e.detector_mask for e in effects}) != n:
        raise ValueError('unique detector masks required: certify through two first')
    if any(e.observable_mask.bit_length()>64 for e in effects):
        raise ValueError('at most 64 observables supported')
    index_bits=max(1,(n-1).bit_length())
    available=64-2*index_bits
    bits=available if projection_bits is None else projection_bits
    if not 1 <= bits <= available:
        raise ValueError('projection does not fit a uint64 pair key')
    chunks=max(1,max(e.detector_mask.bit_length() for e in effects)+63>>6)
    detectors=np.array([[(e.detector_mask>>(64*c))&((1<<64)-1) for c in range(chunks)] for e in effects],dtype=np.uint64)
    observables=np.array([e.observable_mask for e in effects],dtype=np.uint64)
    rng=random.Random(3139898)
    columns=[rng.getrandbits(bits) for _ in range(chunks*64)]
    hashed=[]
    for effect in effects:
        word=effect.detector_mask
        value=0
        while word:
            low=word & -word
            value ^= columns[low.bit_length()-1]
            word ^= low
        hashed.append(value)
    hashed=np.array(hashed,dtype=np.uint64)
    pairs=n*(n-1)//2
    keys=np.empty(pairs,dtype=np.uint64)
    metadata=dict(method='exact-linear-projection-full-syndrome-checked',distinct_fault_effects=n,
                  pair_count=pairs,table_bytes=int(keys.nbytes),index_bits=index_bits,projection_bits=bits,
                  projection_seed=3139898,unique_single_effect_syndromes=True)
    start=time.monotonic()
    emit(dict(event='compact-build-start',**metadata))
    position=0
    for left in range(n-1):
        count=n-left-1
        keys[position:position+count]=((hashed[left]^hashed[left+1:])<<np.uint64(2*index_bits)) | (np.uint64(left)<<np.uint64(index_bits)) | np.arange(left+1,n,dtype=np.uint64)
        position+=count
    assert position==pairs
    emit(dict(event='compact-sort-start',elapsed_seconds=time.monotonic()-start,**metadata))
    keys.sort()
    emit(dict(event='compact-scan-start',elapsed_seconds=time.monotonic()-start,**metadata))
    witness,comparisons=scan_pairs(keys,detectors,observables,index_bits)
    metadata.update(full_syndrome_comparisons=int(comparisons),elapsed_seconds=time.monotonic()-start)
    indices=[int(i) for i in witness] if len(witness) else None
    if indices is not None:
        d=o=0
        assert len(set(indices))==4
        for i in indices:
            d ^= effects[i].detector_mask
            o ^= effects[i].observable_mask
        assert d==0 and o!=0
    emit(dict(event='compact-complete',found_witness=indices is not None,**metadata))
    return indices,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,action='append',required=True)
    p.add_argument('--lower-report',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    lower={}
    for path in args.lower_report:
        saved=json.loads(path.read_text())
        assert saved['status']=='complete'
        for item in saved['cases']:
            if item['exact']['lower_bound']>=4:
                assert [(a['fault_count'],a['status']) for a in item['exact']['attempts'][:3]]==[(1,'unsat'),(2,'unsat'),(3,'unsat')]
                lower[item['circuit_sha256']]=(item['exact'],str(path),binary.digest(path.read_bytes()))
    report=dict(status='running',maximum_faults=4,cases=[],
                source_sha256={str(Path(__file__).resolve()):binary.digest(Path(__file__).read_bytes()),
                               **faults.inventory()['source_sha256']})
    save(args.output/'report.json',report)
    for source in args.folder:
        schedule=json.loads((source/'schedule.json').read_text())
        for basis in ('X','Z'):
            folder=args.output/(schedule['schedule_id']+'_'+basis)
            folder.mkdir()
            c=stim.Circuit.from_file(source/f'{basis}.stim')
            digest=binary.digest(str(c).encode())
            def progress(event):
                save(folder/'progress.json',event)
                print(json.dumps(dict(schedule=schedule['schedule_id'],basis=basis,**event)),flush=True)
            start=time.monotonic()
            reused=lower.get(digest)
            exact=reused[0] if reused else faults.certify_through_three(c,heartbeat_seconds=20,progress=progress)
            exact=dict(exact,attempts=exact['attempts'][:3])
            if exact['lower_bound']>=4:
                effects=faults.extract_fault_effects(faults.detector_error_model(c))
                indices,stats=find_four(effects,progress=progress)
                exact.update(lower_bound=5 if indices is None else 4,distance=None if indices is None else 4,
                             witness=None if indices is None else dict(fault_count=4,effects=[asdict(effects[i]) for i in indices]),
                             compact_search=stats)
                exact['attempts'].append(dict(fault_count=4,status='unsat' if indices is None else 'sat',
                                             method=stats['method'],elapsed_seconds=stats['elapsed_seconds']))
            upper=None
            if exact['witness']:
                errors=[faults.explain_masks(c,e['detector_mask'],e['observable_mask']) for e in exact['witness']['effects']]
                physical=faults.replay(c,errors)
                (folder/'witness.stim').write_text(physical.pop('faulted_circuit'))
                save(folder/'physical_witness.json',physical)
                upper=physical['fault_count']
            item=dict(source=str(source),basis=basis,schedule_id=schedule['schedule_id'],circuit_sha256=digest,
                      exact=exact,upper_bound=upper,wall_seconds=time.monotonic()-start,
                      reused_lower_certificate=None if reused is None else dict(path=reused[1],sha256=reused[2]))
            save(folder/'certificate.json',item)
            report['cases'].append(item)
            save(args.output/'report.json',report)
            print(json.dumps(dict(stage='exact_complete',schedule=schedule['schedule_id'],basis=basis,lower=exact['lower_bound'],upper=upper)),flush=True)
    report['status']='complete'
    save(args.output/'report.json',report)


if __name__=='__main__':
    main()
