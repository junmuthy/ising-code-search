"""Complete one-seed support-swap/deletion neighborhoods of the new hits."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import time

from orbit_extension import evaluate, geometry, module_generators, orbit
from orbit_campaign import save
from sparse_core import commute, permute, rank, save_certificate, space_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    geo = geometry('four_sheet')
    counts, seen, accepted = Counter(), set(), []
    start = time.monotonic()
    sources = sorted(args.root.glob('*/original/candidate.json'))
    with (args.output/'candidates.jsonl').open('x') as stream:
        for source in sources:
            data = json.loads(source.read_text())
            if not data.get('seeds'):
                continue
            fold, seeds = data['certificate']['fold'], data['seeds']
            recipes = [('control', -1, 0)]
            for index, word in enumerate(seeds):
                occupied = [q for q in range(64) if word >> q & 1]
                empty = [q for q in range(64) if not word >> q & 1]
                recipes.extend(('swap', index, (1 << a) | (1 << b)) for a in occupied for b in empty)
                recipes.extend(('delete_two', index, (1 << a) | (1 << b)) for a, b in itertools.combinations(occupied, 2))
            for operation, index, mask in recipes:
                altered = seeds[:]
                if index >= 0:
                    altered[index] ^= mask
                hx = list(dict.fromkeys(v for seed in altered for v in orbit(seed, geo['actions'])))
                hz = [permute(v, fold) for v in hx]
                record = dict(source=str(source), operation=operation, seed_index=index, xor_mask=mask,
                              seeds=altered, rank=rank(hx))
                certificate = None
                if record['rank'] != 24:
                    record['status'] = 'wrong_rank'
                elif not commute(hx, hz):
                    record['status'] = 'noncommuting'
                else:
                    key = space_key(hx, hz)
                    record['row_space_id'] = key
                    candidate = dict(hx=hx, hz=hz, seeds=altered,
                                     module_generator_count=module_generators(hx, geo['generators']))
                    if (key, tuple(fold)) in seen:
                        record['status'] = 'duplicate_code_fold'
                    else:
                        seen.add((key, tuple(fold)))
                        info, certificate = evaluate(candidate, geo, fold)
                        record.update(info)
                    if certificate:
                        output = dict(data, **candidate)
                        output['certificate'] = certificate
                        # Increments are reconstructed rather than inherited.
                        partial, increments = [], []
                        for seed in altered:
                            old = rank(partial)
                            partial += orbit(seed, geo['actions'])
                            increments.append(rank(partial) - old)
                        output['orbit_rank_increments'] = increments
                        folder = args.output/'certified'/key
                        if not folder.exists():
                            save_certificate(output, folder)
                        accepted.append(dict(id=key, operation=operation, path=str(folder)))
                counts[record['status']] += 1
                stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(json.dumps(dict(source=str(source), counts=dict(counts))), flush=True)
    save(args.output/'summary.json', dict(status='complete_defined_neighborhood', counts=dict(counts),
         accepted=accepted, seconds=time.monotonic()-start, sources=list(map(str, sources)),
         scope='Single support swap or two-bit deletion in one seed, other seed and fold fixed; includes two positive controls.'))


if __name__ == '__main__':
    main()
