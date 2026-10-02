"""Complete the joint search modulo certified joint coefficient/fold maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import joint_fold_search as search
from joint_fold_orbits import inverse, transport
from joint_fold_parallel import worker
from algebra import compose


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def checkpoint_interrupted(root):
    """Seal only committed slices after the caller has stopped its own run.

    The candidate tail is preserved verbatim, not counted as completed work.
    This is not safe to use as an assertion that an active process has stopped;
    process termination must be confirmed by the execution tool first.
    """
    root = Path(root)
    if (root/'summary.json').exists():
        raise FileExistsError(root/'summary.json')
    rows = read_rows(root/'slices.jsonl')
    assert all(r['status'] in ('complete_slice', 'covered_by_prior_certificate') for r in rows)
    manifest = search.make_manifest(6)
    assert json.loads(json.dumps(manifest)) == json.loads((root/'manifest.json').read_text())
    counts, slices = search.Counter(), search.Counter()
    for r in rows:
        counts.update(r.get('candidate_counts', {}))
        slices[r['status']] += 1
    limit = sum(counts.values())
    with (root/'candidates.jsonl').open() as stream:
        total_lines = sum(1 for _ in stream)
    assert total_lines >= limit
    parameters = dict(min_distance=6, maxweight=10, seconds=3600, max_slices=10000,
                      output=str(root), covered=sorted({str(Path(r['prior']).parent) for r in rows if 'prior' in r}))
    result = dict(status='interrupted_checkpoint', parameters=parameters,
                  configurations=manifest['raw_slices'], configurations_processed=len(rows),
                  slice_counts=dict(slices), candidate_counts=dict(counts),
                  manifest_sha256=search.digest(manifest),
                  candidate_journal_completed_prefix_rows=limit,
                  ignored_uncommitted_candidate_tail_lines=total_lines-limit,
                  note='Execution-tool exit 130 confirmed; completed prefix retained before symmetry-reduced continuation.',
                  checkpoint_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  search_driver_sha256=hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def direct_certificates(folders, configs):
    lookup = {search.config_key(c): i for i, c in enumerate(configs)}
    direct = {}
    cache = {}
    for folder in folders:
        folder = Path(folder).resolve()
        rows = read_rows(folder/'slices.jsonl')
        summary_path = folder/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['status'] == 'interrupted_checkpoint':
                assert len(rows) >= summary['configurations_processed']
                rows = rows[:summary['configurations_processed']]
        for record in rows:
            source = folder
            while record['status'] == 'covered_by_prior_certificate':
                path = Path(record['prior'])
                if path not in cache:
                    cache[path] = {r['slice_id']: r for r in read_rows(path)}
                record = cache[path][record['prior_slice_id']]
                source = path.parent.resolve()
            if record['status'] != 'complete_slice':
                continue
            key = search.config_key(record)
            assert record['eligible_partners'] == record['processed_eligible_partners']
            assert record['partner_weight_ceiling'] >= 6
            direct[lookup[key]] = dict(source_folder=str(source), source_slice_id=record['slice_id'],
                                       accepted=record['candidate_counts'].get('accepted', 0))
    return direct


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = search.make_manifest(6)
    configs = list(search.configurations(manifest))
    orbits = json.loads(Path(args.orbits).read_text())
    assert orbits['configuration_keys'] == [search.config_key(c) for c in configs]
    direct = direct_certificates([args.checkpoint], configs)
    todo = [o['representative'] for o in orbits['orbit_records'] if not set(o['members']) & set(direct)]
    chunks = [todo[i:i+args.shard_size] for i in range(0, len(todo), args.shard_size)]
    (output/'schedule.json').write_text(json.dumps(dict(parameters=vars(args),
        direct_before=len(direct), joint_orbits=orbits['orbits'], representatives_to_search=todo,
        shards=chunks), indent=2)+'\n')
    print(json.dumps(dict(direct_before=len(direct), joint_orbits=orbits['orbits'], remaining=len(todo),
                          shards=len(chunks), workers=args.workers)), flush=True)
    parameters = dict(min_distance=6, maxweight=10, seconds=args.seconds, max_slices=10000)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, parameters, chunk, str(output/f'shard_{i:03d}')) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    direct.update(direct_certificates([r['output'] for r in results], configs))
    records, unresolved = [], []
    for orbit in orbits['orbit_records']:
        available = sorted(set(orbit['members']) & set(direct))
        if not available:
            unresolved.extend(orbit['members'])
            continue
        maps = orbit['root_to_member']
        for target in orbit['members']:
            source = target if target in direct else available[0]
            q = compose(maps[str(target)], inverse(maps[str(source)]))
            sc, tc = configs[source], configs[target]
            a, b = (sc['fixed'], 0) if sc['fixed_side'] == 'a' else (0, sc['fixed'])
            aa, bb, pp = transport(a, b, sc['fold'], q)
            assert pp == tc['fold']
            assert (aa, bb) == ((tc['fixed'], 0) if tc['fixed_side'] == 'a' else (0, tc['fixed']))
            records.append(dict(target_index=target, source_index=source,
                                target_config_key=search.config_key(tc), **direct[source],
                                physical_permutation=q, kind='direct' if target == source else 'joint_equivalence'))
    records.sort(key=lambda r: r['target_index'])
    (output/'coverage.json').write_text(json.dumps(records, indent=2)+'\n')
    result = dict(status='complete_joint_equivalence_scope' if not unresolved else 'partial_joint_equivalence_scope',
                  parameters=vars(args), configurations=len(configs), covered_configurations=len(records),
                  direct_configurations=len(direct), transported_configurations=sum(r['kind']=='joint_equivalence' for r in records),
                  unresolved_indices=sorted(unresolved), accepted_direct_slices=sum(bool(r['accepted']) for r in direct.values()),
                  workers=results, orbit_manifest_sha256=hashlib.sha256(Path(args.orbits).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'workers'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seal-interrupted', action='store_true')
    parser.add_argument('--orbits')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--shard-size', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=1800)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.seal_interrupted:
        print(json.dumps(checkpoint_interrupted(args.checkpoint)), flush=True)
    else:
        assert args.output and args.orbits
        run(args)
