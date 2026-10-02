"""Resume unfinished joint slices in bounded local worker processes.

Each worker runs the already-tested exact driver on an explicit slice list.
Only the final consolidation can claim complete joint coverage. No agents or
external services are used. Existing output directories are never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
import multiprocessing
from pathlib import Path

import joint_fold_search as search


def worker(parameters, indices, output):
    original = search.configurations
    selected = set(indices)
    search.configurations = lambda manifest: (c for i, c in enumerate(original(manifest)) if i in selected)
    args = argparse.Namespace(**parameters, covered=[], output=output)
    try:
        with Path(output+'.log').open('x') as log, redirect_stdout(log):
            search.run(args)
    finally:
        search.configurations = original
    path = Path(output)/'summary.json'
    summary = json.loads(path.read_text())
    summary['status'] = ('complete_configured_shard' if summary['status'] == 'complete_joint_scope'
                         else 'partial_configured_shard')
    summary['global_indices'] = indices
    summary['shard_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(summary, indent=2)+'\n')
    return dict(output=output, status=summary['status'], configurations=summary['configurations'],
                processed=summary['configurations_processed'], candidate_counts=summary['candidate_counts'],
                seconds=summary['elapsed_seconds'])


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = search.make_manifest(args.min_distance)
    configs = list(search.configurations(manifest))
    prior = search.load_prior(args.covered, args.min_distance, args.maxweight)
    todo = [i for i, c in enumerate(configs) if search.config_key(c) not in prior]
    chunks = [todo[i:i+args.shard_size] for i in range(0, len(todo), args.shard_size)]
    (output/'schedule.json').write_text(json.dumps(dict(parameters=vars(args),
        total_configurations=len(configs), remaining_indices=todo, shards=chunks), indent=2)+'\n')
    print(json.dumps(dict(total=len(configs), remaining=len(todo), shards=len(chunks), workers=args.workers)), flush=True)
    parameters = dict(min_distance=args.min_distance, maxweight=args.maxweight,
                      seconds=args.seconds, max_slices=10000)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, parameters, chunk, str(output/f'shard_{index:03d}'))
                   for index, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    complete = all(r['status'] == 'complete_configured_shard' for r in results)
    (output/'workers.json').write_text(json.dumps(dict(complete=complete, results=results), indent=2)+'\n')
    if complete:
        # Run the full configuration list, linking all complete slice proofs.
        # The consolidated journal is the only family-level coverage claim.
        combined = argparse.Namespace(**parameters, covered=args.covered+[r['output'] for r in results],
                                      output=str(output/'combined'))
        search.run(combined)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-distance', type=int, choices=(6, 8), default=6)
    parser.add_argument('--maxweight', type=int, choices=(8, 10), default=10)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--shard-size', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--covered', action='append', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
