"""Audit direct worker slices and assemble the complete equivalence proof."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from audit_joint_fold import audit
from joint_equivalence_complete import read_rows
from joint_fold_orbits import transport
from joint_fold_search import (GROUP, automorphisms, configurations, config_key,
                               make_manifest)


def audit_worker(folder):
    folder = Path(folder)
    destination = folder/'rejection_audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    with (folder/'audit.log').open('x') as log, redirect_stdout(log):
        result = audit(folder, allow_partial=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    return dict(folder=str(folder), audit=str(destination), status=result['status'],
                complete_slices=result['complete_partner_sets'], seconds=result['elapsed_seconds'])


def verify(root, checkpoint_audit, worker_audits):
    root = Path(root)
    started = time.monotonic()
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == 'complete_joint_equivalence_scope'
    assert summary['accepted_direct_slices'] == 0
    records = json.loads((root/'coverage.json').read_text())
    manifest = make_manifest(6)
    configs = list(configurations(manifest))
    assert len(records) == len(configs) == 3264
    assert sorted(r['target_index'] for r in records) == list(range(len(configs)))
    certificates, hashes, visited = set(), {}, {}
    pending = [Path(checkpoint_audit), *map(Path, worker_audits)]
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        blob = path.read_bytes()
        report = json.loads(blob)
        assert report['status'] in ('verified_complete_joint_negative', 'verified_partial_joint_negative')
        assert report['minimum_distance'] == 6 and report['maximum_displayed_check_weight'] == 10
        visited[path] = hashlib.sha256(blob).hexdigest()
        previous = report.get('previous_audit')
        if previous:
            previous = Path(previous)
            assert hashlib.sha256(previous.read_bytes()).hexdigest() == report['previous_audit_sha256']
            pending.append(previous)
        for folder, data in report['sources'].items():
            certificates.update((folder, sid) for sid in data['selected_slice_ids'])
            bounds = hashes.setdefault(folder, {})
            n = data['candidate_rows']
            if n in bounds:
                assert bounds[n] == data['candidates_prefix_sha256']
            bounds[n] = data['candidates_prefix_sha256']
    # Check every reused certificate's exact input prefix in a single pass
    # per journal. Uncommitted tails are deliberately outside these proofs.
    for folder, bounds in hashes.items():
        digest = hashlib.sha256()
        maximum = max(bounds)
        count = 0
        with (Path(folder)/'candidates.jsonl').open('rb') as stream:
            for line in stream:
                if count >= maximum:
                    break
                digest.update(line)
                count += 1
                if count in bounds:
                    assert digest.hexdigest() == bounds[count]
        assert count == maximum
    sources = {}
    auts = set(automorphisms())
    direct, transported, eligible_direct, eligible_represented = set(), 0, 0, 0
    for record in records:
        target = configs[record['target_index']]
        source = configs[record['source_index']]
        assert record['target_config_key'] == config_key(target)
        key = record['source_folder'], record['source_slice_id']
        assert key in certificates
        folder = key[0]
        if folder not in sources:
            sources[folder] = {r['slice_id']: r for r in read_rows(Path(folder)/'slices.jsonl')}
        saved = sources[folder][key[1]]
        assert saved['status'] == 'complete_slice' and config_key(saved) == config_key(source)
        assert saved['partner_weight_ceiling'] == 6
        assert not saved['candidate_counts'].get('accepted', 0)
        eligible_represented += saved['eligible_partners']
        if key not in direct:
            direct.add(key)
            eligible_direct += saved['eligible_partners']
        q = record['physical_permutation']
        assert sorted(q) == list(range(64))
        alpha = tuple(GROUP.mul(q[g] % 32, GROUP.inverse[q[0] % 32]) for g in range(32))
        assert alpha in auts and alpha[16] == 16
        a, b = (source['fixed'], 0) if source['fixed_side'] == 'a' else (0, source['fixed'])
        aa, bb, pp = transport(a, b, source['fold'], q)
        assert pp == target['fold']
        assert (aa, bb) == ((target['fixed'], 0) if target['fixed_side'] == 'a' else (0, target['fixed']))
        # Linearity plus a bijection of all monomial coefficient images proves
        # that EVERY weight-bounded partner is transported, not only a sample.
        images = []
        for g in range(32):
            a, b = ((source['fixed'], 1 << g) if source['fixed_side'] == 'a'
                    else (1 << g, source['fixed']))
            aa, bb, pp = transport(a, b, source['fold'], q)
            assert pp == target['fold']
            assert (aa if target['fixed_side'] == 'a' else bb) == target['fixed']
            image = bb if target['fixed_side'] == 'a' else aa
            assert image.bit_count() == 1
            images.append(image)
        assert len(set(images)) == 32
        if record['kind'] == 'joint_equivalence':
            transported += 1
        else:
            assert record['kind'] == 'direct' and q == list(range(64))
    assert transported == summary['transported_configurations']
    assert len(direct) == summary['direct_configurations']
    return dict(status='verified_complete_joint_equivalence_negative', n=64, k=16,
                minimum_distance=6, maximum_displayed_check_weight_excluded=10,
                coefficient_shapes=17, configurations=len(configs), joint_orbits=1092,
                direct_configurations=len(direct), transported_configurations=transported,
                directly_enumerated_eligible_pairs=eligible_direct,
                eligible_pairs_represented_by_full_scope=eligible_represented,
                scope=manifest['scope'], independent_basis_extension='See cyclic-module proof in README.md.',
                source_audits={str(p): h for p, h in visited.items()},
                coverage_sha256=hashlib.sha256((root/'coverage.json').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-started)


def watch(args):
    root = Path(args.root)
    schedule = json.loads((root/'schedule.json').read_text())
    folders = [root/f'shard_{i:03d}' for i in range(len(schedule['shards']))]
    submitted, futures, results = set(), {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        while len(results) < len(folders):
            if time.monotonic()-started > args.seconds:
                raise TimeoutError('Worker audit watch limit reached; partial reports are preserved.')
            for folder in folders:
                if folder in submitted or not (folder/'summary.json').exists():
                    continue
                try:
                    summary = json.loads((folder/'summary.json').read_text())
                except json.JSONDecodeError:
                    continue
                if summary['status'] != 'complete_configured_shard':
                    continue
                assert not summary['candidate_counts'].get('accepted', 0), 'A hit needs a positive-code audit.'
                futures[pool.submit(audit_worker, str(folder))] = folder
                submitted.add(folder)
            for future in list(futures):
                if future.done():
                    result = future.result()
                    results.append(result)
                    del futures[future]
                    print(json.dumps(result), flush=True)
            if len(results) < len(folders):
                time.sleep(1)
    (root/'worker_audits.json').write_text(json.dumps(results, indent=2)+'\n')
    while not Path(args.checkpoint_audit).exists() or not (root/'coverage.json').exists():
        assert time.monotonic()-started < args.seconds
        time.sleep(1)
    result = verify(root, args.checkpoint_audit, [r['audit'] for r in results])
    (root/'full_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_audits'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint-audit', required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--seconds', type=float, default=2400)
    watch(parser.parse_args())
