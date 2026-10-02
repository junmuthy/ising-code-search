"""Recheck joint-search coverage and negative certificates from saved journals.

All partner sets and all rejection witnesses are checked. Selected certificates
are additionally checked with an independent array-based GF(2) implementation.
Duplicate references are resolved within their original journal, even when
only some of that journal's slices contribute to the final coverage.
"""
import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from joint_fold_search import (GROUP, config_key, configurations, digest,
                               make_manifest, partner_space, enumerate_kernel,
                               sparse_supports, checks, space_key)
from algebra import basis, commute, nullspace, permute, rank, reduce, unpack


def read_slices(path, trailing_partial=False):
    lines = Path(path).read_text().splitlines(keepends=True)
    if trailing_partial and lines and not lines[-1].endswith('\n'):
        lines.pop()
    return [json.loads(line) for line in lines]


def audit(folder, array_stride=1000, allow_partial=False, live=False, previous_checks=None):
    from certify import gf_rank
    folder = Path(folder).resolve()
    started = time.monotonic()
    manifest = json.loads((folder/'manifest.json').read_text())
    rows = read_slices(folder/'slices.jsonl', trailing_partial=live)
    if live:
        assert all(r['status'] in ('complete_slice', 'covered_by_prior_certificate') for r in rows)
        ceilings = {r['fixed_weight']+r['partner_weight_ceiling'] for r in rows if r['status'] == 'complete_slice'}
        assert len(ceilings) == 1
        summary = dict(status='bounded_scope', parameters=dict(min_distance=manifest['minimum_distance'],
                                                               maxweight=ceilings.pop()),
                       configurations_processed=len(rows))
    else:
        summary = json.loads((folder/'summary.json').read_text())
    if summary['status'] == 'interrupted_checkpoint':
        # The summary seals a prefix, not the whole mutable journal. A late
        # completed-slice flush must not expand the scope being certified.
        assert len(rows) >= summary['configurations_processed']
        rows = rows[:summary['configurations_processed']]
    partial_statuses = ('bounded_scope', 'interrupted_checkpoint', 'complete_configured_shard')
    assert summary['status'] == 'complete_joint_scope' or ((allow_partial or live) and summary['status'] in partial_statuses)
    rebuilt = make_manifest(summary['parameters']['min_distance'])
    if live:
        summary['manifest_sha256'] = digest(rebuilt)
    assert digest(rebuilt) == summary['manifest_sha256']
    # JSON object keys become strings; compare the round-tripped manifest,
    # rather than sorting its integer-keyed proof maps in a different order.
    assert json.loads(json.dumps(rebuilt)) == manifest
    configs = {config_key(c): c for c in configurations(manifest)}
    assert len(rows) == summary['configurations_processed']
    assert {config_key(r) for r in rows} <= set(configs)
    if summary['status'] == 'complete_joint_scope':
        assert len(rows) == len(configs)
    assert len({r['slice_id'] for r in rows}) == len(rows)
    source_slices, selected = {}, defaultdict(dict)
    for record in rows:
        path = folder/'slices.jsonl'
        while record['status'] == 'covered_by_prior_certificate':
            path = Path(record['prior'])
            if path not in source_slices:
                source_slices[path] = {r['slice_id']: r for r in read_slices(path)}
            record = source_slices[path][record['prior_slice_id']]
        assert record['status'] == 'complete_slice'
        assert config_key(record) in configs
        selected[path.parent][record['slice_id']] = record
    counts, checked, arrays, hashes = Counter(), Counter(), Counter(), {}
    old_sources = {}
    if previous_checks is not None:
        old = json.loads(Path(previous_checks).read_text())
        assert old['status'] in ('verified_complete_joint_negative', 'verified_partial_joint_negative')
        assert old['minimum_distance'] == summary['parameters']['min_distance']
        assert old['maximum_displayed_check_weight'] == summary['parameters']['maxweight']
        old_sources = old['sources']
    reused = 0
    central = GROUP.physical()[2]
    for source, chosen in selected.items():
        original_slices = {r['slice_id']: r for r in
                           (rows if live and source == folder else read_slices(source/'slices.jsonl'))}
        for r in original_slices.values():
            r['_fold_key'] = tuple(r['fold'])
        partners, statuses = defaultdict(set), defaultdict(Counter)
        references = OrderedDict()
        matrix_cache = OrderedDict()
        prefix_hash = hashlib.sha256()
        prior = old_sources.get(str(source), {})
        previous_rows = prior.get('candidate_rows', 0)
        if live and source == folder:
            limit = sum(r.get('processed_eligible_partners', 0) for r in rows)
        else:
            source_summary = json.loads((source/'summary.json').read_text())
            limit = source_summary.get('candidate_journal_completed_prefix_rows')
        journal_rows = 0
        with (source/'candidates.jsonl').open() as stream:
            for index, line in enumerate(stream):
                if limit is not None and index >= limit:
                    break
                prefix_hash.update(line.encode())
                journal_rows += 1
                if journal_rows == previous_rows:
                    assert prefix_hash.hexdigest() == prior['candidates_prefix_sha256']
                row = json.loads(line)
                r = original_slices[row['slice_id']]
                key = row['slice_id'], row['a'], row['b']
                assert key not in references
                status = row['status']
                refs = row.get('reference')
                if status == 'duplicate_code_fold':
                    rk = refs['slice_id'], refs['a'], refs['b']
                    assert rk in references
                    prior_code, prior_fold, prior_status = references[rk]
                    assert prior_code == row['code_id'] and prior_fold == r['_fold_key']
                    assert prior_status == refs['status']
                    references.move_to_end(rk)
                else:
                    # Mirror the search's bounded LRU of independently
                    # evaluated rows; duplicates never become reference roots.
                    references[key] = row['code_id'], r['_fold_key'], status
                    if len(references) > 100000:
                        references.popitem(last=False)
                if row['slice_id'] in chosen:
                    v = row['b' if r['fixed_side'] == 'a' else 'a']
                    assert row[r['fixed_side']] == r['fixed']
                    assert v not in partners[row['slice_id']]
                    partners[row['slice_id']].add(v)
                    statuses[row['slice_id']][status] += 1
                    counts[status] += 1
                if journal_rows <= previous_rows:
                    reused += 1
                    continue
                # Two coefficients alone do not determine HZ; retain the fold.
                mk = row['a'], row['b'], r['_fold_key']
                if mk not in matrix_cache:
                    hx, hz = checks(row['a'], row['b'], r['fold'])
                    matrix_cache[mk] = hx, hz, basis(hz)
                    if len(matrix_cache) > 256:
                        matrix_cache.popitem(last=False)
                hx, hz, zstab = matrix_cache[mk]
                assert space_key(hx, hz) == row['code_id']
                assert commute(hx, hz)
                assert max(v.bit_count() for v in hx+hz) <= summary['parameters']['maxweight']
                if status == 'duplicate_code_fold':
                    continue
                sample = index % array_stride == 0 or arrays[status] == 0
                if 'witness' in row:
                    word = sum(1 << q for q in row['witness'])
                    assert len(set(row['witness'])) == len(row['witness'])
                    assert 0 < word.bit_count() < summary['parameters']['min_distance']
                    assert commute(hx, [word]) and reduce(word, zstab)
                    if sample:
                        h, z = unpack(hx, 64), unpack(hz, 64)
                        e = unpack([word], 64)
                        assert not np.any(h@e.T % 2)
                        assert gf_rank(np.vstack((z, e))) > gf_rank(z)
                elif status == 'wrong_combined_rank':
                    assert rank(hx) != 24 or rank(hz) != 24
                    if sample:
                        assert gf_rank(unpack(hx, 64)) != 24 or gf_rank(unpack(hz, 64)) != 24
                elif status == 'wrong_logical_relations':
                    # Tx^4 is the only nontrivial relation to test here:
                    # Ty^4=1 and [Tx,Ty]=1 already hold physically.
                    witness = None
                    for h, s in ((hx, hz), (hz, hx)):
                        sb = basis(s)
                        for v in nullspace(h, 64):
                            moved = permute(v, central)^v
                            if reduce(moved, sb):
                                witness = h, s, v, moved
                                break
                        if witness is not None:
                            break
                    assert witness is not None
                    if sample:
                        h, s, v, moved = witness
                        hh, ss, vv = unpack(h, 64), unpack(s, 64), unpack([v], 64)
                        assert not np.any(hh@vv.T % 2)
                        assert gf_rank(np.vstack((ss, unpack([moved], 64)))) > gf_rank(ss)
                else:
                    # Do not silently certify an unimplemented reason or a hit.
                    raise AssertionError(f'Additional audit required for {status}')
                checked[status] += 1
                if sample:
                    arrays[status] += 1
                if index and index % 100000 == 0:
                    print(json.dumps(dict(source=str(source), rows=index, seconds=round(time.monotonic()-started, 1))), flush=True)
        for sid, r in chosen.items():
            info, kernel = partner_space(r['fixed'], r['fold'], r['fixed_side'])
            assert info['kernel'] == r['kernel']
            cap = summary['parameters']['maxweight']-r['fixed_weight']
            if len(kernel) <= 22:
                expected, _ = enumerate_kernel(kernel, cap)
            else:
                expected, _ = sparse_supports(info['equation_images'], cap)
            assert partners[sid] == set(expected)
            assert len(expected) == r['eligible_partners'] == r['processed_eligible_partners']
            assert dict(statuses[sid]) == r['candidate_counts']
        assert journal_rows >= previous_rows
        if limit is not None:
            assert journal_rows == limit
        hashes[str(source)] = dict(candidate_rows=journal_rows,
                                   candidates_prefix_sha256=prefix_hash.hexdigest(),
                                   selected_slice_ids=sorted(chosen))
    complete = summary['status'] == 'complete_joint_scope'
    return dict(status='verified_complete_joint_negative' if complete else 'verified_partial_joint_negative',
                minimum_distance=summary['parameters']['min_distance'],
                maximum_displayed_check_weight=summary['parameters']['maxweight'],
                configurations=len(configs), complete_partner_sets=len(rows),
                candidate_counts=dict(counts), rejection_certificates_rechecked=dict(checked),
                independent_array_crosschecks=dict(arrays), array_stride=array_stride,
                reused_verified_journal_rows=reused,
                previous_audit=str(Path(previous_checks).resolve()) if previous_checks else None,
                previous_audit_sha256=hashlib.sha256(Path(previous_checks).read_bytes()).hexdigest() if previous_checks else None,
                sources=hashes, elapsed_seconds=time.monotonic()-started,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--array-stride', type=int, default=1000)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--live', action='store_true', help='Audit a completed prefix of a still-running journal.')
    parser.add_argument('--previous-checks', help='Reuse hash-verified rejection checks from an earlier prefix audit.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.root, args.array_stride, args.allow_partial, args.live, args.previous_checks)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}), flush=True)
