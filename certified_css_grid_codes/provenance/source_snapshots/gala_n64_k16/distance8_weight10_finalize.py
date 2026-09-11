"""Balance a certified weight-ten logical basis and audit the search coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from distance8_weight10_search import (CosetEnumerator, audit_basis, export_basis,
    logical_shifts, make_catalog, physical_shifts, seed_basis, unit_orbits)
from distance8_logical_handoff import simplify
from distance8_logical_search import coset_weights
from sparse_core import HERE, combine, pairing, permute, unpack
from certify import gf_rank, in_space, moved


def balance_options(options, fold, trials=128, random_seed=20260911):
    """Bounded coordinate descent; no claim of globally optimal overlap."""
    groups, count = len(options), len(options[0])
    assert all(len(o) == count for o in options)
    operators = np.asarray([[unpack([word, permute(word, fold)], 64) for word in group]
                            for group in options], dtype=np.int16)
    flat = operators.reshape(-1, 64)
    overlaps = flat @ flat.T
    np.fill_diagonal(overlaps, 0)
    costs = overlaps.reshape(groups*count, 2, groups*count, 2).max(axis=(1, 3))
    loads = operators.sum(axis=2).reshape(groups*count, 64)
    base = np.arange(groups)*count

    def score(choice):
        indices = base+choice
        load = loads[indices].sum(axis=0)
        return int(load.max()), int(costs[np.ix_(indices, indices)].max()), int(load@load)

    rng = np.random.default_rng(random_seed)
    best = None
    for trial in range(trials):
        choice = np.zeros(groups, dtype=int) if trial == 0 else rng.integers(count, size=groups)
        current = score(choice)
        for _ in range(12):
            improved = False
            for i in rng.permutation(groups):
                selected = choice[i]
                for option in range(count):
                    proposed = choice.copy()
                    proposed[i] = option
                    candidate = score(proposed)
                    if candidate < current:
                        selected, current = option, candidate
                        improved = True
                choice[i] = selected
            if not improved:
                break
        if best is None or current < best[0]:
            best = current, choice.copy()
    assert score(best[1]) == best[0]
    words = [options[i][int(c)] for i, c in enumerate(best[1])]
    return words, dict(score=list(best[0]), choices=best[1].tolist(), trials=trials,
                      random_seed=random_seed, options_per_logical=count,
                      peak_load_optimal=best[0][0] == 5,
                      peak_load_lower_bound=5, overlap_optimality='not proved')


def run(source, destination, trials=128, weight_eight_proof=None):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    started = time.monotonic()
    data, provenance, _ = simplify()
    proof_path = Path(weight_eight_proof or HERE/'distance8_code_study/weight_eight_grid_v1/report.json').resolve()
    proof = json.loads(proof_path.read_text())
    assert proof['all_weight_eight_canonical_grids'] == 0 and proof['seed_classes_tested'] == 192
    assert proof['source_candidate_sha256'] == provenance['candidate_source_sha256']
    summary = json.loads((source/'summary.json').read_text())
    saved_catalog = json.loads((source/'catalog.json').read_text())
    catalog, canonical = make_catalog(data)
    assert saved_catalog == catalog
    entries = [json.loads(line) for line in (source/'cosets.jsonl').read_text().splitlines()]
    assert len(entries) == len({r['seed'] for r in entries})
    eligible = {r['seed'] for r in catalog if r['saved_hadamard_permutation'] is not None}
    assert {r['seed'] for r in entries} == eligible
    assert {r['seed'] for r in entries if r['minimum_weight'] == 10} == set(summary['hit_seeds'])
    assert summary['status'] == 'saved_fold_weight_ten_found'
    # Re-enumerate every eligible coset with the pre-existing standalone
    # implementation, not merely the search's own minimum witness checks.
    for index, row in enumerate(entries):
        checked = coset_weights(data['hz'], combine(data['certificate']['z'], row['seed']), chunk=17)
        assert checked['minimum_weight'] == row['minimum_weight']
        assert checked['minimum_word'] == row['minimum_word']
        assert checked['enumerated'] == row['enumerated'] == 2**24
        assert {str(k): v for k, v in checked['histogram'].items()} == row['histogram']
        if (index+1)%64 == 0:
            print(json.dumps(dict(audited_cosets=index+1, total=len(entries))), flush=True)
    engine = CosetEnumerator(data['hz'])
    fold = data['certificate']['fold']
    shifts = physical_shifts(data)
    ranking, minima = [], {}
    for seed in summary['hit_seeds']:
        _, words = engine.scan(combine(data['certificate']['z'], seed), collect_weight=10)
        assert words and len(set(words)) == len(words)
        minima[str(seed)] = words
        options = [[permute(w, p) for w in words] for p in shifts]
        selected, balancing = balance_options(options, fold, trials=trials)
        ranking.append(dict(seed=seed, **balancing, z_words=selected))
        print(json.dumps(dict(balanced_seed=seed, score=balancing['score'], minimum_representatives=len(words))), flush=True)
    ranking.sort(key=lambda r: (*r['score'], r['seed'].bit_count(), r['seed']))
    best = ranking[0]
    audit = export_basis(data, best['seed'], best['z_words'][0], fold,
                         destination/'best', z_words=best['z_words'])
    report = dict(status='verified_weight_ten_optimal_saved_translation_grid',
                  source=str(source), source_summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
                  source_cosets_sha256=hashlib.sha256((source/'cosets.jsonl').read_bytes()).hexdigest(),
                  source_catalog_sha256=hashlib.sha256((source/'catalog.json').read_bytes()).hexdigest(),
                  source_candidate_sha256=provenance['candidate_source_sha256'],
                  source_distance_audit=provenance['distance'],
                  unit_seed_classes=32768, translation_orbits=2048,
                  saved_fold_compatible_orbits=len(eligible), cosets_reenumerated=len(entries),
                  weight_ten_orbits=len(ranking), selected_seed=best['seed'],
                  proof=dict(lower_bound=10, upper_bound=10,
                             odd_weights='All-qubit X and Z are stabilizers, forcing even pure CSS logical support',
                             weight_eight='Complete earlier 192-seed obstruction for the saved translations',
                             scope='Pure canonical CSS logical grids under the saved physical translations; common-basis H',
                             other_folds='Not needed: the smallest possible support is attained using the saved fold'),
                  ranking=ranking, all_weight_ten_seed_representatives=minima,
                  audit=audit, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    report['weight_eight_obstruction'] = dict(path=str(proof_path), sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('ranking', 'all_weight_ten_seed_representatives', 'audit')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--trials', type=int, default=128)
    parser.add_argument('--weight-eight-proof', help='Prior complete grid obstruction; defaults to the saved repository certificate')
    args = parser.parse_args()
    run(args.source, args.output, args.trials, args.weight_eight_proof)
