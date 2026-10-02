"""Exact low-weight logical analysis of the fixed [[64,16,8]] code."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from sparse_core import basis, combine, commute, permute, rank, reduce, unpack
from algebra import canonical_x, compose, pairing, permutation_matrix, power, transpose
from distance8_logical_handoff import SOURCE, simplify
from certify import gf_rank, in_space


def coset_weights(stabilizers, representative, width=64, chunk=32):
    """Enumerate every element of one stabilizer coset, using bounded memory."""
    generators = list(basis(stabilizers).values())
    halves = []
    for part in (generators[:len(generators)//2], generators[len(generators)//2:]):
        values = np.zeros(1, dtype=np.uint64)
        for row in part:
            values = np.concatenate((values, values ^ np.uint64(row)))
        halves.append(values)
    histogram = np.zeros(width+1, dtype=np.int64)
    best_weight, best_word = width+1, None
    for start in range(0, len(halves[1]), chunk):
        words = halves[1][start:start+chunk, None] ^ halves[0][None, :] ^ np.uint64(representative)
        weights = np.bitwise_count(words)
        histogram += np.bincount(weights.ravel(), minlength=width+1)
        w = int(weights.min())
        if w <= best_weight:
            candidate = int(words[weights == w].min())
            if w < best_weight or candidate < best_word:
                best_weight, best_word = w, candidate
    assert int(histogram.sum()) == 2**len(generators)
    assert reduce(best_word ^ representative, basis(stabilizers)) == 0
    return dict(stabilizer_rank=len(generators), enumerated=int(histogram.sum()),
                minimum_weight=best_weight, minimum_word=best_word,
                histogram={int(w): int(c) for w, c in enumerate(histogram) if c})


def all_weight_eight(checks, width=64):
    """Enumerate each zero-syndrome weight-eight support exactly once.

    Its unique split into its smallest four and largest four positions is
    accepted. Other equal-syndrome quadruple pairs are deliberately ignored.
    """
    columns = transpose(checks, width)
    earlier = defaultdict(list)
    words = []
    for q in itertools.combinations(range(width), 4):
        syndrome = columns[q[0]] ^ columns[q[1]] ^ columns[q[2]] ^ columns[q[3]]
        group = earlier[syndrome]
        for left in group:
            if left[3] < q[0]:
                words.append(sum(1 << i for i in left+q))
        group.append(q)
    assert len(set(words)) == len(words)
    return sorted(words)


def run(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    data, handoff, _ = simplify()
    x, z = data['certificate']['x'], data['certificate']['z']
    # Translation transitivity and the fold transport this minimum to all
    # 32 fixed-basis single-logical cosets; do not confuse it with code distance.
    fixed = coset_weights(data['hz'], z[0])
    print(json.dumps(dict(fixed_basis_single_Z_minimum=fixed['minimum_weight'],
                          elapsed_seconds=time.monotonic()-started)), flush=True)
    words = all_weight_eight(data['hx'])
    classes = defaultdict(list)
    stabilizers = basis(data['hz'])
    for word in words:
        assert word.bit_count() == 8 and commute(data['hx'], [word])
        assert reduce(word, stabilizers)
        label = sum(((word & v).bit_count() % 2) << i for i, v in enumerate(x))
        assert label and reduce(word ^ combine(z, label), stabilizers) == 0
        classes[label].append(word)
    class_rank = rank(list(classes))
    assert gf_rank(unpack(list(classes), 16)) == class_rank
    # Every individual returned support is checked independently in arrays.
    checks, stabs = unpack(data['hx'], 64), unpack(data['hz'], 64)
    word_array = unpack(words, 64)
    assert not np.any(checks @ word_array.T % 2)
    assert in_space(word_array ^ unpack([combine(z, sum(((w & v).bit_count()%2) << i
        for i, v in enumerate(x))) for w in words], 64), stabs)
    fold = data['certificate']['fold']
    x_words = [permute(w, fold) for w in words]
    assert commute(data['hz'], x_words)
    witnesses = {str(label): dict(count=len(values), representative=min(values),
                                 logical_labels=[list(divmod(i, 4)) for i in range(16) if label >> i & 1])
                 for label, values in sorted(classes.items())}
    report = dict(status='complete_fixed_code_weight_eight_analysis', n=64, k=16, d=8,
                  fixed_basis_single_logical_coset=fixed,
                  fixed_basis_all_32_single_logical_minimum=fixed['minimum_weight'],
                  transfer_argument='Verified transitive logical translations and involutive Hadamard fold',
                  weight_eight_pure_Z_operators=len(words), weight_eight_pure_X_operators=len(x_words),
                  weight_eight_Z_logical_classes=len(classes), weight_eight_Z_logical_span_rank=class_rank,
                  weight_eight_class_label_parities=dict(Counter(label.bit_count()%2 for label in classes)),
                  all_weight_eight_CSS_logical_basis_possible=class_rank == 16,
                  basis_possible_flag_meaning='False is an exact obstruction; True only passes a necessary spanning test',
                  classes=witnesses, source_candidate=handoff['candidate_source'],
                  source_candidate_sha256=handoff['candidate_source_sha256'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    destination.mkdir(parents=True)
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (destination/'weight_eight_words.json').write_text(json.dumps(dict(Z=words, X=x_words), indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('classes', 'fixed_basis_single_logical_coset')}), flush=True)
    return report


def check_equivariant_bases(catalog, destination):
    """Exhaust every weight-eight cyclic seed class for the saved translations."""
    catalog, destination = Path(catalog), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    data, provenance, _ = simplify()
    old = json.loads((catalog/'report.json').read_text())
    words = json.loads((catalog/'weight_eight_words.json').read_text())
    xold, zold = data['certificate']['x'], data['certificate']['z']
    fold = data['certificate']['fold']
    translations = [compose(power(data['px'], i), power(data['py'], j))
                    for i in range(4) for j in range(4)]
    x_table = {pairing([w], zold)[0]: w for w in words['X']}
    results = []
    for label, record in old['classes'].items():
        label = int(label)
        # In F2[C4 x C4], precisely the odd-parity elements are units.
        if label.bit_count()%2 == 0:
            continue
        z = [permute(record['representative'], q) for q in translations]
        assert rank(pairing(z, xold)) == 16
        x = canonical_x(z, xold)
        # Independent array check of the uniquely determined dual cosets.
        assert np.array_equal(unpack(x, 64) @ unpack(z, 64).T % 2, np.eye(16, dtype=int))
        x_labels = pairing(x, zold)
        available = [v in x_table for v in x_labels]
        h = pairing([permute(v, fold) for v in z], z)
        h_is_permutation = permutation_matrix(h)
        if h_is_permutation:
            assert all(available)
        results.append(dict(seed_logical_class=label, seed_word=record['representative'],
                            dual_X_classes=x_labels, dual_weight_eight_available=available,
                            saved_H_is_individual_up_to_permutation=h_is_permutation))
    assert len(results) == 192
    compatible = sum(all(r['dual_weight_eight_available']) for r in results)
    report = dict(status='complete_saved_translation_weight_eight_basis_test',
                  seed_classes_tested=len(results), all_weight_eight_canonical_grids=compatible,
                  saved_H_compatible_seeds=sum(r['saved_H_is_individual_up_to_permutation'] for r in results),
                  scope='Pure CSS logical bases with the saved regular C4 x C4 translation action, arbitrary stabilizer dressing',
                  not_excluded=['Different physical permutation representations',
                                'Non-equivariant bases', 'Mixed-Pauli logical bases',
                                'Weight-eight products of logicals'],
                  explanation='All 192 unit seed classes with weight-eight Z representatives are tested; the dual X cosets are unique',
                  source_catalog=str(catalog.resolve()),
                  source_catalog_sha256=hashlib.sha256((catalog/'report.json').read_bytes()).hexdigest(),
                  source_words_sha256=hashlib.sha256((catalog/'weight_eight_words.json').read_bytes()).hexdigest(),
                  source_candidate_sha256=provenance['candidate_source_sha256'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results)
    destination.mkdir(parents=True)
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--grid-input', help='Use a complete weight-eight catalog to test equivariant canonical bases')
    args = parser.parse_args()
    if args.grid_input:
        check_equivariant_bases(args.grid_input, args.output)
    else:
        run(args.output)
