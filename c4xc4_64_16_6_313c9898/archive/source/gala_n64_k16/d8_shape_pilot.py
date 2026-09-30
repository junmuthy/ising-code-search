"""A bounded extension beyond the one-edit neighborhoods: normalized shapes.

Classify weight-four coefficient supports under translations and group
automorphisms, discard exact fixed-block obstructions, and test each surviving
shape with BOTH known d=8 folds on BOTH sides. This does not exhaust all joint
coefficient/fold classes: only the two specified folds are searched here.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import d8_local_complete as runner
from d8_partner_search import GROUP, SEEDS, seed_data
from algebra import bits


def canonical_translation(word):
    support = list(bits(word))
    return min(sum(1 << GROUP.mul(g, t) for g in support) for t in range(32))


def shape_manifest():
    classes = {}
    for triple in itertools.combinations(range(1, 32), 3):
        word = canonical_translation(1 | sum(1 << g for g in triple))
        if word not in classes:
            classes[word] = runner.fixed_distance_obstruction(word)
    survivors = {word for word, obstruction in classes.items() if obstruction is None}
    automorphisms = sorted({tuple(v%32 for v in f[:32]) for f in GROUP.folds()})
    remaining = survivors.copy()
    orbits = []
    while remaining:
        seed = min(remaining)
        orbit = {canonical_translation(sum(1 << p[g] for g in bits(seed))) for p in automorphisms}
        assert orbit <= survivors and orbit <= remaining
        remaining -= orbit
        orbits.append(dict(representative=seed, translated_classes=sorted(orbit)))
    assert len(automorphisms) == 128
    return dict(weight=4, anchored_supports_examined=4495, translation_classes=len(classes),
                group_automorphisms=len(automorphisms), surviving_translation_classes=len(survivors),
                surviving_shape_classes=len(orbits), orbits=orbits,
                obstructions={word: proof for word, proof in classes.items() if proof is not None},
                limitation='Only two saved folds will be combined with these shapes; not a complete family no-go.')


def run(args):
    manifest = shape_manifest()
    configs = []
    for orbit in manifest['orbits']:
        for seed_id in SEEDS:
            fold = seed_data(seed_id)['certificate']['fold']
            for side in ('a', 'b'):
                configs.append(dict(seed_id=seed_id, fixed=orbit['representative'], fixed_side=side,
                                    fixed_weight=4, fold=fold, mutation='normalized_weight_four_shape'))
    print(json.dumps(dict(shape_classes=len(manifest['orbits']), slices=len(configs))), flush=True)
    # Explicit dependency injection reuses the exact runner without changing
    # its persisted source or the original neighborhood configuration builder.
    old_configurations = runner.configurations
    try:
        runner.configurations = lambda phase: configs
        runner.run(args)
    finally:
        runner.configurations = old_configurations
    manifest['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (Path(args.output)/'shape_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', default='normalized_shapes')
    parser.add_argument('--maxweight', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--covered', action='append', default=[])
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
