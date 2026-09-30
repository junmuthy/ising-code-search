"""Pinned distance-six code; no changes to the archived code or logical basis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / 'fault_distance_milestone1'
spec = importlib.util.spec_from_file_location('d8_binary_helpers', SHARED / 'model.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
GALA, REVISION = shared.GALA, shared.REVISION
digest, save, pivots, residual = shared.digest, shared.save, shared.pivots, shared.residual
same_space, move, supports, array, relations = shared.same_space, shared.move, shared.supports, shared.array, shared.relations
SOURCE = 'certified_css_grid_codes/n64_k16_d6'


def archived(name):
    return subprocess.check_output(['git', '-C', str(GALA), 'show', f'{REVISION}:{SOURCE}/{name}'])


def load(presentation='independent', seed=None):
    name = ('alternatives/translation_closed/' if presentation == 'redundant' else '') + 'candidate.json'
    blob = archived(name)
    data = json.loads(blob)
    fold = data['certificate']['fold']
    checks = dict(X=data['hx'], Z=data['hz'])
    if seed is not None:
        assert presentation == 'independent'
        catalog = json.loads(archived('certificates/check_weight_enumeration.json'))
        words = catalog['hx']['low_weight_rows'][:]
        rng = random.Random(seed)
        rng.shuffle(words)
        words.sort(key=int.bit_count)
        basis, selected = {}, []
        for word in words:
            if residual(word, basis):
                selected.append(word)
                basis = pivots(selected)
        checks = dict(X=selected, Z=[move(word, fold) for word in selected])
    logicals = dict(X=data['certificate']['x'], Z=data['certificate']['z'])
    assert (data['n'], data['k']) == (64, 16)
    for kind, key in (('X', 'hx'), ('Z', 'hz')):
        assert same_space(checks[kind], data[key]) and len(pivots(checks[kind])) == 24
        assert same_space([move(w, data['px']) for w in checks[kind]], checks[kind])
        assert same_space([move(w, data['py']) for w in checks[kind]], checks[kind])
    assert same_space([move(w, fold) for w in checks['X']], checks['Z'])
    assert same_space([move(w, fold) for w in checks['Z']], checks['X'])
    assert [fold[fold[q]] for q in range(64)] != list(range(64))
    assert [fold[fold[fold[fold[q]]]] for q in range(64)] == list(range(64))
    assert not np.any(array(checks['X']) @ array(checks['Z']).T % 2)
    for kind, other in (('X', 'Z'), ('Z', 'X')):
        assert not np.any(array(checks[kind]) @ array(logicals[other]).T % 2)
    assert np.array_equal(array(logicals['X']) @ array(logicals['Z']).T % 2, np.eye(16, dtype=np.uint8))
    identity = digest(json.dumps(checks, sort_keys=True).encode())[:16]
    return dict(data=data, checks=checks, logicals=logicals, fold=fold,
                supports={s: supports(rows) for s, rows in checks.items()},
                logical_supports={s: supports(rows) for s, rows in logicals.items()},
                relations={s: relations(rows) for s, rows in checks.items()},
                presentation=presentation, basis_seed=seed, basis_id=identity,
                provenance=dict(repository=str(GALA), revision=REVISION, source=SOURCE,
                                sha256={name: digest(blob)}))
