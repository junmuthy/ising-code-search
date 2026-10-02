"""Pinned [[64,16,8]] inputs and exact binary structural checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
GALA = Path('/home/judah_unmuth/gala-code-search')
REVISION = 'd41842b014e2a8383fd2bba4b5322aa5aec3e553'
SOURCE = 'certified_css_grid_codes/n64_k16_d8'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def bootstrap():
    folder = ROOT / 'inputs'
    folder.mkdir(exist_ok=False)
    records = {}
    for name in ('candidate.json', 'metadata.json', 'certificates/distance_audit.json'):
        blob = subprocess.check_output(['git', '-C', str(GALA), 'show', f'{REVISION}:{SOURCE}/{name}'])
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        records[name] = digest(blob)
    save(folder / 'provenance.json', dict(repository=str(GALA), revision=REVISION, source=SOURCE, sha256=records))


def pivots(rows):
    result = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in result:
                result[p] = word
                break
            word ^= result[p]
    return result


def residual(word, space):
    for p in sorted(space, reverse=True):
        if word >> p & 1:
            word ^= space[p]
    return word


def same_space(a, b):
    space = pivots(b)
    return len(pivots(a)) == len(space) and all(not residual(word, space) for word in a)


def move(word, permutation):
    return sum(1 << permutation[q] for q in range(len(permutation)) if word >> q & 1)


def supports(rows):
    return [[q for q in range(64) if word >> q & 1] for word in rows]


def array(rows):
    return np.asarray([[(word >> q) & 1 for q in range(64)] for word in rows], dtype=np.uint8)


def relations(rows):
    basis, dependent = {}, []
    for i, word in enumerate(rows):
        combination = 1 << i
        while word:
            p = word.bit_length() - 1
            if p not in basis:
                basis[p] = (word, combination)
                break
            word ^= basis[p][0]
            combination ^= basis[p][1]
        if not word:
            dependent.append([j for j in range(len(rows)) if combination >> j & 1])
    return dependent


def load():
    provenance = json.loads((ROOT / 'inputs/provenance.json').read_text())
    for name, expected in provenance['sha256'].items():
        assert digest((ROOT / 'inputs' / name).read_bytes()) == expected
    data = json.loads((ROOT / 'inputs/candidate.json').read_text())
    assert (data['n'], data['k']) == (64, 16)
    def lift(word):
        result = [0] * 32
        for g in range(32):
            if word >> g & 1:
                i, j = divmod(g, 4)
                for h in range(32):
                    p, q = divmod(h, 4)
                    result[4 * ((i + p) % 8) + (j + q) % 4] ^= 1 << h
        return result
    a, b = lift(data['a']), lift(data['b'])
    hx = [aa | bb << 32 for aa, bb in zip(a, b)]
    fold = data['certificate']['fold']
    hz = [move(word, fold) for word in hx]
    checks = dict(X=hx, Z=hz)
    logicals = dict(X=data['certificate']['x'], Z=data['certificate']['z'])
    for kind, key in (('X', 'hx'), ('Z', 'hz')):
        assert same_space(checks[kind], data[key])
        assert len(pivots(checks[kind])) == 24
        assert len(set(checks[kind])) == 32
        assert all(word.bit_count() == 12 for word in checks[kind])
        assert all(word.bit_count() == 10 for word in logicals[kind])
    assert not np.any(array(hx) @ array(hz).T % 2)
    assert not np.any(array(hx) @ array(logicals['Z']).T % 2)
    assert not np.any(array(hz) @ array(logicals['X']).T % 2)
    assert np.array_equal(array(logicals['X']) @ array(logicals['Z']).T % 2, np.eye(16, dtype=np.uint8))
    assert [fold[fold[q]] for q in range(64)] == list(range(64))
    dependency = {kind: relations(rows) for kind, rows in checks.items()}
    assert all(len(rows) == 8 for rows in dependency.values())
    return dict(data=data, checks=checks, logicals=logicals, relations=dependency,
                supports={kind: supports(rows) for kind, rows in checks.items()},
                logical_supports={kind: supports(rows) for kind, rows in logicals.items()},
                fold=fold, provenance=provenance)


if __name__ == '__main__':
    bootstrap()
    code = load()
    print(json.dumps(dict(status='pinned_inputs_verified', revision=REVISION, n=64, k=16,
                          ranks=[24, 24], checks=[32, 32], dependencies=[8, 8])))
