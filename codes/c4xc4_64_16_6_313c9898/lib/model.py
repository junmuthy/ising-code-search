"""Portable binary helpers; original function bodies unchanged."""
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


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

