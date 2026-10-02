"""Explicit small group algebras and the faithful two-dimensional lift."""
import itertools
import numpy as np

from sparse_core import combine
from algebra import bits, transpose


class SmallGroup:
    def __init__(self, family, alpha=0, beta=0):
        self.family, self.alpha, self.beta = family, alpha, beta
        if family == 'extension':
            self.states = list(itertools.product(range(4), range(4), range(2)))
            def multiply(a, b):
                i, j, t = a
                p, q, r = b
                return ((i+p)%4, (j+q)%4, (t+r+j*p+alpha*((i+p)//4)+beta*((j+q)//4))%2)
        else:
            moduli = {'c8c4': (8, 4), 'c4c4c2': (4, 4, 2), 'c4c4': (4, 4)}[family]
            self.states = list(itertools.product(*map(range, moduli)))
            def multiply(a, b):
                return tuple((v+w)%m for v, w, m in zip(a, b, moduli))
        self.order = len(self.states)
        index = {v: i for i, v in enumerate(self.states)}
        self.table = np.asarray([[index[multiply(a, b)] for b in self.states] for a in self.states])
        self.inverse = [next(j for j in range(self.order) if self.table[i, j] == 0) for i in range(self.order)]
        self.generators = [index[tuple(int(i == j) for i in range(len(self.states[0])))] for j in range(len(self.states[0]))]
        self.monomials = []
        for g in range(self.order):
            rows = [0]*self.order
            for h in range(self.order):
                rows[int(self.table[g, h])] = 1 << h
            self.monomials.append(rows)

    def mul(self, a, b):
        return int(self.table[a, b])

    def polynomial_mul(self, a, b):
        out = 0
        for g in bits(a):
            for h in bits(b):
                out ^= 1 << self.mul(g, h)
        return out

    def lift(self, a):
        out = [0]*self.order
        for g in bits(a):
            out = [x^y for x, y in zip(out, self.monomials[g])]
        return out

    def checks(self, a, b):
        return two_block(self.lift(a), self.lift(b))

    def right(self, g):
        return [s*self.order+self.mul(h, g) for s in range(2) for h in range(self.order)]

    def physical(self):
        x, y = self.generators[:2]
        central = self.generators[2] if len(self.generators) == 3 else self.mul(self.mul(x, x), self.mul(x, x))
        return self.right(x), self.right(y), self.right(central)

    def automorphism(self, images):
        result = []
        for state in self.states:
            value = 0
            for exponent, image in zip(state, images):
                for _ in range(exponent):
                    value = self.mul(value, image)
            result.append(value)
        if len(set(result)) != self.order:
            return None
        if not np.array_equal(np.asarray(result)[self.table], self.table[np.ix_(result, result)]):
            return None
        return result

    def folds(self):
        x, y = self.generators[:2]
        if self.family == 'c8c4':
            images = itertools.product(range(self.order), repeat=2)
        else:
            central = self.generators[2] if len(self.generators) == 3 else 0
            xs = [x, self.inverse[x], self.mul(x, central), self.mul(self.inverse[x], central)]
            ys = [y, self.inverse[y], self.mul(y, central), self.mul(self.inverse[y], central)]
            images = [v for a, b in itertools.product(xs, ys)
                      for v in ((a, b, central), (b, a, central))]
        seen = set()
        for imageset in images:
            p = self.automorphism(imageset)
            if p is None:
                continue
            for swap in (False, True):
                fold = tuple((s^swap)*self.order+p[h] for s in range(2) for h in range(self.order))
                if fold not in seen:
                    seen.add(fold)
                    yield list(fold)


def two_block(a, b):
    size = len(a)
    return ([x | y << size for x, y in zip(a, b)],
            [x | y << size for x, y in zip(transpose(b, size), transpose(a, size))])


RING = SmallGroup('c4c4')


def mat_mul(a, b):
    return [RING.polynomial_mul(a[2*i], b[j]) ^ RING.polynomial_mul(a[2*i+1], b[2+j])
            for i in range(2) for j in range(2)]


def decode(v):
    return [(v >> (16*i)) & 65535 for i in range(4)]


def commutator(a, b):
    return sum((x^y) << (16*i) for i, (x, y) in enumerate(zip(mat_mul(a, b), mat_mul(b, a))))


def lift_gl(a):
    entries = [RING.lift(v) for v in a]
    return [entries[2*r][g] | entries[2*r+1][g] << 16 for r in range(2) for g in range(16)]


def translations():
    return ([16*s+4*((i+1)%4)+j for s in range(4) for i in range(4) for j in range(4)],
            [16*s+4*i+(j+1)%4 for s in range(4) for i in range(4) for j in range(4)])


def gl_folds():
    for sigma in ((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0), (1, 2, 3, 0), (3, 0, 1, 2)):
        for a, b, c, d in ((1, 0, 0, 1), (-1, 0, 0, -1), (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 2, -1)):
            yield [16*sigma[s]+4*((a*i+b*j)%4)+(c*i+d*j)%4
                   for s in range(4) for i in range(4) for j in range(4)]


def initial_control():
    h = [(1 << g) | (1 << (16+g)) for g in range(16)]
    h += [(1 << (32+g)) | (1 << (40+g)) for g in range(8)]
    z = [1 << (48+g) for g in range(16)]
    return h, z


def transvection(h, z, a, b, order):
    from algebra import pairing, commute
    entries = [a, a, b, b]
    v = [sum(1 << (16*s+RING.mul(g, t)) for s in range(4) for t in bits(entries[order[s]])) for g in range(16)]
    assert commute(v, v)
    def move(rows):
        return [row^combine(v, coefficients) for row, coefficients in zip(rows, pairing(rows, v))]
    return move(h), move(z)
