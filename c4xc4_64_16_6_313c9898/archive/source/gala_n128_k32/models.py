"""Faithful GL(2,2) and central-extension lifts, all in explicit bit order."""
from __future__ import annotations

import itertools
import numpy as np

from algebra import bits, pack, transpose


def add(a,b):
    return (((a//8+b//8)%4)*8+(a%8+b%8)%8)


TRANSLATE = [[add(i,j) for j in range(32)] for i in range(32)]


def ring_mul(a,b):
    out = 0
    for i in bits(a):
        for j in bits(b):
            out ^= 1 << TRANSLATE[i][j]
    return out


def mat_mul(a,b):
    return [ring_mul(a[2*i],b[j]) ^ ring_mul(a[2*i+1],b[2+j]) for i in range(2) for j in range(2)]


def mat_commutator(a,b):
    return sum((x^y) << (32*i) for i,(x,y) in enumerate(zip(mat_mul(a,b),mat_mul(b,a))))


def decode_mat(v):
    return [(v>>(32*i)) & ((1<<32)-1) for i in range(4)]


def lift_gl(a):
    return [sum(sum(1 << (32*c+TRANSLATE[g][t]) for t in bits(a[2*r+c])) for c in range(2))
            for r in range(2) for g in range(32)]


def two_block(a,b):
    size = len(a)
    hx = [x | (y<<size) for x,y in zip(a,b)]
    hz = [x | (y<<size) for x,y in zip(transpose(b,size),transpose(a,size))]
    return hx,hz


def gl_checks(f,g):
    return two_block(lift_gl(f),lift_gl(g))


def translation(dx,dy,sheets=4):
    return [32*s+((i+dx)%4)*8+(j+dy)%8 for s in range(sheets) for i in range(4) for j in range(8)]


def affine_fold(sigma, phi=(1,0,0,1), offsets=None):
    """phi maps (i,j) to (a*i+b*j mod4, c*i+d*j mod8)."""
    offsets = offsets or [(0,0)] * len(sigma)
    a,b,c,d = phi
    return [32*sigma[s]+((a*i+b*j+offsets[s][0])%4)*8+(c*i+d*j+offsets[s][1])%8
            for s in range(len(sigma)) for i in range(4) for j in range(8)]


class Extension:
    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha,beta
        self.states = list(itertools.product(range(4),range(8),range(2)))
        self.table = np.empty((64,64),dtype=np.int64)
        for a,(i,j,t) in enumerate(self.states):
            for b,(p,q,r) in enumerate(self.states):
                zz = (t+r+j*p+alpha*((i+p)//4)+beta*((j+q)//8))%2
                self.table[a,b] = (((i+p)%4)*8+(j+q)%8)*2+zz
        self.inverse = [next(b for b in range(64) if self.table[a,b] == 0) for a in range(64)]

    def mul(self,a,b):
        return int(self.table[a,b])

    def polynomial_mul(self,a,b):
        out = 0
        for i in bits(a):
            for j in bits(b):
                out ^= 1 << int(self.table[i,j])
        return out

    def commutator(self,a,b):
        return self.polynomial_mul(a,b) ^ self.polynomial_mul(b,a)

    def lift(self,a):
        # Column h -> sum_g gh, matching qLDPC's left regular representation.
        rows = [0]*64
        for g in bits(a):
            for h in range(64):
                rows[int(self.table[g,h])] ^= 1 << h
        return rows

    def checks(self,a,b):
        return two_block(self.lift(a),self.lift(b))

    def right(self,g):
        return [64*s+int(self.table[h,g]) for s in range(2) for h in range(64)]

    def aut(self, ax, ay):
        """Enumerate a proposed automorphism from generator images; verify it."""
        def power(g,n):
            out=0
            for _ in range(n): out=self.mul(out,g)
            return out
        z=self.mul(self.mul(ax,ay),self.inverse[self.mul(ay,ax)])
        p=[self.mul(self.mul(power(ax,i),power(ay,j)),power(z,t)) for i,j,t in self.states]
        if len(set(p)) != 64:
            return None
        if not np.array_equal(np.asarray(p)[self.table],self.table[np.ix_(p,p)]):
            return None
        return p

    def folds(self):
        seen=set()
        for ax in (16,self.inverse[16],17,self.inverse[17]):
            for ay in (2,self.inverse[2],3,self.inverse[3]):
                within=self.aut(ax,ay)
                if within is None: continue
                for swap in (False,True):
                    p=tuple(64*(s^swap)+within[h] for s in range(2) for h in range(64))
                    if p not in seen:
                        seen.add(p)
                        yield list(p)


def positive_control():
    a = sum(1 << (i*8+j) for i,j in ((0,0),(3,3),(3,6)))
    b = sum(1 << (i*8+j) for i,j in ((0,0),(2,4),(3,0)))
    def theta(g):
        i,j=divmod(g,8)
        return i*8+(-2*i-j)%8
    c=sum(1<<theta(g) for g in bits(a))
    d=sum(1<<theta(g) for g in bits(b))
    def lift(v):
        return [sum(1<<add(i,t) for t in bits(v)) for i in range(32)]
    aa,bb,cc,dd=map(lift,(a,b,c,d))
    at,bt,ct,dt=(transpose(v,32) for v in (aa,bb,cc,dd))
    hx=[aa[i] | bb[i]<<64 | ct[i]<<128 for i in range(32)]
    hx += [aa[i]<<32 | bb[i]<<96 | dt[i]<<128 for i in range(32)]
    hz=[cc[i] | dd[i]<<32 | at[i]<<128 for i in range(32)]
    hz += [cc[i]<<64 | dd[i]<<96 | bt[i]<<128 for i in range(32)]
    fold=[32*(0,2,1,3,4)[s]+theta(i) for s in range(5) for i in range(32)]
    return hx,hz,translation(1,0,5),translation(0,1,5),fold
