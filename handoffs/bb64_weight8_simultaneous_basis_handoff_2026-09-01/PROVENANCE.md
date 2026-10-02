# Provenance and claim boundary

## Published code

The check construction and exact `[[64,8,8]]` parameters are attributed to
Zijian Liang and Yu-An Chen, “Self-dual bivariate bicycle codes with
transversal Clifford gates,” arXiv:2510.05211v2.  See `CITATION.bib`.

The code is reconstructed here from

\[
f=1+x+y+y^{-1},\qquad g=f^\dagger,
\]

on the 32-element abelian twisted torus with `x^4 y^4=1` and `y^8=1`.

## Derived simultaneous basis

The all-weight-eight logical basis, regular `C_4 x C_2` labeling, physical
translation generators, Hadamard permutation, and `3+2+3` disjoint exact covers
were derived in the local GALA/Ising code-search project on 2026-09-01.

The final low-weight query required the logical classes `\{2,3,5\}` to have
the exact weight pattern `(8,8,8)` and mutually disjoint supports.  Z3 returned
`sat` in `0.022690 s`.  The original solver records are included under
`provenance/`.

`verify_package.py` independently checks the matrices and all derived
algebraic claims using only NumPy.  It treats `d=8` as the published exact
distance; it does not reproduce the paper's lower-distance proof.

## Not claimed

This package does not claim that three batches are minimal, that the saved
dressed representatives are literal physical translates of one another, or
that a particular syndrome circuit achieves a specified circuit fault
distance.  It also contains no circuit-level STAR-performance simulation.
