# Self-dual `[[64,8,8]]` BB code — Ising handoff

This archive is a self-contained matrix-level handoff for the self-dual
bivariate-bicycle code of Liang and Chen, arXiv:2510.05211v2.  It includes the
stabilizers, two complementary logical presentations, the physical and logical
translation actions, four disjoint STAR batches, and scripts that reconstruct
and verify the data using only NumPy.

## Code definition

The code is defined on the twisted 32-element torus

\[
x^4y^4=1,\qquad y^8=1
\]

by

\[
f=1+x+y+y^{-1},\qquad g=f^\dagger.
\]

In the saved presentation,

\[
H_X=H_Z=[F\mid F^T].
\]

There are 32 displayed checks of each CSS type, 28 independent checks of each
type, and every displayed check has weight eight.  Thus

\[
n=64,\qquad k=64-28-28=8.
\]

The published exact distance is eight.  Every raw grid logical included here
has weight eight, providing an explicit upper-distance witness.

## The two logical presentations

The distinction between these presentations is important.

### 1. Raw translation-grid presentation

`raw_grid_z` is an ordered regular `C4 x C2` orbit:

\[
(1,T_2,T_4,T_4T_2,T_4^2,T_4^2T_2,T_4^3,T_4^3T_2).
\]

All eight representatives have weight eight.  The four batches

\[
\{0,1\},\quad\{2,3\},\quad\{4,5\},\quad\{6,7\}
\]

are internally pairwise disjoint, so each batch supports two simultaneous
STAR injections.  `raw_dual_x` is the canonical X basis dual to these Z
operators.

The same-support X representatives have a full-rank alternating pairing with
the raw Z grid, saved as `raw_zx_pairing`; it is not a permutation matrix.

### 2. Hadamard-plus-permutation presentation

`hperm_z` and `hperm_x` are canonical dual logical bases.  They obey

\[
hperm_z\,hperm_x^T=I_8.
\]

Transversal physical Hadamard acts exactly as logical Hadamard followed by

\[
\pi=(0\ 1)(2\ 3)(4\ 5)(6\ 7):
\qquad
Z_i\mapsto X_{\pi(i)},\quad X_i\mapsto Z_{\pi(i)}.
\]

No stabilizer equivalence is needed for this statement: the corresponding
binary support vectors are exactly equal.  The GF(2) transformation between
the raw grid and this basis is saved as `hperm_change_from_raw_z`.

The Hadamard basis is not the raw-disjoint presentation.  Therefore the
archive establishes both desired structures on the same logical space and
provides their exact relationship; it does not claim that the transformed
Hadamard basis retains the four raw-disjoint support batches.  That simultaneous
basis optimization remains a separate question.

## Quick verification

Python 3 and NumPy are sufficient:

```bash
python verify_package.py
python reconstruct_code.py
```

The first script checks the CSS ranks and commutator, check weights, logical
independence and pairing, disjoint batches, the regular `C4 x C2` action, and
the Hadamard-plus-SWAP action.  The second reconstructs `H_X=H_Z` directly from
the polynomial and twisted-torus relations and compares it with the archive.

## Physical indexing

- Qubits `0..31`: first BB half.
- Qubits `32..63`: second BB half.
- Within either half, `(x,y)` has index `8*x+y`, with `0<=x<4`, `0<=y<8`.
- Crossing `x=4` adds four to the y coordinate, implementing `x^4=y^4`.

Permutation arrays use the convention `output[permutation[i]] = input[i]`.

## Citation

Zijian Liang and Yu-An Chen, “Self-dual bivariate bicycle codes with
transversal Clifford gates,” arXiv:2510.05211v2 (2026), especially Table 1,
Figure 1, and Appendix A.  See `CITATION.bib`.
