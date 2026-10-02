# Weight-12 nonabelian four-grid GALA search

## Result

The search found a genuinely active, nonabelian polynomial GALA code over

\[
S_3\times C_8\times C_4
\]

with certified parameters

\[
\boxed{[[1152,580,6]]}.
\]

It contains four selected, independent logical sectors, each carrying a
translated `C_8 x C_4` grid of 32 logical qubits.  All 128 selected logical
`Z` supports are pairwise disjoint.  A physical ZX fold acts on these 128
logicals as independent Hadamards followed by a fixed, translation-compatible
permutation between grid pairs.

This is a positive answer to the weight question: adding `S_3` does **not**
force stabilizer weight 16.  This candidate has uniform stabilizer weight 12.

## Construction

The code uses the `L=12`, `J=3` GALA protograph and the natural degree-three
binary representation of `S_3`.  Every one of the six `F` entries is a single
group monomial:

\[
\begin{aligned}
F_0 &= 1,\\
F_1 &= \sigma^{-1}x^7y^3,\\
F_2 &= \tau_0x^7,\\
F_3 &= x^4y^3,\\
F_4 &= x^7y,\\
F_5 &= \sigma^{-1}x^7.
\end{aligned}
\]

The other half is constrained by

\[
G_i=F_i^\dagger.
\]

The candidate ID is
`s3-l12-j3-w12-vertex-fold-5390517e81fd1dd8`.

The physical size is

\[
12\times 3\times 8\times 4=1152.
\]

There are 288 presented checks of each CSS type, each check matrix has rank
286, and therefore

\[
k=1152-286-286=580.
\]

The active row differences `0`, `+/-1`, and `+/-2` cancel exactly, while the
unused offset `3` remains nonzero.  Thus this is genuine active orthogonality,
not a BB code disguised by notation.

## Ising logical grids

Each seed below has weight six.  Translating it by `C_8 x C_4` produces 32
pairwise-disjoint logical supports forming one grid.

| Grid | Weight-six seed support | Internal translation fibres |
| --- | --- | --- |
| `0` | `[32,452,485,764,829,915]` | `[1,14,15,23,25,28]` |
| `1` | `[64,396,560,703,776,888]` | `[2,12,17,21,24,27]` |
| `2` | `[96,215,295,639,979,1103]` | `[3,6,9,19,30,34]` |
| `3` | `[128,259,365,668,1016,1083]` | `[4,8,11,20,31,33]` |

The resulting 128 supports have:

- combined logical rank 128 modulo the stabilizers;
- no physical overlap between any two supports;
- support union size `128 x 6 = 768`, leaving 384 physical qubits outside the
  selected logical supports;
- exact `C_8` and `C_4` translation actions inherited from the physical
  abelian factors.

The unused physical qubits and the additional 452 encoded logical degrees of
freedom are not claimed to form additional STAR-compatible grids.

## ZX duality

This code is ZX-dual through a physical block reflection; it does not satisfy
`H_X = H_Z` without that permutation.  After the reflection, the `X` and `Z`
check spaces agree.  On the selected 128-dimensional sector, the ZX pairing
matrix has rank 128 and has exactly one `1` in every row and column.

Writing coordinates as `(a,b)` in `C_8 x C_4`, the row-to-column pairing is:

| Logical `Z` grid | Paired folded-`X` grid | Coordinate offset |
| --- | --- | --- |
| `0` | `2` | `(-1,0)` |
| `2` | `0` | `(+1,0)` |
| `1` | `3` | `(-1,+1)` |
| `3` | `1` | `(+1,-1)` |

Consequently the transversal physical Hadamard plus the fixed fold is a
logical Hadamard followed by a known involutive permutation: it swaps grids
`0 <-> 2` and `1 <-> 3` with the listed cyclic offsets.  The offsets can be
software-tracked or absorbed into the initial logical-coordinate convention.
For four identical simulations the grid swaps are benign; a protocol that
requires each grid label to remain fixed would need an additional logical
permutation or a stricter search.

## Exact distance certificate

The saved certificate establishes:

- no nontrivial logical of weight at most four by syndrome meet-in-the-middle;
- exhaustive Tanner search at weight five, with no zero-syndrome support;
- an exhaustive weight-six search returning the nontrivial logical support
  `[0,161,352,387,796,1071]`;
- equality of `X` and `Z` distances from the physical ZX fold.

Therefore the exact CSS distance is six.

## Stabilizers and schedule

Every `X` and `Z` check generator has weight 12.  Every physical data qubit
participates in three checks of each CSS type, hence six interactions in a
complete `X+Z` extraction round.  The combined check/data incidence graph has
maximum degree 12.  Bipartite edge coloring therefore gives a 12-layer CNOT
schedule, and 12 layers is optimal under the model of one ancilla per check,
one two-qubit gate per qubit per layer, and unrestricted routing.

There is one important STAR caveat.  Column degree three is odd, so the
even-syndrome-parity property of the bicycle-chain code is absent.  The
one-round TMR masking argument does not transfer directly; repeated syndrome
measurement or a separately validated measurement protocol is required.

## Search observations

The `J/L <= 1/4` constraint is important.  Denser protographs such as
`L=10,J=4` or `L=12,J=4` use every row difference and leave no latent offset
for active orthogonality.

At weight 12 with `L=12,J=3`:

- exact self-duality with uniform data degree three puts the all-ones vector
  into the stabilizer and prevents the desired odd, same-support logicals;
- all five inequivalent even-degree layouts were tested; among 1,225 lifted
  algebraic survivors, all 476 candidates reaching the low-weight screen had
  a logical of weight at most four;
- allowing a nontrivial ZX fold gave 67 structurally viable edge-reflection
  codes and 90 structurally viable vertex-reflection codes with no logical of
  weight at most four;
- exhaustive weight-six grid enumeration found 435 and 808 translated grid
  orbits in those two branches, respectively;
- no single grid had full self-pairing rank, but combining four disjoint grids
  produced two full-rank edge-reflection codes and seven full-rank
  vertex-reflection codes.

The last point is the key search lesson: individual grids can have singular
ZX self-pairing while their off-diagonal pairing with other grids makes the
combined 128-logical sector nondegenerate.

## Reproduction

Recompute the compact final certificate with:

```bash
.venv/bin/python -m searches.s3.certify_s3_many_copy
```

The machine-readable result is
[`results/s3-many-copy/recommended-certificate.json`](../../results/s3-many-copy/recommended-certificate.json).

The vertex-reflection search pipeline is:

```bash
.venv/bin/python -m searches.s3.run_s3_many_copy_fold_search \
  --vertex-reflection --algebraic-target 300 \
  --output results/s3-many-copy/l12-j3-w12-vertex-fold.jsonl

.venv/bin/python -m searches.s3.screen_s3_many_copy_fold \
  --vertex-reflection \
  --input results/s3-many-copy/l12-j3-w12-vertex-fold.jsonl \
  --output results/s3-many-copy/l12-j3-w12-vertex-fold-distance-grid.jsonl

.venv/bin/python -m searches.s3.enumerate_s3_many_copy_fold_grids \
  --vertex-reflection \
  --input results/s3-many-copy/l12-j3-w12-vertex-fold-distance-grid.jsonl \
  --output results/s3-many-copy/l12-j3-w12-vertex-fold-grid-orbits.jsonl

.venv/bin/python -m searches.s3.combine_s3_many_copy_fold_grids \
  --vertex-reflection \
  --input results/s3-many-copy/l12-j3-w12-vertex-fold-grid-orbits.jsonl \
  --output results/s3-many-copy/l12-j3-w12-vertex-fold-four-grid-combinations.jsonl
```

The final regression tests are in
[`tests/test_s3_many_copy.py`](../../tests/test_s3_many_copy.py).

## Comparison with the abelian four-grid code

| Code | Useful selected logicals | Physical qubits | Useful rate | Distance | Check weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abelian four-grid code | `128` | `896` | `1/7` | `7` | `12` |
| This `S_3` code | `128` | `1152` | `1/9` | `6` | `12` |

The nonabelian result demonstrates that active GALA packing with check weight
below 16 is possible, but this first hit does not beat the abelian code in
physical efficiency or distance.  It should be treated as a new viable search
anchor, not yet as the preferred architecture.
