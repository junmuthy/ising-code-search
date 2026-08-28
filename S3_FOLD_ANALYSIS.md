# Fold and full logical-action audit for `[[384,194,6]]`

This report performs the candidate-specific follow-up on
`s3-l4-j1-w16-e1bb058d91825488` requested after the initial `S_3` search.

## 1. Structured alternative ZX folds

The enumeration covers the direct structured permutation family:

- all 24 permutations of the four protograph qubit blocks;
- all six permutations of the natural three-point `S_3` coordinate;
- all 128 automorphisms of `C_8 x C_4`;
- global bottom translations quotiented out.

Thus 18,432 representatives were checked, corresponding to 589,824
permutations when the 32 translations are restored. Every candidate was tested
in both directions:

```text
rowspan(H_X P) = rowspan(H_Z)
rowspan(H_Z P) = rowspan(H_X).
```

Exactly two representatives are ZX folds:

| Fold | Block permutation | Top permutation | Bottom action | Order | Pairing rank on the clean grid |
| --- | --- | --- | --- | ---: | ---: |
| original | `[1,0,3,2]` | `[0,1,2]` | `x -> x`, `y -> y` | 2 | 0 |
| alternative | `[2,3,0,1]` | `[0,1,2]` | `x -> x^-1`, `y -> y^-1` | 2 | 0 |

Composing either representative with a global bottom translation only
row/column-permutes the translation-circulant logical pairing and therefore
does not change its rank. No fold in this structured family implements a
logical Hadamard on the certified 32-grid.

## 2. Complete logical Clifford action

qLDPC supplies canonically paired logical bases with
`L_X L_Z^T = I_194`. For each physical fold, the saved analysis constructs the
full `388 x 388` binary symplectic matrix on logical Pauli coordinates.

Both folds have the same classification:

| Property | Value |
| --- | ---: |
| Logical symplectic dimension | 388 |
| Matrix rank | 388 |
| Fold order | 2 |
| Fixed-space dimension | 194 |
| `rank(F-I)` | 194 |
| Grid rank | 32 |
| Folded-grid rank | 32 |
| Grid/image intersection | 0 |
| Grid-orbit closure dimension | 64 |
| Restricted symplectic rank of that closure | 0 |

Define the fold form on logical Z classes by

```text
B(u,v) = < Z(u), F(Z(v)) >.
```

For both physical folds, `B` is:

- `194 x 194` and full rank;
- symmetric;
- alternating: `B(v,v)=0` for every logical Z class.

Alternation is invariant under every change of logical Z basis. More strongly,
the full symplectic quadratic invariant

```text
q(p) = < p, F(p) >
```

vanishes for every one of the 388 logical Pauli coordinates. This property is
invariant under arbitrary symplectic changes of logical basis, while an
independent logical Hadamard has nonzero `q`. Consequently the fold cannot be
converted into 194 independent logical Hadamards even by a non-CSS basis
change. Its exact canonical normal form consists of 97 hyperbolic pairs. The
generated basis was verified directly to obey

```text
F: Z(u_i) -> X(v_i)    F: X(u_i) -> Z(v_i)
F: Z(v_i) -> X(u_i)    F: X(v_i) -> Z(u_i)
```

for `i=0,...,96`. This is transversal Hadamard followed by a logical swap in
each pair.

## The certified grid inside the normal form

The hyperbolic completion keeps the original 32 translated Z classes fixed as
`u_0,...,u_31`. It constructs partner classes `v_0,...,v_31` such that

```text
B(u_i,u_j) = 0
B(v_i,v_j) = 0
B(u_i,v_j) = delta(i,j).
```

Therefore the grid does have an exact 32-qubit partner sector algebraically,
and the physical fold performs Hadamard plus SWAP between the two sectors.
However, the first exact completion obtained from the canonical qLDPC logical
basis is not STAR-compatible:

| Fold | Raw partner Z weight | Mean | Pairwise disjoint? |
| --- | --- | ---: | --- |
| original | 50–172 | 137.4375 | no |
| alternative | 46–160 | 134.1875 | no |

These are weights of one explicit representative per partner class, not
minimum-weight certificates; adding stabilizers could reduce them. They do
show that the algebraically natural completion is dense rather than another
clean translation grid.

The 64-dimensional orbit closure generated only by the grid and its folded
image is totally isotropic. Any nondegenerate fold-invariant subsystem
containing it consequently needs at least 128 Pauli dimensions, corresponding
to at least 64 logical qubits. The explicit grid-plus-partner construction
attains that algebraic lower bound.

## Artifacts

- [`analyze_s3_candidate_folds.py`](analyze_s3_candidate_folds.py) reruns the audit.
- [`gala_search/s3_fold_analysis.py`](gala_search/s3_fold_analysis.py) contains the algorithms.
- [`results/s3-compact-fold-analysis.json`](results/s3-compact-fold-analysis.json) contains the fold list and summaries.
- [`results/s3-compact-logical-action.npz`](results/s3-compact-logical-action.npz) contains both `388 x 388` symplectic matrices, the hyperbolic basis transforms, the grid coordinates, invariant closures, and explicit partner representatives.

## Conclusion

The alternative-fold search is negative for the clean grid, but the full
logical-action calculation is constructive: the fold is not a dense arbitrary
Clifford. It is exactly 97 Hadamard-plus-SWAP pairs, and the clean grid can be
anchored as one side of 32 of them. The remaining code-design problem is now
sharply defined: find low-weight, preferably translated and disjoint,
representatives of the 32 partner classes—or prove that no such partner module
exists.
