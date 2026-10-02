# `[[288,40,8]]` BB automorphism screen

## Outcome

The verified `[[288,40,8]]` code is a valid weight-eight BB code with a
standard ZX fold and native order-eight translation, but the complete
color-preserving Tanner automorphism group of its canonical presentation does
not contain the required regular logical `C_4 x C_8` module.

Consequently, this presentation does not provide the 32-site half-grid through
Tanner-graph automorphisms.  This is an exact negative result within that
automorphism class, not a no-go theorem for all stabilizer-preserving physical
permutations or all alternative presentations of the code.

## Correct generators and source discrepancy

The verified machine-readable record uses

\[
A=1+x^2+y^4+x^2y^4,
\qquad
B=1+x^4+y^8+x^4y^2
\]

over \(\mathbb F_2[C_{16}\times C_9]\).  Its MILP record certifies
\(d_X=d_Z=8\).

Table 2 of arXiv:2606.02418 displays different class-`e` generators that are
duplicated in the following class-`f` row.  Those displayed generators
reconstruct as `[[288,36,8]]`, not `[[288,40,8]]`.  The diagnostic is preserved
without modification in `results/run_001`; `results/run_002` uses the verified
record saved in `source_record.json`.

## Verified code properties

| Property | Result |
| --- | --- |
| Parameters | `[[288,40,8]]` |
| BB lattice | `C_16 x C_9` |
| Check weights | `w_X=w_Z=8` |
| CSS commutator | zero |
| ZX fold | exact: maps `H_X` to `H_Z` and conversely |
| Native `x` logical order | `16` |
| Native `x^2` logical order | `8` |
| Native `y` logical order | `3` |
| Tanner components | two identical components |
| Component parameters | `[[144,20,8]]` each |

The two-component statement follows from the Tanner decomposition: each
component contains 144 data qubits, 72 X checks, and 72 Z checks; each X and Z
submatrix has GF(2) rank 62.  Since an automorphism exchanges the two
components and the full distance is eight, the decomposition is
`[[144,20,8]]` direct-sum `[[144,20,8]]`.

## Exhaustive automorphism result

The exact color-preserving canonical Tanner automorphism group has order

\[
|\operatorname{Aut}_{\rm Tanner}|=10,368.
\]

All `10,368` elements were enumerated.  On the 40-dimensional logical Z space
the search found:

- 324 elements of logical order four;
- 864 elements of logical order eight;
- 279,936 order-four/order-eight pairs considered;
- 69,984 logically commuting pairs;
- 54,432 pairs that also commute as physical permutations.

For every logically commuting pair \((X,Y)\), the exact regular-module socle
test vanishes:

\[
(X+I)^3(Y+I)^7=0.
\]

Thus none contains a cyclic 32-dimensional regular
\(\mathbb F_2[C_4\times C_8]\) submodule.  Random logical representatives are
not involved in this negative result.

## Files

- `analyze.py`: reproducible reconstruction and exhaustive screen.
- `source_record.json`: exact upstream verified record and commit.
- `results/run_001/`: paper-table mismatch diagnostic (`[[288,36,8]]`).
- `results/run_002/summary.json`: corrected exhaustive result.
- `results/run_002/checks-native-and-logicals.npz`: checks, native translations,
  ZX fold, and logical bases.
