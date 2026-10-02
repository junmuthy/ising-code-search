# All-weight-eight simultaneous Ising basis for a `[[64,8,8]]` BB code

This package contains a complete, machine-checkable presentation of the
Liang--Chen self-dual `[[64,8,8]]` bivariate-bicycle code in one logical basis
that simultaneously provides:

1. a regular logical `C_4 x C_2` grid induced by physical code permutations;
2. transversal physical Hadamard acting as logical Hadamard plus a known grid
   translation;
3. exact-cover batches of internally disjoint logical supports in both the X
   and Z bases;
4. weight-eight representatives for every one of the eight logical qubits in
   both bases.

The check matrices define the published code; the logical basis and batching
are the derived Ising-oriented presentation supplied here.

## Quick verification

With Python and NumPy available, run

```bash
python verify_package.py
python reconstruct_code.py
```

The first command verifies the CSS ranks and commutator, self-duality, check
weights, logical kernels and independence, canonical ZX pairing, both exact
covers, physical code automorphisms, regular `C_4 x C_2` action, and the exact
Hadamard permutation.  The second reconstructs the checks from the group and
polynomial and compares them bit-for-bit with the archive.

## Code definition

Use the 32 monomials represented by exponent pairs `(a,b)` with
`0 <= a < 4`, `0 <= b < 8`, subject to

\[
x^4y^4=1,\qquad y^8=1.
\]

For

\[
f=1+x+y+y^{-1},\qquad g=f^\dagger,
\]

the displayed CSS matrices are

\[
H_X=H_Z=[F\mid F^T].
\]

They have shape `32 x 64`, rank `28`, and uniform row weight eight.  Hence the
code encodes eight logical qubits.  The exact distance `d=8` is the published
Liang--Chen result; the package contains explicit weight-eight logicals.

## Logical grid and translations

Label logical `i` by `i=2a+b`, with `(a,b) in C_4 x C_2`.  The physical
permutations stored in the archive induce

\[
T_x:(a,b)\mapsto(a+1,b),\qquad
T_y:(a,b)\mapsto(a,b+1).
\]

Thus

\[
T_x=(0\;2\;4\;6)(1\;3\;5\;7),
\]

\[
T_y=(0\;1)(2\;3)(4\;5)(6\;7).
\]

They commute and generate a regular, transitive `C_4 x C_2` action.

## Transversal Hadamard

The saved X and Z bases are canonical and obey

\[
H^{\otimes64}:Z_i\mapsto X_{p(i)},\qquad
H^{\otimes64}:X_i\mapsto Z_{p(i)},
\]

with

\[
p=(0\;5)(1\;4)(2\;7)(3\;6)=T_x^2T_y.
\]

These are exact support equalities, not merely equality of logical classes.

## Disjoint exact covers

The Z batches are

\[
\{0,1,7\}\sqcup\{4,6\}\sqcup\{2,3,5\},
\]

and the X batches are

\[
\{2,4,5\}\sqcup\{1,3\}\sqcup\{0,6,7\}.
\]

Each logical occurs exactly once in its basis's cover.  Supports are pairwise
disjoint within a batch.  Supports in different batches may overlap, so this
is a three-wave STAR presentation rather than simultaneous injection of all
eight logicals.

Every saved logical representative has weight eight.  Because the code has
distance eight, these representatives are individually minimum weight.

## Main artifact

`bb64_weight8_basis.npz` is the authoritative compact archive.  See
`DATA_DICTIONARY.md` for every array and `matrices_csv/` for text exports.

The physical translations permute logical classes exactly.  A translated
dressed support can differ from the particular saved representative by a
stabilizer.  This distinction matters when compiling representative-level
injection circuitry, but it does not alter the deterministic logical
automorphism or grid relabeling.

No syndrome-extraction schedule, circuit fault-distance certificate, or
circuit-level STAR simulation is claimed by this package.
