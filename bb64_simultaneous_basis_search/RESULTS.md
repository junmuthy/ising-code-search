# Simultaneous Ising basis for the `[[64,8,8]]` BB code

## Two-batch refinement

The focused follow-up in `fewer_batch_search` improves the operational cover
from `3+2+3` to `4+4` while preserving the exact X/Z representative map.  The
preferred artifact is
`fewer_batch_search/results/run_001_weight8_60s/basis_batches2_candidate0.npz`.

Its exact Z cover is

\[
\{0,3,4,7\}\;\sqcup\;\{1,2,5,6\},
\]

and its exact X cover is

\[
\{1,2,5,6\}\;\sqcup\;\{0,3,4,7\}.
\]

Every X and Z representative has weight eight.  The check matrices, distance,
regular `C_4 x C_2` action, and Hadamard permutation
`(0 5)(1 4)(2 7)(3 6)` are unchanged.  All eight one-batch tests were exactly
`unsat` at logical weight eight; two of the eight candidates gave two-batch
witnesses, with no timeouts.  See the new run's `RESULTS.md` for the exact
search boundary and files.

## Weight-eight refinement (three-batch predecessor)

`run_003_low_weight` improves the original positive result below.  The same
basis properties and the same `3+2+3` exact cover are retained, but the one
weight-sixteen Z representative has been replaced by a weight-eight
representative that is disjoint from its two batch partners.  Consequently all
eight Z representatives and all eight canonical X representatives now have
weight eight.  The preferred artifact is
`results/run_003_low_weight/simultaneous_basis_weight64.npz`; see that run's
`RESULTS.md` and `weight64_certificate.json` for the new certificate.

Since the code distance is eight, every representative in this refined basis
is individually minimum weight.  The remainder of this document records the
earlier `run_002` weight-72 construction and its search boundary for
provenance.

`run_002` found and independently certified a single logical presentation that
simultaneously has the three requested properties:

1. a regular logical `C_4 x C_2` action induced by physical code
   permutations;
2. transversal physical Hadamard equal to logical Hadamard followed only by a
   known logical permutation;
3. internally disjoint logical-support batches that cover all eight logicals
   exactly once.

This is a new presentation of the same Liang--Chen self-dual `[[64,8,8]]`
bivariate-bicycle code.  The check matrices and code distance did not change.

## Certified properties

| Property | Certified result |
|---|---|
| Code parameters | `[[64,8,8]]` |
| CSS checks | `H_X = H_Z` |
| Independent checks of each type | `28` |
| Check weight | `8` for every displayed row |
| Logical grid | regular, transitive `C_4 x C_2` |
| Logical `x` translation | `[0 2 4 6][1 3 5 7]` |
| Logical `y` translation | `[0 1][2 3][4 5][6 7]` |
| Hadamard permutation | `[0 5][1 4][2 7][3 6]` |
| Z-batch sizes | `3+2+3` |
| X-batch sizes | `3+2+3` |
| Logical Z weights | `[8,8,8,8,8,16,8,8]` |
| Logical X weights | `[16,8,8,8,8,8,8,8]` |

With the index convention `i=2a+b`, the eight logicals have coordinates
`(a,b) in C_4 x C_2`.  The two physical permutation generators induce

\[
(a,b)\mapsto(a+1,b),\qquad (a,b)\mapsto(a,b+1),
\]

respectively.  The actions commute and their orbit on any logical contains all
eight grid sites.

The exact Z-support cover is

\[
\{0,1,7\}\;\sqcup\;\{4,6\}\;\sqcup\;\{2,3,5\}.
\]

Each set is internally pairwise disjoint, and every logical index occurs once.
Because the canonical X basis is the Hadamard-permuted copy of these same
supports, its exact cover is

\[
\{2,4,5\}\;\sqcup\;\{1,3\}\;\sqcup\;\{0,6,7\}.
\]

Thus both bases can be processed in three parallel STAR batches.  There is no
duplication or omission of a logical in either cover.

Transversal Hadamard acts as

\[
H^{\otimes 64}:\quad
Z_i\mapsto X_{p(i)},\qquad
X_i\mapsto Z_{p(i)},
\]

where

\[
p=(0\;5)(1\;4)(2\;7)(3\;6).
\]

This action is exact for the saved representatives, not merely a full-rank ZX
pairing.  The logical permutation is known and reversible, so an Ising control
layer can relabel subsequent rotations and bonds by `p`.

## Files

- `results/run_002/simultaneous_basis_weight72.npz` is the compact matrix
  artifact to use.  It contains `H_X`, `H_Z`, canonical logical X/Z bases, both
  physical grid permutations, all induced logical action matrices, the
  Hadamard permutation, and the exact-cover batch labels.
- `results/run_002/weight72_certificate.json` is the machine-readable summary
  and verification record.
- `results/run_002/success.json` records the original feasibility witness.
- `results/run_002/summary.json` records the exhaustive affine class scan and
  solver outcomes.
- `search.py` reproduces the class and arbitrary-dressing search.
- `certify_weight72.py` rebuilds and independently checks the refined artifact.

## Search boundary and caveats

The class-level scan is exhaustive over all affine automorphisms of the
published twisted torus that preserve this code, all `128` commuting regular
action pairs, and all `255` nonzero seeds per pair.  These reduce to eight
distinct unordered logical-basis class sets.  It is not an enumeration of
non-affine physical automorphisms outside this family.

The one- and two-batch dressing probes each timed out at 20 seconds for the
first class set, so this run proves a three-batch positive result but does not
prove three batches minimal.  Similarly, the refined total Z weight is `72`;
the unresolved total-weight caps `24` and `28` for its final three-logical batch
mean global support-weight optimality is not claimed.

Finally, the physical translations permute the saved logical *classes* exactly.
A translated dressed representative can differ from the corresponding saved
representative by a stabilizer.  This is sufficient for a deterministic
logical automorphism and classical Ising relabeling, but the distinction should
be retained when compiling a representative-level injection circuit.
