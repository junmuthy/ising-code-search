# `GL(2,2) × C4 × C4`, `L=4,J=2`, `n=128` preflight

## Purpose

This is a deliberately generator-free preflight for a single logical
`C4 × C4` Ising grid.  It repeats the gates that exposed the failure of the
earlier `GL(2,2) × C4`, `[[32,4,6]]` search before spending time on sparse
polynomial enumeration or distance certification.

The represented binary top module has dimension two.  With four physical
protograph blocks and a 16-element bottom translation group,

\[
n=4\cdot 2\cdot 16=128.
\]

The target is one 16-logical-qubit grid, so a ZX-folded CSS code with equal
check ranks needs

\[
\operatorname{rank}H_X=\operatorname{rank}H_Z
=\frac{128-16}{2}=56.
\]

## What failed at `n=32`

The earlier one-row search did not fail at its first geometry test.  It found
100 weight-six seeds whose four translates were disjoint and whose logical ZX
pairing had rank four.  The failure occurred when stabilizers were imposed:

- generator relations guaranteeing CSS and an exact or row-space ZX fold
  usually made the complete constrained generator space incapable of reaching
  the required check rank;
- a more flexible solved-`G` scan produced many target-rank, weight-at-most-12
  candidates, but none retained a compatible logical ZX fold;
- after allowing the total encoded dimension to float, every one of 714
  self-dual structural candidates had exact distance two.

Thus a full-rank seed pairing alone is not a useful green light.

## Preflight gates

The new script checks the following in order:

1. A weight-six seed occupies six of the eight physical translation sheets.
   Its 16 `C4 × C4` translates must be pairwise disjoint and independent.
2. A physical ZX operation is Hadamard followed by a
   translation-commuting permutation.  Its induced `16 × 16` pairing on the
   grid must have rank 16.
3. The linear space defined by the seed-kernel equations and an exact forward
   ZX fold must have binary check-row capacity at least 56.  This is an exact
   upper-bound screen over the entire polynomial space, before imposing a
   weight limit.
4. On the spaces surviving gate 3, test the simple automatic-CSS relations

   \[
   G=qF,\qquad G=q\,\operatorname{swap}(F),
   \]

   for every monomial (q\in C_4\times C_4).

## Saved run

The run `gl2-c4xc4-l4-j2-preflight-260828-v1` produced:

| Gate | Result |
|---|---:|
| Concrete seed/fold pairs tested | 469 |
| Full-rank disjoint logical-grid witnesses | 100 |
| Maximum rank using ordinary top-independent folds | 8 |
| Independent exact-fold spaces tested | 80 |
| Independent spaces with check-rank capacity at least 56 | 12 |
| Automatic-CSS monomial spaces tested | 384 |
| Automatic-CSS spaces with check-rank capacity at least 56 | 0 |

The independent-space capacity distribution was:

| Check-rank capacity | Number of spaces |
|---:|---:|
| 16 | 48 |
| 28 | 4 |
| 32 | 4 |
| 48 | 12 |
| 56 | 10 |
| 64 | 2 |

Only the two coupled check-coordinate folds produced survivors: each yielded
six passing spaces among the 20 witnesses.  The ordinary folds yielded none.

After imposing the simple automatic-CSS relations, the capacities collapsed
to 14 or 16 in all 384 cases, far below 56.

## Decision

The `n=128` branch is **not ruled out**: disjoint grid geometry, nonsingular
logical ZX pairing, and sufficient unconstrained exact-fold check-rank
capacity coexist.  This is better than a pure geometry witness and justifies
retaining 12 specific seed/fold/check-fold spaces.

However, the old automatic-CSS ansatz should not be used for generator
enumeration.  It reproduces the central warning from the `n=32` search by
destroying stabilizer-rank capacity before sparsity or distance is considered.

The next generator stage should proceed only if it introduces a broader CSS
parameterization on the retained 12 spaces.  Before a large scan, it should
require a concrete generator to pass, in this order:

1. exact CSS commutation and a forward/reverse row-space ZX fold;
2. check ranks at least 56, with exact (k=16) preferred but not assumed;
3. nontriviality and rank 16 of the protected grid modulo stabilizers;
4. maximum stabilizer weight at most 12;
5. immediate rejection on any logical of weight below six.

The exact-fold screen is a sufficient structured realization of GALA ZX
duality, not a no-go theorem for every possible row-space fold or arbitrary
physical permutation.  Similarly, the automatic-CSS result currently covers
monomial (q), not every nonzero polynomial in
\(\mathbb F_2[C_4\times C_4]\).

## Files

- `preflight.py`: checkpointed, non-overwriting preflight runner.
- `test_preflight.py`: focused tests of the orbit geometry, exact-fold
  equations, and automatic-CSS commutation.
- `results/gl2-c4xc4-l4-j2-preflight-260828-v1/summary.json`: run summary.
- `results/gl2-c4xc4-l4-j2-preflight-260828-v1/geometry-witnesses.jsonl`:
  the 100 full-rank geometry witnesses.
- `results/gl2-c4xc4-l4-j2-preflight-260828-v1/capacity.jsonl`: incremental
  capacity records.
