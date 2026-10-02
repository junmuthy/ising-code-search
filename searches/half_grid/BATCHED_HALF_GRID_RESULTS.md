# Batched `n=128` half-grid geometry search

## Target

This search tests the smallest proposed lift

```text
GL(2,2) x C8 x C4
```

with the minimal two-block GALA layout.  It has `n=128`, four physical
`C8 x C4` translation sheets, and targets one 32-logical-qubit checkerboard.
The 32 STAR rotations are divided into the two 16-site batches
`a+b = 0 mod 2` and `a+b = 1 mod 2`.  Logical supports may overlap between
the batches but must be disjoint inside either batch.  The target distance is
at least seven and the strict GALA ZX map may include a physical permutation.

## Exact geometry result

The generator search is pruned before polynomial synthesis.  This is an exact
no-go for this particular `n=128`, two-batch, faithful-two-dimensional ansatz.

On one physical translation sheet, disjointness among all even translations
allows at most one even and one odd seed point.  The same is true on every
sheet.  Consequently a two-batch-disjoint seed has weight at most eight.  If
the distance is at least seven, only these occupancy patterns remain:

```text
weight 7: a permutation of (2,2,2,1)
weight 8: (2,2,2,2)
```

Every strict contragredient fold of the faithful `GL(2,2)` representation
swaps the two top coordinates, so its permutation of the four physical
translation sheets has no fixed point.  Let `n_s` be the seed occupancy of
sheet `s` and let `p` be that sheet permutation.  The parity of every row of
the translation-circulant logical ZX pairing is

```text
sum_s n_s n_{p(s)} mod 2.
```

It is zero for every allowed weight-seven or weight-eight occupancy pattern
and for all four structured folds, including the two order-four coupled
folds.  Therefore the all-ones logical vector lies in the kernel of the
`32 x 32` pairing matrix.  Its rank is strictly less than 32, so transversal
ZX cannot act invertibly on the proposed logical checkerboard.

The obstruction is independent of the polynomial generators and of the
locations of the seed points inside their parity classes.  Searching
weight-at-most-12 stabilizers cannot repair it, so no generator enumeration
is run for this branch.

The saved run exhausts all allowed sheet-occupancy patterns and separately
samples 40,000 concrete supports (5,000 per fold at each of weights seven and
eight).  It finds zero rank-32 pairings; the largest sampled rank is 28.  All
40,000 supports have the predicted all-ones pairing-kernel vector.

## Reproduction

```bash
.venv/bin/python -m searches.half_grid.run_batched_half_grid_search \
  --run-name gl2-n128-two-batch-seed260826 \
  --samples-per-fold 5000
```

The exact certificate and the independent sampled check are written under
[`results/half-grid-batched/gl2-n128-two-batch-seed260826/geometry-certificate.json`](../../results/half-grid-batched/gl2-n128-two-batch-seed260826/geometry-certificate.json).
Focused tests run with:

```bash
.venv/bin/python -m pytest tests/test_batched_half_grid.py -q
```

## Consequence

Batching removes the elementary `n >= 32 d` all-at-once disjointness bound,
but it does not remove the faithful-representation ZX parity obstruction at
`n=128`.  The next genuinely different compact branch is the degree-three
permutation representation at `n=192`, whose transposition fold has a fixed
top coordinate.  That fixed sheet contributes an unpaired `n_s^2` term and
can make the logical pairing augmentation odd.
