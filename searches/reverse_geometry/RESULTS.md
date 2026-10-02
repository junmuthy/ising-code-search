# Natural-`S3` `n=192` reverse-geometry results

## Scope

This search uses the natural degree-three permutation representation of

```text
S3 x C8 x C4
```

in the minimal two-block layout, giving `n=192` and six physical translation
sheets.  It targets one 32-logical-qubit `C8 x C4` grid, divided into the two
16-site checkerboard injection batches.  Stabilizer weight is limited to 12.

The workflow is deliberately reversed: logical support and ZX pairing are
solved before polynomial generators.

## Logical geometry

The first cancellation template uses one singleton on a top sheet fixed by an
`S3` transposition.  Equal supports on exchanged sheet pairs cancel twice over
`GF(2)`.  The run found 100 weight-seven supports with:

- exact `C8 x C4` translation orbits;
- disjointness inside both checkerboard batches;
- rank 32 as physical vectors;
- identity `32 x 32` logical ZX pairing.

Five fold-tied annihilator spaces had nullity 105 or 106 and rank capacity
94--96, exceeding the rank 80 needed for `k=32`.

The corresponding artifacts are under
`results/reverse-geometry/natural-s3-w7-seed260826/`.

## First generator survivors and the invariant-triplet failure

An exact-six-term, no-zero-column pilot found three CSS structural survivors:

```text
two [[192,72]] candidates, check weights 8/12
one [[192,64]] candidate, check weights 8/10
```

All retained the 32 target logicals, the exact displayed ZX fold, even
syndrome parity, and nonzero degree on every qubit.  The complete
weight-one-through-four screen nevertheless found a weight-three logical in
both Pauli sectors of every code, supported on top coordinates

```text
[0,32,64].
```

This is the invariant `(1,1,1)` vector of the natural permutation module.
The original cancellation geometry had nonzero top-parity projection in only
one physical half, forcing its polynomial annihilator to have zero top
augmentation.  That makes the localized invariant triplet a kernel vector.

Artifacts:

- `results/reverse-geometry/natural-s3-w12-fold-tied-exact6-nozero-pilot/`
- `results/reverse-geometry/natural-s3-w12-fold-tied-exact6-certification/`

## Corrected projected geometry

The geometry search was broadened to arbitrary two-batch supports with
nonzero top-parity projection in both physical halves.  It found 100
permutation-paired witnesses after 22,486 trials:

```text
weight 7: 62
weight 8: 27
weight 9: 5
weight 10: 6
```

All first ten annihilator spaces remain rank-capable.  Their nullities are
96--103 and their rank-capacity upper bounds are 94--96.  Thus correcting the
logical geometry does not collapse the available check space.

Artifacts:

- `results/reverse-geometry/natural-s3-projected-geometry-seed260828/`
- `results/reverse-geometry/natural-s3-projected-annihilators/`

## Weight-at-most-12 generator result

Repeated sparse MILPs with explicit nonzero augmentation timed out without
incumbents, so meet-in-the-middle synthesis was used instead.  Each random
partition exhaustively matches the relevant `2+2`, `2+3`, and `3+3`
syndromes for generator weights four, five, and six.

For five corrected geometries with ten partitions each:

- every exact weight-four or weight-six annihilator match had zero top
  augmentation;
- no weight-five match was found;
- no candidate passed the augmentation and nonzero-column filters.

The physical data fold was then held fixed while all six natural-`S3`
permutations were allowed on check coordinates.  Across three geometries,
six check folds, and five partitions per space, the search found:

```text
1,327 weight-six syndrome matches
162 weight-four syndrome matches
0 weight-five syndrome matches
0 nonzero-augmentation retained generators
0 structural candidates
```

This is a bounded negative rather than a mathematical no-go.  Random
partitioning has high coverage for any individual four-to-six-term support,
but it does not exhaust every sparse support globally.  It does show that the
zero-augmentation failure is systematic across the tested geometries and all
displayed check-coordinate folds, rather than an accident of one seed.

Artifacts:

- `results/reverse-geometry/natural-s3-projected-sparse-mitm/`
- `results/reverse-geometry/natural-s3-projected-check-folds/`

## General check-row-space ZX search

The displayed-row condition was then removed.  For independent sparse
polynomials `F` and `G`, the checks are

```text
HX = [F,G]
HZ = [G^T,F^T].
```

The search requires only that the physical fold exchange `row(HX)` and
`row(HZ)`.  For a fixed `F`, the target logical kernels and CSS equation
`FG+GF=0` are affine in `G`.

An unguided pilot tried 50 nonzero-augmentation `F` supports on the first two
projected geometries.  All 50 affine systems were exactly inconsistent.
Eliminating `G` from the two logical-kernel equations exposed the relevant
linear compatibility code for `F`.

The exact augmentation-rank screen over all 100 projected witnesses found:

```text
67 geometries: every compatible F has zero top augmentation
33 geometries: nonzero-augmentation compatible directions remain
```

A bounded minimum-weight MILP found an augmented compatible `F` for 23 of
the latter geometries within three seconds; ten had no incumbent.  The 23
incumbents had formal weights:

```text
weight 2:  2
weight 4:  9
weight 6:  1
weight 8:  9
weight 10: 2
```

The twelve geometries with an incumbent of weight at most six were then
searched with multiple representatives.  This produced 204 distinct
compatible augmented `F` supports:

```text
weight 2:  40
weight 4: 156
weight 6:   8
```

Under the total formal-term budget of 12, none admitted a `G` completion.
Twenty-four failed the affine equations before cardinality optimization; the
other 180 had no incumbent within the two-second MILP bound.  A ten-second
follow-up on 20 hard weight-four supports proved two infeasible and left 18
as timeouts, again with no completion.

Repeating those same 20 supports with a 20-second limit proved seven
infeasible and left 13 as timeouts.  Thus the doubled allowance resolved five
additional cases negatively and still produced no `G` completion.

Consequently no candidate reached the check-row-space ZX test, much less the
rank, connectivity, or distance screens.  This is a bounded computational
negative: the timeout cases are unresolved, and the support enumerator does
not exhaust every compatible `F` of weight at most six.

Artifacts:

- `results/reverse-geometry/natural-s3-projected-rowspace-pilot/`
- `results/reverse-geometry/natural-s3-projected-rowspace-guided/`
- `results/reverse-geometry/natural-s3-projected-rowspace-minimum-all100/`
- `results/reverse-geometry/natural-s3-projected-rowspace-guided-low-f/`
- `results/reverse-geometry/natural-s3-projected-rowspace-guided-hard-extended/`
- `results/reverse-geometry/natural-s3-projected-rowspace-guided-hard-20s/`

## Current boundary

The reverse-order idea succeeds at its first two tasks:

1. ideal two-batch logical geometry exists abundantly at `n=192`;
2. its complete linear annihilator has ample stabilizer-rank capacity.

The row-space relaxation shows that the displayed fold was not the only
bottleneck.  In the tested low-weight sector, the projected logical geometry
and CSS orthogonality already prevent a sparse independent `G` completion.
The remaining untested region consists of hard MILP timeouts, compatible
`F` supports missed by randomized partitioning, and geometries outside the
saved set of 100.  A materially broader next step would have to change the
protograph/representation or relax the weight budget, rather than merely
allow a more general check-row permutation.
