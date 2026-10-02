# Half-checkerboard `[[256,32,>=6]]` search

This search targets one 32-site checkerboard in each QEC block.  The physical
lift is

\[
S_3^{\mathrm{GL}(2,2)}\times C_8\times C_4,
\]

with `L=4,J=2`, so one block has `n=256`.  A red block and a black block use
512 physical data qubits in total.

The implementation is seed-first.  It samples a graph-supported logical seed
using six of the eight physical `C8 x C4` translation fibres, translates it
through the entire logical grid, and rejects a proposed ZX fold unless the
resulting `32 x 32` pairing matrix has rank 32.  Only surviving fold/seed pairs
are passed to polynomial synthesis.

The usual GL alternating-form fold swaps the two top coordinates in the same
way on every protograph sheet.  In the initial precheck its sampled pairing
rank never exceeds 24.  The search therefore also includes
translation-commuting *coupled-protograph folds*: top-zero and top-one
coordinates may follow different sheet permutations.  These longer-cycle
folds readily expose rank-32 seed pairings.

For a selected seed and data fold, polynomial synthesis uses four independent
entries `F0,F1,G0,G1`.  It imposes the linear conditions

\[
H_X z=0,
\qquad
H_Z=R H_X P,
\]

where `P` is the data fold and `R` is a compatible permutation of check
coordinates.  HiGHS minimizes the number of group-ring monomials, subject to
at least one monomial in every entry.  CSS orthogonality, the reverse ZX-fold
condition, exact `k=32`, the logical-grid quotient rank, and binary stabilizer
weights are checked after lifting with the actual binary transpose.

Polynomial coefficients use four `S3` elements whose faithful lifts form a
basis of `M2(F2)`.  This is still the full two-dimensional matrix algebra, but
removes the two-dimensional kernel created by treating all six group elements
as independent variables.  Without this quotient, a MILP can report a sparse
formal polynomial whose binary check matrix is identically zero.

Before sparse optimization, the implementation lifts a basis of the complete
linear constraint nullspace and ranks the union of all possible `HX` rows.
An exact `k=32` folded code needs `rank(HX)=rank(HZ)=112`.  If the union row
space has rank below 112, that seed/fold choice is rejected exactly, regardless
of polynomial weight, without spending time in the MILP.

The output is isolated under `results/half-grid/<run-name>/`; the runner
refuses to overwrite an existing run.

Run a quick fold/seed feasibility scan with:

```bash
.venv/bin/python -m searches.half_grid.run_half_grid_search \
  --run-name gl2-l4-j2-seed260826-precheck \
  --seed-only --standard-trials 2000 \
  --trials-per-fold 20 --target-witnesses 100
```

Include bounded sparse-polynomial attempts with:

```bash
.venv/bin/python -m searches.half_grid.run_half_grid_search \
  --run-name gl2-l4-j2-seed260826-pilot \
  --trials-per-fold 20 --target-witnesses 100 \
  --polynomial-witnesses 2 --maximum-terms 16 --milp-seconds 30 \
  --feasibility-only
```

These are bounded searches, not exhaustive no-go tests.  A full candidate is
not accepted until all exact structural tests pass; distance certification is
the next stage after a structural hit.

## Initial run

The first saved precheck is under
`results/half-grid/gl2-l4-j2-seed260826-precheck-v2/`.

- In 2,000 weight-six samples for each ordinary GL fold, the maximum ZX
  pairing rank was 24.
- The coupled-fold scan found 100 rank-32 witnesses after testing 435
  fold/seed pairs.  Its best witnesses have a 32-entry permutation pairing.
- Thus the logical-support and ZX requirements are compatible at `n=256`;
  the earlier rank-zero/rank-24 behavior belongs to the ordinary involutory
  fold ansatz, not to every translation-commuting fold.

The exact rank-capacity pilot is stored under
`results/half-grid/gl2-l4-j2-rank-capacity-five-witnesses/`.  It tested all
four check-coordinate folds for each of five logical witnesses:

- 18 of 20 choices are exactly incapable of reaching `k=32`.  Their complete
  constrained polynomial spaces have maximum possible `rank(HX)` below 112.
- Two choices, both belonging to witness 1, have capacity exactly 112 and
  therefore survive this no-go filter.
- Focused 60-second feasibility searches on that surviving constraint space
  found no polynomial with at most 16 or at most 32 basis terms.  Both runs
  ended at the solver time limit without an incumbent, so this is not a proof
  that such sparse polynomials do not exist.

No `[[256,32,>=6]]` code has been claimed yet.  The positive result is the
existence of full-rank, disjoint logical-grid fold witnesses and multiple
linear generator spaces with sufficient rank capacity.

The runner can perform that individual-rank probe without invoking the MILP:

```bash
.venv/bin/python -m searches.half_grid.run_half_grid_search \
  --run-name gl2-half-grid-rank-probe \
  --polynomial-witnesses 20 --probe-only \
  --rank-probe-trials 1000 --rank-probe-restarts 20
```

## Completed 100-witness rank and CSS screens

The complete saved rank probe is
`results/half-grid/gl2-l4-j2-rank-probe-all100/`.  It tested all four check
folds for 100 full-pairing logical witnesses:

- 358 of 400 constrained polynomial spaces are exact rank-capacity no-gos.
- 42 spaces survive the capacity bound.
- 24 of those 42 sampled an individual polynomial with `rank(HX)=112`; rank
  112 appeared 1,018 times in aggregate.  Thus exact `k=32` check rank is not
  the main obstruction in the unrestricted linear space.
- A retained witness-31 polynomial has `rank(HX)=rank(HZ)=112`, a disjoint
  weight-six logical `C8 x C4` grid, and ZX pairing rank 32.  It nevertheless
  fails CSS commutation and the reverse fold.  Its dense checks have weights
  124 and 148.  This rejected candidate is stored under
  `results/half-grid/gl2-l4-j2-witness31-fold0-target112/`.

For the `L=4,J=2` block-circulant construction, both

```text
(G0,G1) = x^u y^v (F0,F1)
(G0,G1) = x^u y^v (F1,F0)
```

guarantee CSS commutation by pairwise cancellation of the two aggregate
commutators, even when `F0` and `F1` contain noncommuting GL matrices.  The
unshifted relations were screened over all 100 witnesses and all four check
folds.  Every one of the 400 choices in each family is an exact rank-capacity
no-go; the largest possible `rank(HX)` is only 64.  The runs are saved as
`gl2-l4-j2-auto-css-identity-all100` and
`gl2-l4-j2-auto-css-swap-all100` under `results/half-grid/`.

Finally, all 32 central shifts were tested for both relations on witness 31
and all four check folds.  All 128 choices in each relation are exact no-gos,
again with maximum rank capacity 64.  These runs are stored as
`gl2-l4-j2-auto-css-identity-w31-allshifts` and
`gl2-l4-j2-auto-css-swap-w31-allshifts`.

The current search stopped after these requested shift scans.  The result is
a useful separation of mechanisms: coupled folds solve the logical ZX
pairing problem, and the automatic relations solve CSS, but within this
tested ansatz their intersection collapses the stabilizer-rank capacity.
