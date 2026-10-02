# Native `L=6` faithful-2D-`S_3` search

This search revisits the previously untested `L=6` protograph over

\[
S_3\times C_8\times C_4
\]

using the faithful two-dimensional representation of `S_3`. Every output is
stored under `results/s3-linear-fast-falsification/l6-native/`; no previous
search result was replaced.

## Geometry and packing constraint

The physical length is

\[
n=6\cdot2\cdot32=384.
\]

Two support-disjoint logical checkerboards at distance `d` require

\[
384\geq2\cdot32d.
\]

Thus a two-grid `L=6` code with `d>=6` must have exactly `d=6`, and its two
weight-6 translated grids must partition all 384 physical qubits. Any
`d>=7` code is automatically too small to hold two such disjoint grids.

The ZX construction uses the only self-inverse shift of the three-entry half
protograph, `s=0`, together with the internal alternating-form swap.

## `J=1`: active-orthogonality family

Three native polynomial degree patterns were sampled:

| Entry weights | Ring check weight | Algebraic samples lifted | Result |
|---|---:|---:|---|
| `(1,1,2)` | 8 | 200 | Every candidate has a logical of weight at most four |
| `(1,2,2)` | 10 | 200 | Every candidate has a logical of weight at most four |
| `(2,2,2)` | 12 | 200 | Every candidate has a logical of weight at most four |

The first pass imposed maximum binary stabilizer weight 16. A second pass
relaxed it to 24, ensuring that candidates initially rejected only for their
binary lift weight were also tested. All 600 candidates still failed the
weight-at-most-four logical screen.

This is the genuinely active-GALA `L=6` option: offsets one and two are
latent and were required to be nonzero. In the tested native polynomial
patterns, the extra nonabelian structure was not enough to overcome the
small-`J` logical obstruction.

## `J=2`: fully active family

For `J=2`, all three cyclic row offsets are active, so there is no latent row
and no active-orthogonality advantage. It is nevertheless a nonabelian,
ZX-folded family and was screened separately.

From 200 algebraically valid samples in each degree pattern, 365 survived the
weight-at-most-four screen. Of these, 344 had maximum stabilizer weight at
most 16 and were taken through exact sparse distance searches.

| Distance/grid result | Count |
|---|---:|
| Logical of weight 5 | 27 |
| Certified `d=6`, with a graph-supported grid | 85 |
| Certified `d=6`, no graph-supported grid | 18 |
| `d>=7`, with a weight-7 grid | 15 |
| `d>=7`, exhaustive no weight-7 grid | 178 |
| `d>=7`, weight-7 grid search node-limited | 21 |

The 100 candidates with known grid seeds were jointly enumerated. All 100
enumerations were exhaustive and none produced two full-rank, jointly
ZX-complete grids.

Among the 85 distance-6 grid codes:

- 16 have at least one physically disjoint pair of translated grid supports;
- 15 reach combined logical rank 64;
- three have best joint ZX rank 32;
- one reaches joint ZX rank 56;
- none reaches the required rank 64.

No individual grid is self-ZX-complete. The best one-grid ranks are zero in
95 codes, 24 in four codes, and 16 in one code.

Only 20 of the 344 distance-screened candidates have the even-syndrome-parity
property useful for the one-round STAR protocol. Just one of those exposes a
distance-6 grid seed, but its translated orbit has logical rank 31 rather
than 32; it therefore fails even before the two-grid ZX test.

## Best near miss and local search

The closest broad-search candidate is

`s3-linear-l6-j2-w8-f54266873942780c`, with

\[
[[384,128,6]],\qquad w_{\max}=10.
\]

It has two independent, support-disjoint 32-qubit grids, but their combined
ZX pairing rank is only 56 rather than 64.

Its complete one-monomial neighborhood was then tested:

| Stage | Count |
|---|---:|
| Raw mutations | 571 |
| Algebraically valid and bottom-connected | 201 |
| No logical through weight four | 179 |
| Distance-6 grid codes | 107 |
| `d>=7` weight-7 grid codes | 9 |

All 116 grid enumerations were exhaustive. Twenty-four neighbors reproduce
joint ZX rank 56, but none improves it. The parent code admits only one simple
physical ZX fold, so changing the fold does not rescue the missing
eight-dimensional sector.

## Conclusion

No tested `L=6` code satisfies all of:

- distance at least six;
- two independent, support-disjoint `C_8 x C_4` logical grids;
- full joint ZX pairing rank 64;
- the physical translation action on both grids.

The result is strong evidence against the tested native weight-8/10/12
families, but it is not a mathematical no-go theorem for every possible
`L=6` polynomial. The most informative obstruction is the stable rank-56
ceiling around the best low-weight near miss, combined with the exact physical
packing requirement at `n=384`.

## Main outputs

- `structural-w16.jsonl` and `structural-w24.jsonl`: `J=1` searches.
- `j2-structural-w24.jsonl`: `J=2` structural screen.
- `j2-distance-w16.jsonl`: exact weight-5/6 screen.
- `j2-grid-sets-w16.jsonl`: exhaustive grid combinations.
- `local-rank56-structural.jsonl`, `local-rank56-distance.jsonl`, and
  `local-rank56-grid-sets.jsonl`: one-monomial neighborhood.
- `rank56-fold-analysis.json`: complete simple-fold check of the best near
  miss.
