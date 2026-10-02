# Batched `C4 x C2` Ising-grid search

This directory implements the compact half-grid search with eight protected
logical sites and either two or four STAR-injection batches.  It is isolated
from the earlier `C8 x C4` searches and never overwrites their artifacts.

## Acceptance gates

Candidates are tested in this order:

1. `n <= 200`, `k >= 8`, connected Tanner graphs, and actual check weight at
   most 16;
2. exact CSS commutation and strict presentation-level GALA ZX duality;
3. physical `C4` and `C2` translation automorphisms;
4. a certified absence of nontrivial X/Z logicals below weight six;
5. an invariant eight-dimensional regular `C4 x C2` logical module with
   full-rank ZX pairing;
6. disjoint representatives within at least one translation-covariant
   two- or four-batch partition, after arbitrary stabilizer dressing.

The strict-fold generator family uses

\[
\rho(G_j)=\rho(F_{j+a})^T,
\]

where `a` is the selected fold axis and `rho` is the represented binary lift.
At `L=4`, axis zero gives the particularly strong identity fold `H_X = H_Z`.
Axis one gives strict GALA ZX duality through a fixed reflection of the two
protograph blocks in each physical half.  In both cases transversal Hadamard,
followed by the recorded qubit permutation when necessary, exchanges the
X- and Z-check spaces exactly.  The logical search uses the same fold to pair
the protected logical grid.

Three represented lift groups are implemented:

| Search family | Physical length at `L=4` | Representation detail |
| --- | ---: | --- |
| `abelian` | 32 | trivial top over `C4 x C2` |
| `s3-linear` | 64 | faithful two-dimensional `GL(2,2)` lift |
| `s3-natural` | 96 | natural three-dimensional permutation lift |

The same implementation also accepts one entry (`L=2`), giving lengths 16,
32, and 48 for these three lifts, or three entries (`L=6`), giving lengths
48, 96, and 144.

For the faithful representation, group inverse is not binary transpose.  Its
Z parent is therefore built from the actual represented binary transpose,
while retaining the same GALA block-circulant construction.  The reported
check-weight ceiling is always applied to the lifted binary matrices.

## Check-weight profiles

For `L=4,J=1`, if the two F entries have term weights `(w_0,w_1)`, the nominal
binary check weight is

\[
w_{\rm check}=2(w_0+w_1).
\]

The intended scan order is:

| Priority | Profiles | Nominal check weight |
| --- | --- | ---: |
| 1 | `(2,2)` | 8 |
| 2 | `(3,3)`, `(2,4)`, `(4,2)` | 12 |
| 3 | `(4,4)`, `(3,5)`, `(5,3)` | 16 |

Actual lifted row weights are recorded separately because represented-group
collisions can lower them.  Weight 16 is a hard ceiling, not the default.

## Reproducible runs

Use the repository virtual environment and give Numba a writable cache:

```bash
cd /home/judah_unmuth/gala-code-search
export NUMBA_CACHE_DIR=/tmp/gala-numba-cache

.venv/bin/python -m searches.batched_c4xc2_search.run_search \
  --family quotient-seed --run-id quotient-seed-v1

.venv/bin/python -m searches.batched_c4xc2_search.run_search \
  --family abelian --profile 2,2 --candidates 1000 \
  --run-id abelian-l4-w8-seed260831-v1

.venv/bin/python -m searches.batched_c4xc2_search.run_search \
  --family s3-linear --profile 3,3 --candidates 10000 \
  --run-id s3-linear-l4-w12-seed260831-v1

.venv/bin/python -m searches.batched_c4xc2_search.run_search \
  --family s3-natural --profile 3,3 --candidates 10000 \
  --run-id s3-natural-l4-w12-seed260831-v1

.venv/bin/python -m searches.batched_c4xc2_search.run_search \
  --family s3-natural --profile 4,4 --fold-axis 1 --candidates 1000 \
  --run-id s3-natural-l4-w16-fold1-seed260831-v1
```

Every candidate is appended immediately to `candidates.jsonl`.  Progress and
stage counts are written atomically to `summary.json` at least every 30
seconds.  A stopped run can be continued with the identical options plus
`--resume`.

Final hits also save exact `H_X` and `H_Z` arrays in a compressed NPZ file.

For small logical spaces (`k <= 12`), the logical-module gate enumerates every
nonzero logical class exactly.  A syndrome-hash prefilter finds logicals
through weight four without MILP; the MILP remains responsible for weight
five and for the final `d >= 6` certificate.

## Initial negative control

The quotient of the saved `[[384,200,7]]` polynomial is structurally clean:
it is a connected exact-ZX `[[96,52,*]]` code with uniform weight-16 checks
and column degree four.  It also has many full-rank regular `C4 x C2` logical
modules.  However, it has a nontrivial weight-two logical, so quotienting the
old polynomial does not preserve distance.  The runner saves this result as a
reproducible first-stage control.

See `searches/paired_polynomial/RESULTS.md` for the completed initial abelian scans and nonabelian pilot
yields.

## Independent `F/G` half-swap family

The compact follow-up keeps `L=2,J=1,n=48`, but searches `F` and `G`
independently.  Both supports are required to be invariant under represented
binary transpose.  Therefore

\[
H_X=[A\mid B],\qquad H_Z=[B\mid A],
\]

and transversal Hadamard followed by swapping the two 24-qubit halves is an
exact presentation-level GALA ZX fold.  Unlike the earlier `G=F^T` family,
CSS commutation is an active nonabelian search constraint.

Run a checkpointed profile with

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_halfswap_search \
  --f-weight 6 --g-weight 10 --candidates 1000 \
  --run-id s3-natural-l2-halfswap-w16-610-pilot-1000-260831-v1
```

The runner writes each candidate immediately, checks `d >= 6`, exhaustively
enumerates logical classes whenever `k <= 12`, and attempts batch dressing
only after finding a regular ZX-paired grid.

The completed even-weight pilots can be diagnosed reproducibly with

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.analyze_halfswap_survivors \
  results/batched-c4xc2-search/s3-natural-l2-halfswap-*-pilot-1000-260831-v1 \
  --run-id s3-natural-l2-halfswap-d6-action-analysis-260831-v1
```

This computes the exact induced `C4 x C2` matrices on the logical quotient
and the dimension of the algebra spanned by the eight translations.  The
latter is an immediate upper bound on the rank of any cyclic logical orbit.

## X-reflected half-swap family

The next broadening replaces individual transpose invariance by

\[
F^\dagger=\alpha(F),\qquad G^\dagger=\alpha(G),\qquad
\alpha:x\mapsto x^{-1},\ y\mapsto y.
\]

Consequently, x exponents need not occur in inverse pairs.  The exact ZX
operation is transversal Hadamard followed by x reflection within both
24-qubit halves and their exchange.  A fixed x reflection of the check rows
maps the resulting presentation exactly to the opposite CSS sector.

The runner places the necessary translation-algebra gate before distance:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_twisted_halfswap_search \
  --f-weight 8 --g-weight 8 --candidates 1000 \
  --run-id s3-natural-l2-xreflect-halfswap-w16-88-pilot-1000-260831-v1
```

Each durable record contains the exact induced translation-algebra dimension.
Only dimension-eight candidates reach the low-distance screen.

The witness-guided follow-up starts from every dimension-eight pilot survivor,
uses fold-preserving support-orbit swaps, and ranks candidates by their exact
logical spectrum through weight four:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_twisted_witness_beam \
  results/batched-c4xc2-search/s3-natural-l2-xreflect-halfswap-*-pilot-1000-260831-v1 \
  --run-id s3-natural-l2-xreflect-witness-beam20-g3-260831-v1 \
  --generations 3 --beam-width 20 --progress-every 100 \
  --checkpoint-seconds 15
```

Its `evaluations.jsonl` is append-only, and `beam-generation-N.json` preserves
every selected presentation.  Supplying the same options with `--resume`
continues an interrupted run without reevaluating candidates.

To cross unfavorable one-orbit intermediates, seed the final beam directly
and request exact two-orbit replacements:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_twisted_witness_beam \
  results/batched-c4xc2-search/s3-natural-l2-xreflect-witness-beam20-g3-260831-v1/beam-generation-3.json \
  --run-id s3-natural-l2-xreflect-witness-double-beam20-g2-n1000-260901-v1 \
  --generations 2 --beam-width 20 --mutation-radius 2 \
  --neighbors-per-parent 1000 --mutation-seed 260901 \
  --progress-every 100 --checkpoint-seconds 15
```

This run found an exact natural-`S3` fiber obstruction: all 16 translated
three-point fibers are independent X and Z logical classes in every one of
its 2,464 algebra-qualified survivors.  See `searches/paired_polynomial/RESULTS.md` for the complete
counts and interpretation.

## Faithful `GL(2,2)` x-reflected family

The faithful two-dimensional representation removes the natural lift's
fixed three-point fiber and gives `L=2,n=32`.  The runner below implements
the actual represented transpose, the x-reflected half-swap ZX fold, and the
translation-algebra-first gate:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_faithful_twisted_search \
  --f-weight 8 --g-weight 8 --candidates 1000 \
  --check-weight-ceiling 16 \
  --run-id s3-linear-l2-xreflect-halfswap-w16-88-pilot-1000-260901-v1
```

The exact-commuting centralizer ansatz

\[
G=p(x,y)I+q(x,y)F
\]

uses the same runner with:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_faithful_twisted_search \
  --f-weight 4 --commutant-p-weight 8 --commutant-q-weight 1 \
  --candidates 1000 --check-weight-ceiling 16 \
  --run-id s3-linear-l2-commutant-f4-p8-q1-pilot-1000-260901-v1
```

Setting `--commutant-p-weight 0` runs the proportional-block boundary
diagnostic.  It restores full translation algebra in some candidates but
always exposes a weight-two logical in the completed pilots.  All candidate
records and summaries are append-only; use a fresh run ID or the exact saved
configuration with `--resume`.

## Larger faithful protograph probes

The `n=64` and `n=96` probes use

\[
G_i=p_iI+qF_{i+a}
\]

to construct active CSS orthogonality while avoiding proportional physical
halves.  For example:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.run_faithful_twisted_protograph_search \
  --f-profile 2,2 --p-profile 2,2 --coupling-shift 1 \
  --candidates 1000 --check-weight-ceiling 16 \
  --run-id s3-linear-twisted-proto-n64-f22-p22-a1-pilot-1000-260901-v1
```

The completed probes restore full translation algebra but expose only
distance-two through distance-four codes.  See `searches/paired_polynomial/RESULTS.md` for aggregate
counts.

## Published `[[64,8,8]]` acceptance test

The Liang--Chen self-dual BB code
([arXiv:2510.05211v2](https://arxiv.org/abs/2510.05211), Table 1 and Appendix
A) on the twisted torus
`x^4 y^4=1, y^8=1` is reconstructed and tested with:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.analyze_published_selfdual_bb64 \
  --output-dir results/batched-c4xc2-search/published-selfdual-bb64-ising-analysis-260901-v1

NUMBA_CACHE_DIR=/tmp/gala-numba-cache .venv/bin/python -m \
  batched_c4xc2_search.search_published_bb64_grid_presentations \
  --grid-npz results/batched-c4xc2-search/published-selfdual-bb64-ising-analysis-260901-v1/published-bb64-checks-and-grid.npz \
  --output-dir results/batched-c4xc2-search/published-selfdual-bb64-grid-presentations-260901-v2
```

The second command finds a regular, fully ZX-paired `C4 x C2` logical grid
whose weight-eight representatives split into four raw disjoint pairs.  The
exact matrices, logical orbit, and automorphism permutations are saved in its
NPZ artifact.  Its ZX pairing is full rank but dense (three ones per row), so
a sitewise-Hadamard requirement is stronger than the formal GALA ZX-duality
test and remains to be checked in a simultaneous logical basis.
