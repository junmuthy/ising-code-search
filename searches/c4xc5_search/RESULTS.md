# `C4 x C5`, `n=40` single-row search

## Target

The search asks for a CSS code on

\[
n=2|C_4\times C_5|=40
\]

with floating `k >= 4`, distance at least six, check weight at most 12, and a
protected logical `C4` sector with all of the row-Ising properties:

- four pairwise-disjoint logical Z representatives;
- physical `C4` translation cycling the four representatives;
- a GALA ZX duality, allowing Hadamard followed by a physical permutation;
- full-rank ZX pairing on this same four-dimensional sector;
- connected Tanner graph.

No accepted code was found in the families below.

## 1. Exact self-dual family

The first construction was

\[
H_X=H_Z=[A\mid A^T],
\qquad b=a^\dagger,
\]

with every nonzero `a` of weight at most six.  This is an exact enumeration of

\[
\sum_{w=1}^{6}\binom{20}{w}=60{,}459
\]

polynomials.  Of these, 24,259 had `k >= 4` and 23,240 also had connected
Tanner graphs.  None had a compatible disjoint logical `C4` seed.  Because an
odd-weight seed pairs nondegenerately with itself under the identity fold, the
weight-seven and weight-nine seed test is exact for this family.

Artifact: [`exhaustive-w12-structural-v1/summary.json`](../../results/c4xc5-search/exhaustive-w12-structural-v1/summary.json)

## 2. Why the universal BB fold was insufficient

For arbitrary BB polynomials,

\[
H_X=[A\mid B],\qquad H_Z=[B^T\mid A^T],
\]

the universal ZX fold swaps the two physical halves and inverts the group
coordinate.  This proves code-level ZX duality, but it cannot close a single
logical grid on itself.  For an involutive half-swapping permutation `P`, every
overlap between `z` and `Pz` occurs in a doubled left/right pair, so

\[
z\cdot Pz=0
\]

over `GF(2)`.  The entire translated grid therefore has ZX pairing rank zero.
The blind and seed-constrained two-polynomial pilots reproduced this
obstruction.  They are diagnostics, not evidence against all two-polynomial
BB codes: the fold sends the grid into a different logical sector.

Artifacts:

- [`general-two-poly-pilot-v1/summary.json`](../../results/c4xc5-search/general-two-poly-pilot-v1/summary.json)
- [`seed-first-pilot-v1/summary.json`](../../results/c4xc5-search/seed-first-pilot-v1/summary.json)
- [`solved-pilot-v1/summary.json`](../../results/c4xc5-search/solved-pilot-v1/summary.json)

## 3. Half-preserving automorphism-dual family

To avoid the rank-zero mechanism, the final scan used each nontrivial
involutive group automorphism

\[
\sigma(x,y)\in\{(x^{-1},y),(x,y^{-1}),(x^{-1},y^{-1})\}
\]

and imposed

\[
b=\sigma(a)^\dagger.
\]

Hadamard followed by the half-preserving physical permutation `sigma` then
exchanges the X- and Z-check spaces.  Unlike the universal fold, these
permutations geometrically permit pairing rank four.

All `60,459` supports for each of the three involutions were checked, for
`181,377` polynomial/fold cases total.  The rank distribution included 30,840
codes with `k=4`, 15,960 with `k=6`, and 17,760 with `k=8`.  After rank and
connectivity filtering, 69,720 connected cases with `k >= 4` remained.  A
four-restart exact zero-syndrome meet-in-the-middle search found no disjoint
kernel seed whose ZX pairing had rank four.

The polynomial enumeration is exhaustive, but the logical-seed screen uses
four randomized fibre partitions and is therefore a bounded screen rather
than a proof that no such logical seed exists.  No structural candidate
reached distance certification.

Artifact: [`exhaustive-automorphism-dual-structural-v1/summary.json`](../../results/c4xc5-search/exhaustive-automorphism-dual-structural-v1/summary.json)

## Interpretation

At `n=40` and check weight at most 12, the simplest exact self-dual family is
ruled out, the universal BB fold necessarily pairs a single grid with another
logical sector, and the three natural half-preserving automorphism-dual
families gave a strong negative bounded result.  The cleanest follow-up, if we
return to this group, is a pairing-aware exhaustive kernel solver for the
69,720 eligible automorphism-dual codes.  A different follow-up is to accept
two logical `C4` sectors so the universal BB fold may exchange them—the same
mechanism that is an obstruction for a single-copy target can be useful for a
multi-copy target.

## Reproduction and checkpoints

Focused tests:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache \
  .venv/bin/python -m pytest tests/test_c4xc5_search.py -q
```

Future automorphism-dual runs stream every structural candidate immediately
and atomically update `progress.json` every 1,000 polynomials and at every
completed polynomial weight.  The runner also prints each checkpoint with
flushing enabled.

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache \
  .venv/bin/python -m searches.c4xc5_search.run_automorphism_dual \
  --run-name NEW-UNIQUE-NAME --seed-restarts 4
```
