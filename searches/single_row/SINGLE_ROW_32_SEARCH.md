# `[[32,4,6]]` single-row search

## Goal

This search targets one four-spin Ising row in one CSS block.  A successful
code must have:

- parameters `[[32,4,6]]`;
- four pairwise-disjoint, translation-related logical supports;
- a physical `C_4` translation cycling those four logical qubits;
- a transversal ZX operation (Hadamard followed by a fixed permutation) whose
  pairing restricted to the four row logicals has rank four;
- CSS checks and, initially, maximum stabilizer weight at most 12.

The search is deliberately separate from the qLDPC repository.  qLDPC is used
only to construct final `CSSCode` objects.

## Representation and protograph

The lift representation is

```text
GL(2,2) ~= S_3  x  C_4.
```

Each represented lift block therefore has size `2 * 4 = 8`.  The minimal
candidate with 32 data qubits uses the `L=4`, `J=2` GALA protograph:

```text
H_X = [F_0 F_1 | G_0 G_1],
H_Z = [G_0^T G_1^T | F_0^T F_1^T].
```

Each entry is expanded in the 16-element basis consisting of four independent
`2 x 2` binary matrices and four `C_4` translations.  Thus the unrestricted
ansatz has 64 binary generator coefficients.  The target check ranks are

```text
rank(H_X) = rank(H_Z) = 14,
k = 32 - 14 - 14 = 4.
```

## Disjoint row seed

A seed has weight six and selects one physical qubit in six of the eight
`C_4` fibres.  Its four `C_4` translates are disjoint, each has weight six,
and eight physical qubits remain as slack.  The seed search first keeps only
physical ZX folds for which the induced four-by-four pairing on this orbit has
rank four.

If an accepted code contains these four nontrivial logicals and has no logical
operator of weight below six, then its distance is exactly six: the seed orbit
itself supplies weight-six logical witnesses.

The seed stage found 100 full-rank witnesses after testing 495 seed/fold pairs.
The observed pairing-rank distribution was:

```text
rank 0: 75   rank 1: 24   rank 2: 135   rank 3: 161   rank 4: 100
```

## Generator ansatzes tested

1. **Automatic CSS relation.**  Set
   `G = q(x) * (a F_0 + b F_1, b F_0 + a F_1)`, where `q` is a nonzero
   `C_4` polynomial and `(a,b)` is identity, swap, or sum.  All 45 relations
   are CSS by construction.
2. **Independent entries with an exact displayed-check fold.**  Treat all 64
   coefficients independently and impose both the seed-kernel equations and
   the exact forward ZX-fold equations linearly.
3. **Automatic CSS with a rowspace fold.**  Keep only the seed equations during
   synthesis, then test whether the resulting check rowspaces are exchanged by
   any compatible physical fold.  This avoids overconstraining the redundant
   displayed stabilizer rows.
4. **Sparse `F`, affine solved `G`.**  Sample sparse `F` supports and solve the
   complete CSS commutation and seed-kernel equations as an affine binary
   system for `G`.  Every compatible physical fold is then tested.  This is the
   broadest implemented ansatz because it does not assume an automatic
   polynomial relation between `F` and `G`.

Every structurally accepted candidate is checked for CSS commutation, exact
dimension, disjoint nontrivial row logicals, `C_4` action, full-rank logical ZX
pairing, connected Tanner graph, and absence of logical operators of weights
one through five.

## Current results

| Search | Scope | Result |
|---|---:|---|
| Automatic CSS, exact fold | 3,600 relation/fold spaces | Every space fails the exact rank-capacity test |
| Independent entries, exact fold | 400 witness/check-fold spaces | 350 exact rank-capacity no-gos; exhaustive search of the 50 rank-capable spaces found no structural hit |
| Automatic CSS, rowspace fold | 4,500 relation spaces | 4,496 exact rank-capacity no-gos; sampled search of four survivors found no structural hit |
| Sparse `F`, affine solved `G` | All 100 rank-four witnesses; 910,490 completions | 111,306 target-rank codes and 14,281 weight-12 survivors; no compatible ZX fold |

No `[[32,4,6]]` candidate has yet been found.  The rank-capacity failures above
are exact only for their stated linear ansatzes.  The affine solved-`G` outcome
is a sampled negative and does not rule out the full `GL(2,2) x C_4` family.
The consistent bottleneck is the logical ZX condition, not obtaining `k=4` or
low stabilizer weight.

## Reproduction

Run the focused tests:

```bash
.venv/bin/python -m pytest tests/test_single_row.py -q
```

Run the affine solved-`G` stage without overwriting the saved result:

```bash
.venv/bin/python -m searches.single_row.run_single_row_solved_g \
  --run-name gl2-c4-l4-j2-solved-g-extension \
  --f-trials 2000 --g-samples 16 --maximum-check-weight 12
```

The first ten witnesses are stored under
`results/single-row/n32-k4/gl2-c4-l4-j2-solved-g-first10-v2/`; witnesses 10
through 99 and the combined aggregate are stored under
`results/single-row/n32-k4/gl2-c4-l4-j2-solved-g-remaining90/`.

## Best next extension

The next diagnostic experiment is to repeat the deterministic completion scan
with maximum stabilizer weight 16.  That will determine whether the current
weight-12 filter is hiding ZX-compatible codes.  If the ZX count remains zero,
the more important extension is a less rigid physical fold or a larger
represented top module rather than additional sampling of the same ansatz.

## Floating-`k` completion

The later floating-dimension identity-fold search dropped the requirement
`k=4` while retaining one protected, disjoint, fold-invariant logical `C4`
sector and check weight at most 12.  Sixteen exact seed-constrained generator
spaces produced 714 saved structural candidates:

```text
k=16: 223   k=18: 263   k=20: 196   k=22: 32
```

The original run certified 200 candidates, and the dimension-stratified
follow-ups certified every `k=16` and `k=18` candidate.  The final
deduplicating completion certified the remaining 28 `k=20` candidates.  All
714 saved candidates have exact distance two; none reaches `d >= 6`.

Artifacts:

- [`summary.json`](../../results/single-row/n32-floating/self-dual-w7-w12-v1/summary.json)
- [`remaining-certifications-v1.summary.json`](../../results/single-row/n32-floating/self-dual-w7-w12-v1/remaining-certifications-v1.summary.json)

The previously started weight-16 run was not resumed.  It remains only as an
aborted seed-witness artifact, consistent with the decision to keep this
revisit at stabilizer weight at most 12.
