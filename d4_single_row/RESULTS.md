# Faithful `D4 x C4` single-row search at `n=32`

Here `D4` denotes the order-eight symmetry group of the square.  The search
uses its faithful action on the four square vertices, tensored with the
regular `C4` translation.  With the minimal two-block `L=2`, `J=1` GALA
protograph,

\[
n=2\cdot4\cdot4=32.
\]

The binary span of the eight represented `D4` elements has dimension six.
Each of the two polynomial entries `A,B` therefore has 24 binary coefficients:
six top-algebra basis matrices times four `C4` translations.

## Acceptance conditions

Every retained code must have:

- floating `k >= 4`;
- stabilizer weight at most 12;
- CSS orthogonality, including the nonabelian active-orthogonality condition;
- four pairwise-disjoint logical supports cyclically permuted by physical
  `C4` translation;
- a GALA ZX fold with pairing rank four on that same logical sector;
- a connected Tanner graph;
- no X or Z logical operator of weight below six.

The logical seed has weight seven and occupies seven of the eight physical
`C4` fibres.  Its four translates are disjoint.  Odd weight gives an identity
pairing matrix whenever the physical ZX fold fixes the relevant top fibres.

## Identity-fold phase

Sixteen seed-constrained linear spaces were exhaustively enumerated.  Every
space had nullity eight, hence 255 nonzero generator combinations:

```text
generator combinations:             4,080
check weight at most 12 and CSS:       440
connected structural candidates:       208
distance at least six:                    0
distance two:                           208
```

The exact-fold constraints also restricted every tested code to `k >= 16`.

Artifact: [`identity-fold-sixteen-witness-v1/summary.json`](../results/single-row/d4-c4-n32/identity-fold-sixteen-witness-v1/summary.json)

## Nonidentity involutive-fold phase

There are six involutions in the four-vertex `D4` action.  Three are
fixed-point-free.  They are ruled out geometrically for a single graph-supported
logical sector: every top fibre is exchanged with a distinct mate, so every
overlap is doubled and vanishes over `GF(2)`.  Their logical ZX pairing is
identically zero.

The remaining three data folds were combined with all six involutive check
folds.  Four independent seed witnesses were used per feasible data fold.  The
already-completed identity/identity cases were omitted, leaving 68 exact
constraint spaces:

```text
nullity-zero spaces:                    24
nullity-eight spaces:                   44
generator combinations:            11,220
check weight at most 12 and CSS:     1,576
connected structural candidates:      336
distance at least six:                   0
distance two:                          336
```

Again, all generated codes had `k >= 16`.  The dominant weight-two witnesses
were `[0,8]`, connecting opposite square vertices at the same `C4` position,
and `[0,2]`, connecting opposite positions within one `C4` fibre.

Artifact: [`involutive-fold-four-per-data-v1/summary.json`](../results/single-row/d4-c4-n32/involutive-fold-four-per-data-v1/summary.json)

## Scope of the negative result

The generator enumeration is exact inside every saved seed/fold constraint
space, and all structural candidates were distance-screened.  The seed
witnesses are sampled rather than exhaustive.  The search covers exact
displayed folds made from involutive `D4` top-coordinate permutations; it does
not yet cover arbitrary rowspace-only folds, affine `C4` phase shifts in the
fold, or every inequivalent weight-seven seed.

Within this natural compact ansatz, the failure is stronger than a mere lack
of distance-six examples: the fold and seed constraints force high `k`, low
check rank, and persistent distance-two logicals.  More sampling of the same
16- or 68-space design is therefore less promising than broadening the fold
or protograph.

## Checkpoints and reproduction

Both runners atomically update `progress.json`; the identity runner writes one
checkpoint and one task artifact per seed, while the nonidentity runner does
so for every fold space.

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache \
  .venv/bin/python -m d4_single_row.run_search \
  --run-name NEW-IDENTITY-RUN --seed-witnesses 16 --workers 8

NUMBA_CACHE_DIR=/tmp/gala-numba-cache \
  .venv/bin/python -m d4_single_row.run_fold_search \
  --run-name NEW-FOLD-RUN --witnesses-per-data-fold 4 --workers 8
```
