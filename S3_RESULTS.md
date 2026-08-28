# `S_3 x C_8 x C_4` Ising-code search

This search tests polynomial GALA codes with the natural degree-three lift of
`S_3` and the exact bottom translation group `C_8 x C_4`. The physical bottom
translations act on every certified logical orbit as the regular 8-by-4 grid.

## Families

| Family | Parameters before distance | Generator pattern | ZX relation |
| --- | --- | --- | --- |
| compact | `L=4`, `J=1`, `n=384` | `wt(F_0)=wt(F_1)=4` | `G_0=F_1^dagger`, `G_1=F_0^dagger` |
| extended | `L=8`, `J=2`, `n=768` | `wt(F_i)=2` for `i=0,1,2,3` | `G_(i+2)=F_i^dagger` |

After rejecting cancellations in the natural binary lift, both have check
weight 16 and qubit degree four. The algebraic filters require every active
aggregate commutator to vanish, a latent aggregate commutator to remain
nonzero, and the bottom support to generate all of `C_8 x C_4`.

The original weight-12 pilots were rejected before the saved batches:

- all 100 valid `L=4` samples had a weight-2 logical;
- among 100 valid `L=8` samples, 16 had a weight-2 logical and 84 had a
  weight-4 logical.

This is why the saved search uses uniform-degree-four, weight-16 checks.

## Saved search batches

| Family | Raw samples | Algebraic survivors | Valid binary lifts | Exact `d=6` | No kernel through weight 6 | Any graph seed | Full rank-32 graph orbit | Two disjoint graph sectors | Fold-closed full logical basis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L=4` | 7,888 | 100 | 85 | 20 | 9 | 15 | 4 | 0 | 0 |
| `L=8` | 40,231 | 100 | 93 | 14 | 23 | 10 | 5 | 5 | 0 |

The exhaustive Tanner search fixes the first qubit to one representative of
each bottom-translation orbit. This is complete up to `C_8 x C_4`
translation. It searches kernel weights in increasing order and classifies
each witness modulo the stabilizer row space. A graph seed is additionally
restricted to at most one qubit in each internal fibre; its 32 translations
are therefore pairwise disjoint by construction.

The complete machine-readable records are:

- [`results/s3-l4-w16-seed240828.jsonl`](results/s3-l4-w16-seed240828.jsonl)
- [`results/s3-l8-w16-seed240827.jsonl`](results/s3-l8-w16-seed240827.jsonl)

The later exhaustive revisit using full off-diagonal pairing between different
translation grids is reported in
[`S3_REVISIT_RESULTS.md`](S3_REVISIT_RESULTS.md).  It confirms that none of
the saved `L=4` or `L=8` candidates is rescued by the corrected joint-pairing
criterion.

## A certified compact code

The best clean single-grid representative is
`s3-l4-j1-w16-e1bb058d91825488`:

```text
F_0 = 1 + y^2 + tau_0 x^5 y^3 + tau_2 x^7
F_1 = x^4 y^3 + x^7 y^3 + tau_1 x^7 y^2 + tau_2 x^3
G_0 = F_1^dagger
G_1 = F_0^dagger
```

It has certified parameters `[[384,194,6]]` and passes:

- active CSS orthogonality and a nonzero latent commutator;
- connected bottom support and exact `C_8` and `C_4` stabilizer automorphisms;
- physical translation-type ZX duality;
- check weight 16 and uniform X/Z column degree four;
- no X- or Z-logical below weight six;
- a weight-six seed with support `[32,75,187,261,348,373]`;
- 32 pairwise-disjoint translates arranged as an exact `C_8 x C_4` grid;
- rank 32 for that orbit modulo Z stabilizers.

The crucial limitation is that the ZX-folded X orbit has pairing rank zero
with this Z orbit. The physical code is ZX-dual, but this particular clean
32-qubit grid is sent to a different logical sector; it is not an invariant
logical block on which the fold acts as 32 Hadamards.

The best valid two-sector `L=8` near-miss is
`s3-l8-j2-w16-b77391bd2f2fd82b`, a `[[768,388,6]]` code. It has 64 disjoint
translated representatives, but their combined rank modulo stabilizers is 60
and their combined ZX pairing rank is 56, rather than 64.

## Why useful rate `1/6` fails here

There is a parity obstruction at the desired saturation point. Suppose `N`
disjoint Z representatives have distance-six weight and saturate `n=6N`.
They partition the physical qubits. If the ZX fold closes on those logical
qubits, their folded X representatives also partition the physical qubits.
The sum of every row in the logical Z-X pairing matrix is then
`wt(Z_i) mod 2 = 6 mod 2 = 0`. Consequently the all-ones vector lies in the
kernel of the pairing matrix, so the selected logical basis cannot be
symplectic and complete.

For the present protographs, useful logical grids come in multiples of 32:

- `L=4`: two grids would be the forbidden saturated `64/384 = 1/6`; one grid
  gives only `1/12`;
- `L=8`: four grids would be the forbidden saturated `128/768 = 1/6`; three
  grids give only `1/8`.

Therefore these `L=4` and `L=8` S3 families improve the *global* stabilizer
encoding rate, but they do not improve the rate of disjoint, ZX-closed Ising
logical grids. For that operational metric, the existing packed
`[[448,64,7]]` BB code remains better at `64/448 = 1/7`.

## Reproduce

```bash
.venv/bin/python run_s3_ising_search.py \
  --family l4-w16 --accepted-candidates 100 --seed 240828 \
  --output results/s3-l4-w16-seed240828.jsonl

.venv/bin/python run_s3_ising_search.py \
  --family l8-w16 --accepted-candidates 100 --seed 240827 \
  --output results/s3-l8-w16-seed240827.jsonl
```

The subsequent weight-12 `L=12,J=3` search successfully used physical-support
slack and full combined pairing during selection.  Its four-grid
`[[1152,580,6]]` result is reported in
[`S3_MANY_COPY_RESULTS.md`](S3_MANY_COPY_RESULTS.md).
