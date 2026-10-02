# Abelian `[[32,4,>=6]]` single-row search

## Outcome

The bounded abelian search found no `[[32,4,>=6]]` code satisfying all of the
single-row Ising conditions.  This is an exact negative result within the
families and support bounds below; it is not a claim that no abelian
`[[32,4,6]]` stabilizer code exists.

No distance calculation was needed.  Every candidate failed an earlier exact
structural condition: the target rank or the existence of a disjoint logical
`C4` orbit with full-rank ZX pairing.

## Families searched

| Family | Lift group | Construction | Check-weight ceiling |
|---|---|---|---:|
| Strict GALA | `C8` | `L=4`, `J=2`, Proposition-6 polynomial ZX fold | `12` |
| Strict GALA | `C2 x C4` | `L=4`, `J=2`, Proposition-6 polynomial ZX fold | `12` |
| Self-dual two-block/BB | `C4 x C4` | `H_X=H_Z=[A\mid A^T]` | `12` |

For the strict GALA branches, all two-polynomial supports with combined weight
at most six were enumerated.  For each of the four Proposition-6 sector folds,
the `G` generators were derived from the `F` generators by the required
transpose-and-translate relation.  The code verifies CSS orthogonality, target
ranks `rank(H_X)=rank(H_Z)=14`, Tanner connectivity, and exact ZX exchange up
to the check-row reindexing allowed by the paper.

For `C4 x C4`, all 14,892 nonzero polynomials of weight at most six were
enumerated.  The diagonal involutions are exact physical ZX dualities because
`H_X=H_Z`.

## Exact counts

| Lift group | Generator cases | Target rank | Connected | Formal ZX folds tested | Disjoint logical grids |
|---|---:|---:|---:|---:|---:|
| `C8` | 59,568 | 19,968 | 18,944 | 37,888 | 0 |
| `C2 x C4` | 119,136 | 0 | 0 | 0 | 0 |
| `C4 x C4` | 14,892 | 0 | 0 | 0 | 0 |

The `C2 x C4` check-rank histogram is

```text
rank 4:     352
rank 6:   1,280
rank 7:   3,072
rank 8:  19,584
rank 10: 15,360
rank 12: 39,936
rank 16: 39,552
```

Here `rank(H_X)=rank(H_Z)` in every case.  Rank 14 never occurs.

The `C4 x C4` check-rank histogram is

```text
rank 4:      28
rank 6:     176
rank 7:      64
rank 8:     912
rank 9:     960
rank 10:    320
rank 12:  7,488
rank 16:  4,944
```

Again, rank 14 never occurs.  A supplementary rank screen at polynomial
weights seven and eight also found no rank-14 code: all 11,440 weight-seven
polynomials have rank 16, while weight-eight ranks range from 2 through 12.
Thus raising this branch's check-weight ceiling from 12 to 16 does not rescue
the target dimension.

## Why only weight-seven logical seeds were searched

The desired four logical qubits form one orbit under a physical order-four
translation `T`.  The 32 physical qubits split into eight `C4` fibres.  If the
four translated logical supports are disjoint, a seed can contain at most one
qubit from each fibre.  Distance at least six and disjointness then restrict
the common logical weight to 6, 7, or 8.

For every paper-compatible fold used here, the ZX pairing of an even-weight
disjoint `C4` orbit is a singular circulant matrix: its all-ones Fourier mode
is null.  Direct exhaustive precomputation confirms zero full-rank pairings at
weights six and eight.  Weight seven is therefore the only possible case.

The search enumerated every weight-seven disjoint orbit, modulo an irrelevant
common `C4` translation.  For the viable `C8` folds this is 32,768 seeds per
fold.  None lies in the required kernel as four independent logical classes
for any connected target-rank code.

## Interpretation

`C4 x C4` was worth checking: it gives the logical translation directly and
makes exact ZX duality especially simple.  Its failure is stronger than a bad
distance result, however—the bounded self-dual polynomial family cannot
produce `k=4` at all, even after the quick extension through check weight 16.

The `C8` branch comes closest.  It readily produces connected `[[32,4,*]]`
codes with formal GALA ZX folds, but the same fold/translation symmetry prevents
their four logical qubits from admitting the required disjoint, ZX-complete
orbit.  The next useful abelian experiment should therefore change structure,
not merely sample more generators from this same ansatz—for example a larger
physical lift with shortening/puncturing, or a different protograph that
retains the logical `C4` quotient.

## Reproduction

From the standalone search directory:

```bash
NUMBA_CACHE_DIR="$PWD/results/numba-cache" \
  .venv/bin/python -m searches.abelian.run_abelian_32_search \
  --run-name paper-zx-w12-structural-v1 --no-distance
```

The command refuses to overwrite an existing run.  The saved machine-readable
results are under
`results/single-row/abelian-n32-k4/paper-zx-w12-structural-v1/`.
