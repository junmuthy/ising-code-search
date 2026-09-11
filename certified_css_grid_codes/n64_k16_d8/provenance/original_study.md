# Logical operators of the fixed [[64,16,8]] code

Date: 2026-09-11. This studies the already-certified code
`71e1114cdaa26f7c2bc3cbd2098452d03aee37bfcc62016f59367dc784c7625e`.
No new code family is searched. The 255-class coefficient/fold continuation
has finished, with no lower-check-weight hit.

## Main results

- **The recommended basis now has all 16 X and all 16 Z logicals of weight
  10**, with the same code, physical translations, and physical Hadamard fold.
  Weight 10 is optimal for a full pure CSS logical grid under the saved
  translation action and common-basis H. Every physical qubit appears in
  exactly five of the 32 chosen supports, attaining the minimum possible peak
  combined load.
- The original logical labels admit weight-12 representatives for **all 16 X
  and all 16 Z logicals**, preserving the canonical pairing, logical grid, and
  Hadamard permutation. The old X representatives had weights 16–24.
- Exact enumeration proves **12 is the minimum for each individual logical
  in that original basis**, not merely the best representative found.
- Weight-eight logicals do exist: there are **1,016 pure Z operators**, in
  **414 logical classes**, and 1,016 pure X counterparts. Their logical span
  has dimension 16 in each sector.
- Nevertheless, **no all-weight-eight canonical X/Z logical grid exists for
  the saved translation action**. All 192 cyclic seed classes with weight-eight
  Z representatives have been tested; none has weight-eight representatives
  for its canonical dual X grid. This obstruction already holds without
  imposing the Hadamard condition.

The last statement is not a no-go for other physical symmetry representations,
non-equivariant bases, or mixed-Pauli logical bases. It also does not eliminate
weight-eight products useful for particular state-preparation protocols. A
weight-eight Z grid alone is possible, but its dual X grid cannot also have
weight eight under the saved translations.

## Recommended weight-10 logical basis

The code remains exactly `[[64,16,8]]`, with the original independent check
matrices: 24 weight-12 X checks and 24 weight-12 Z checks. Only logical labels
and representatives change. This does not contradict the earlier minimum
weight 12 for the original individual logical labels.

The selected basis change is particularly simple. In logical classes,
modulo stabilizers, for all `(i,j)` in `Z4 x Z4`:

```text
Z'_(i,j) = Z_(i,j) Z_(i,j+1) Z_(i,j+2)
X'_(i,j) = X_(i,j) X_(i,j+1) X_(i,j+2)
```

The second coordinate is reduced modulo four. These three-neighbor products
form a canonical basis: the binary change-of-basis matrix is orthogonal.
Stabilizer dressing then gives the saved weight-10 physical representatives.
The seed's old logical coefficient integer is `7`.

Translations still act as `(i,j)->(i+1,j)` and `(i,j)->(i,j+1)`. The same
order-two physical fold implements individual logical H followed by

```text
(i,j) -> (i, 2*i + 3 - j) mod 4.
```

Both Hadamard directions are explicitly checked. The chosen representative
lists satisfy the Hadamard pairing exactly; translation identities hold
modulo stabilizers, and need not literally permute the selected supports.

| Property | Balanced recommendation | Lower-overlap alternative |
| --- | --- | --- |
| Logical support sizes | All `32` of weight `10` | All `32` of weight `10` |
| Total logical support | `320` | `320` |
| Maximum combined participation per physical qubit | `5`, exactly optimal | `6` |
| Maximum overlap of any two distinct logical supports | `4` | `3` |
| Logical seed coefficient | `7` | `259` |

The balanced recommendation has load exactly five on **every** physical qubit.
The average-load bound `320/64=5` proves its peak load is optimal. Pairwise
overlap optimality is not proved for either option; the second option is a
trade-off, not a claimed global optimum. Logical support decreases from 384
in the all-weight-12 handoff to 320, a 16.7% reduction. These are support
metrics, not fault-tolerant circuit resource estimates.

Example zero-based physical supports in the balanced handoff:

```text
Z'_(0,0): [1,5,17,20,28,37,47,49,52,59]
X'_(0,0): [6,7,12,16,26,32,38,39,47,48]
```

- [Recommended candidate and logical basis change](weight_ten_handoff_v1/best/candidate.json).
- [All recommended logical supports](weight_ten_handoff_v1/best/logical_supports.json).
- [Binary logical bases](weight_ten_handoff_v1/best/logical_bases.npz).
- [Independent full audit](weight_ten_audit_v1/full_audit.json).
- [Lower-overlap candidate](weight_ten_audit_v1/lower_overlap/candidate.json) and
  [supports](weight_ten_audit_v1/lower_overlap/logical_supports.json).
- [Complete ranking and minimum representatives](weight_ten_handoff_v1/report.json).

### Exhaustive search and optimality

All 32,768 unit seed classes form 2,048 free logical-translation orbits.
Translating a seed simply relabels both canonical bases, preserving Hadamard
compatibility and minimum support. Exactly 256 orbit representatives (4,096
raw unit classes) pass both logical Hadamard directions for the saved fold.

For each of these 256 representatives, all `2^24` Z-stabilizer dressings were
enumerated. Their exact minimum-support distribution is:

| Minimum seed support | Translation-orbit representatives |
| --- | --- |
| `10` | `14` |
| `12` | `220` |
| `14` | `22` |

Translation transitivity and Hadamard transfer each minimum to all 32 logical
cosets of that basis. All 256 enumerations were repeated with the earlier
standalone coset implementation, including agreement of full weight
histograms. Independent 16-by-16 GF(2) inverse/action matrices rechecked the
entire 2,048-orbit algebraic filter and the dual-coset identifications.

For the 14 weight-10 bases, every minimum seed representative was retained
(two or four representatives per seed coset). A reproducible 128-restart
coordinate-descent search selected stabilizer dressings independently for
the 16 Z logicals, with X representatives paired by the Hadamard fold.
The tie-breaking objective was peak combined load, then maximum pairwise
overlap, then sum of squared loads. Both recommended outputs were audited
using array-based commutation, rank, normalizer membership, canonical pairing,
translation action, and two-way H checks.

The minimum possible maximum support is exactly **10** in the stated scope:

1. The complete earlier weight-eight calculation excludes an all-weight-eight
   canonical grid under these translations, regardless of fold.
2. All-qubit X and Z belong to the stabilizer group, so every pure opposite-type
   normalizer has even physical weight; weights nine and eleven are impossible.
3. A certified weight-ten grid now attains the next possible value.

The lower bound also makes the total logical support 320 optimal under the
same common-H/regular-grid contract: transitivity and H equate all individual
coset minima. No claim is made for different physical translation actions,
non-equivariant bases, or mixed-Pauli logical bases. Additional Hadamard folds
were not searched because the optimal target was attained with the saved fold.

- [Saved-fold search summary](weight_ten_saved_fold_v1/summary.json),
  [all 2,048 orbit records](weight_ten_saved_fold_v1/catalog.json), and
  [256 exact coset enumerations](weight_ten_saved_fold_v1/cosets.jsonl).
- [Search implementation](../distance8_weight10_search.py),
  [finalization and balancing](../distance8_weight10_finalize.py),
  [independent algebraic audit](../distance8_weight10_audit.py), and
  [regression tests](../test_distance8_weight10_search.py).

## Same-label weight-12 handoff

Let `P` be the saved involutive physical Hadamard fold and let
`sigma(i,j)=(i,2i+1-j)` modulo four be its logical permutation. Keep the old
Z representatives and set

```text
X'_i = P(Z_sigma(i)).
```

Each new representative is in the original X logical coset. Independent
array-based GF(2) checks establish canonical pairing, normalizer membership,
both translation actions, the trivial central logical action, and both
Hadamard directions. The representative-level Hadamard identities are exact;
translation identities are checked modulo stabilizers. Logical supports may
overlap.

The sum of the 32 logical support sizes decreases from 504 to 384. This is
not a circuit gate-count claim. Check matrices and code distance are unchanged;
the original exact distance certificate is linked by hash rather than
misrepresented as a newly computed distance proof.

- [Candidate with the new representatives](weight_twelve_handoff_v1/candidate.json).
- [Explicit logical supports](weight_twelve_handoff_v1/logical_supports.json).
- [Binary logical bases](weight_twelve_handoff_v1/logical_bases.npz).
- [Independent representative-change checks](weight_twelve_handoff_v1/report.json).

The handoff's `logical_weights_optimal=false` records the knowledge at export.
The subsequent complete coset enumeration below establishes optimality for
these fixed logical labels, not for every possible basis.

## Exact weight-eight investigation

For the original `Z_(0,0)` logical, the calculation enumerates all `2^24`
stabilizer dressings using a bounded-memory meet-in-the-middle array. Its
minimum is 12 (10 representatives attain it). Transitivity of the verified
logical translations transfers that minimum to all 16 Z labels; the physical
Hadamard fold transfers it to all 16 X labels.

Every zero-syndrome weight-eight support is then enumerated by matching
four-qubit syndromes. Its unique split into its smallest four and largest
four positions prevents duplicate counting and proves completeness. Every
returned word is checked for nontriviality and its logical class; array-based
GF(2) checks independently confirm all returned normalizers and class labels.
The fold bijectively transfers this complete Z catalog to X.

The saved distance-eight upper-bound witnesses each act on a product of ten
original logical labels. Code distance eight therefore never implied that
the individual original logicals had weight-eight representatives.

To exhaust all regular grids under the saved translations, identify the
logical module with `F2[C4 x C4]`. A cyclic seed generates the full module
exactly when its coefficient parity is odd. Of the 414 weight-eight classes,
192 are such units. For each, translate a physical weight-eight representative
around the 16-cell grid, calculate its uniquely determined canonical dual X
classes, and look those classes up in the complete X catalog. None passes.
The inverse-pairing calculation is independently checked with arrays.

- [Complete coset and weight-eight catalog report](weight_eight_v1/report.json).
- [All weight-eight physical words](weight_eight_v1/weight_eight_words.json).
- [All 192 canonical-grid tests](weight_eight_grid_v1/report.json).
- [Enumeration and grid-test implementation](../distance8_logical_search.py).
- [Independent small brute-force and full-code regressions](../test_distance8_logical_search.py).

## Existing distance-six alternative for resource-state studies

We already have **six saved distance-six presentations**, not necessarily six
inequivalent codes. The lowest-total-support saved option is
`d9173bb8cd878461c62a0dd5b7f9cb5f494b401aaf662cf40e45eb883551ebcf`.
Its independent audit was rerun during this investigation and again verified
exact distance six in both sectors, the full logical grid, and common-basis H.

| Property | Recommended distance-eight code | Existing distance-six option |
| --- | --- | --- |
| Parameters | `[[64,16,8]]` | `[[64,16,6]]` |
| Independent checks per Pauli type | `24` of weight `12` | `4` of weight `8`, `20` of weight `12` |
| Maximum check weight | `12` | `12` |
| Total independent check support | `576` | `544` |
| Logical translation action | Regular `C4 x C4` | Regular `C4 x C4` |
| Physical translation orders | `8`, `4` | `8`, `4` |
| Physical Hadamard fold order | `2` | `4` |

Both codes are connected. The distance-six option retains individual logical
Hadamards up to permutation even though its physical fold has order four.
Its fold squared acts trivially on logicals.

- [Distance-six candidate](../sparse_campaign/finalists_v1/best_independent/candidate.json).
- [Distance-six audit](../sparse_campaign/finalists_v1/best_independent/audit.json).
- [Distance-six explicit check supports](../sparse_campaign/finalists_v1/best_independent/supports.json).

The 32-support reduction is about 5.6%, but it does not by itself establish a
resource-state advantage. Preparation/verification circuits, fault propagation,
acceptance probabilities, and logical error rates have not been compared.

## Reproduction

Use the existing GALA virtual environment, `PYTHONDONTWRITEBYTECODE=1`,
`OPENBLAS_NUM_THREADS=1`, a writable `NUMBA_CACHE_DIR`, and fresh output paths:

```bash
python experiments/gala_n64_k16/distance8_logical_handoff.py --output /tmp/d8-logical-handoff-new
python experiments/gala_n64_k16/distance8_logical_search.py --output /tmp/d8-logical-catalog-new
python experiments/gala_n64_k16/distance8_logical_search.py \
  --grid-input /tmp/d8-logical-catalog-new --output /tmp/d8-logical-grid-new
python experiments/gala_n64_k16/distance8_weight10_search.py --output /tmp/d8-weight10-search-new
python experiments/gala_n64_k16/distance8_weight10_finalize.py \
  --source /tmp/d8-weight10-search-new --output /tmp/d8-weight10-handoff-new \
  --weight-eight-proof /tmp/d8-logical-grid-new/report.json
python experiments/gala_n64_k16/distance8_weight10_audit.py \
  --handoff /tmp/d8-weight10-handoff-new --output /tmp/d8-weight10-audit-new
```

The weight-ten finalizer links the earlier weight-eight obstruction, using
`--weight-eight-proof` for a fresh certificate or the saved repository
certificate by default. Existing outputs are never overwritten.

The logical-support search is complete; no additional fold or code-search
campaign is running. Alternative physical symmetries and state-specific uses
of weight-eight products remain separate follow-up questions.

Validation: **72 tests passed in 39.62 seconds** across the component,
check-weight, sparse-search, distance-eight partner, joint-fold, logical
handoff, and fixed-code logical-search suites. Small scalar enumerations
cross-check the coset enumeration methods; full-code regressions reproduce
the 1,016-word catalog, all 192 weight-eight basis tests, the complete
2,048-orbit array-based Hadamard filter, and the balanced weight-ten handoff.
