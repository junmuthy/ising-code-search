# Sparse k=16 search: certified weight-12 improvement

Date: 2026-09-11. The first bounded milestone is complete. **Eight new saved
presentations give connected codes with weight-12 checks, the full regular
logical `C4 x C4` grid, and common-basis Hadamard. Six have parameters
`[[64,16,6]]`; two have the stronger parameters `[[64,16,8]]`.** They are not claimed
to be eight inequivalent codes or literature-novel constructions.

No code with a complete weight-at-most-10 presentation passed the bounded
pilots. This is not a no-go theorem. No search process is left running.

The [distance-eight lower-weight follow-up](../distance8_weight_search/README.md)
has now completed exact partner slices, all 680 specified local mutations,
and all 384 fixed-coefficient/fold combinations, plus a bounded shape
extension. None supplies weight-at-most-10 checks while preserving distance
eight and the required logical action. Weight 12 remains the best found;
the exclusions are scoped to those explicit families.

The subsequent [joint coefficient/fold campaign](../joint_fold_campaign/README.md)
has completed the 255 remaining configuration classes, including distance
six, with no lower-check-weight hit. Code searching has stopped. See the
[fixed-code study](../distance8_code_study/README.md) for weight-12 representatives
of all 32 distance-eight logicals, exact weight-eight basis restrictions, and
the distance-six candidate comparison for resource-state investigations.

A subsequent exact search now gives [all-weight-10 logicals](../distance8_code_study/README.md)
for the recommended distance-eight code, using a new canonical logical basis
with the same translations and physical Hadamard fold. Weight 10 is optimal
under that logical-gate contract; the code and its weight-12 checks are unchanged.

## Recommended distance-eight alternative

The final distance audit found two hits with **exactly `d_X=d_Z=8`**, not just
the requested distance at least six. I recommend starting with candidate
`71e1114cdaa26f7c2bc3cbd2098452d03aee37bfcc62016f59367dc784c7625e`, which also
has an order-two physical Hadamard fold.

| Property | Result |
| --- | --- |
| Parameters | `[[64,16,8]]`, connected |
| Translation-closed checks | `32` weight-`12` checks per Pauli type |
| Optimal independent checks | `24` weight-`12` checks per Pauli type |
| Independent total support | `576`, versus `600` for the previous optimized distance-six code |
| Independent peak combined degree | `11`; peak-degree optimality not proved |
| Logical translations | Regular `C4 x C4` grid in one canonical basis |
| Physical translation orders | `8` and `4`, commuting |
| Logical Hadamard permutation | `(i,j) -> (i,2i+1-j)` modulo four |
| Physical Hadamard fold | Order `2`; the paired independent X/Z checks exchange directly in both directions |

In the same two-sheet `C8 x C4` convention used below, this code has
`a=2164531200`, `b=23815`, `HX=[L_a|L_b]`, `HZ=P(HX)`, and

```text
P(s,i,j) = (1-s, i+4j mod 8, 2i+3j mod 4).
```

Exhaustive stabilizer enumeration finds no nonzero stabilizer below weight
12, so its maximum weight and independent total support are exactly optimal
for this code. The smallest translation-closed presentation under the saved
translations uses 32 weight-12 checks per type. Saved Z logicals have weight
12 and X logicals weights 16–24; injection supports remain unoptimized.

Those were the original saved representatives. The new
[same-label handoff](../distance8_code_study/weight_twelve_handoff_v1/candidate.json)
has all X and Z representatives of weight 12. Exact coset enumeration proves
this minimum for those labels; alternative all-weight-eight canonical grids
are excluded for the saved translations, as detailed in the fixed-code study.

The initial independent audits excluded weights through six. A subsequent
complete sorted triple/quadruple syndrome match excludes weight seven, with
direct stabilizer-membership tests. Explicit verified weight-eight witnesses
for the recommended code are

```text
Z: [10,16,17,20,27,29,53,62]
X: [21,28,32,35,38,41,45,58]
```

The other distance-eight presentation is
`1ceb6dd0cf189fb3a5e0fcc9d52f1f9020a6ab8f2dd560d298fc3a46244b92bc`.
Both have exact exclusions through seven and checked weight-eight logicals in
both sectors. The solver supplied upper-bound witnesses; the lower bound is
exact combinatorial enumeration, not a solver timeout or decoder estimate.

- [Full distance-eight audit](distance8_handoff_v1/full_audit.json).
- [Independent candidate](distance8_handoff_v1/independent/candidate.json),
  [binary checks](distance8_handoff_v1/independent/checks.npz),
  [explicit supports](distance8_handoff_v1/independent/supports.json).
- [Translation-closed candidate](distance8_handoff_v1/symmetric/candidate.json),
  [binary checks](distance8_handoff_v1/symmetric/checks.npz).
- [Weight, basis-change, and load report](distance8_handoff_v1/report.json).
- [Supplemental distance certificates](distance_upgrades_v1/), retained separately
  from the original lower-bound-only pilot audits.

The earlier `finalists_v1` ranking minimized check weight and then total
support among codes already meeting distance six. It therefore selected the
544-support code below. The distance-eight handoff is a separate, stronger
distance option at a modest additional support cost, not an overwrite of that
ranking or the earlier certificates.

## Lower-total-support distance-six alternative

Candidate `d9173bb8cd878461c62a0dd5b7f9cb5f494b401aaf662cf40e45eb883551ebcf`
has two useful presentations:

| Property | Translation-closed presentation | Minimum-total-weight independent presentation |
| --- | --- | --- |
| Parameters | `[[64,16,6]]` | Same code |
| Checks per Pauli type | `32` of weight `12` | `4` of weight `8`, `20` of weight `12` |
| Total checks | `64` | `48` |
| Total support, both Pauli types | `768` | `544` |
| Maximum check weight | `12` | `12` |
| Maximum combined qubit degree | `12` | `11` |
| Maximum per-type qubit degree | `8` | `6` |
| Logical grid and Hadamard | Certified | Same logical basis and gates |
| Translations permute the displayed checks | Yes | Generally map checks to products |

The previous code required maximum weight 14, with total support 896 in its
translation-closed presentation or 600 in its optimized independent basis.
The corresponding support reductions are **14.3%** and **9.3%**. In particular,
this is a different code, not another generator choice for the previous code:
their exact minimum achievable maximum check weights differ, which excludes
equivalence by qubit permutation and stabilizer-generator changes.

With logical labels `(i,j)` in `Z4 x Z4`, the verified operations are

```text
Tx: (i,j) -> (i+1,j)
Ty: (i,j) -> (i,j+1)
P H^64: individual logical Hadamards, followed by (i,j) -> (-i,2-j)
```

The physical translations commute and have orders eight and four. Their
logical actions have orders four and four. The fold P has order four, with
`P^2=Tx^4` acting trivially on logicals. These are the same physical-versus-logical
order conventions as in the earlier search; physically order-four generators
for both directions were not required or established.

Saved Z logical representatives all have weight nine; X representatives have
weights 9, 13, or 19. They need not be disjoint and were not optimized for STAR
injection. Gate compatibility is checked in this one canonical basis.

## Exact check-weight optimization

Every new hit was exhaustively enumerated over all `2^24` stabilizers per Pauli
type. Six have a minimum independent basis of 24 weight-12 checks per type.
Two, including the selected code, admit four weight-8 and twenty weight-12
checks per type, totaling 272 per type.

For the selected code there are exactly four weight-8 stabilizers, no other
nonzero stabilizers below weight 12, and 152 weight-12 stabilizers, in each
sector. The rank filtration is therefore

```text
weight <= 7:  rank 0
weight <= 11: rank 4
weight <= 12: rank 24
```

This proves that maximum weight 12 and total support 544 are optimal for this
code's CSS generators. It does not bound other codes. Randomized selection
among optimal bases reduced peak qubit degree; peak-degree optimality is not
proved.

The 156 stabilizers of weight at most 12 split into ten orbits under the saved
physical translations (and the fold square, which is already a translation).
All 1,023 nonempty orbit subsets were tested in each sector. A spanning,
translation-closed set needs total support at least 384 per type, attained by
32 weight-12 checks. Thus the two exported presentations expose the symmetry
versus total-support tradeoff explicitly.

## Why the successful branch differs

The successful family uses the **abelian** group `C8 x C4`, with two regular
32-coordinate data blocks. It is a fold-constrained group-algebra CSS
construction, not the usual square commuting-matrix GALA parent.

For group elements `(i,j)` with `i mod 8`, `j mod 4`, flattened as `4*i+j`, let
`L_a,L_b` be actual binary left-regular lifts. Define

```text
HX = [L_a | L_b]
HZ = P(HX)
P(0,i,j) = (1,3i,-j)
P(1,i,j) = (0,3i+4,-j)
```

The selected coefficient supports are

```text
a = 1319523:
    (0,0), (0,1), (1,1), (1,2), (2,1), (3,1), (4,2), (5,0)
b = 67596:
    (0,2), (0,3), (2,3), (4,0)
```

Although P fixes the ZX relationship by construction, CSS commutation is still
enforced explicitly. For a fixed a and this block-swapping affine fold, its
conditions are linear in b. The search solves that full kernel, samples sparse
partners, then checks ranks, logical relations, regular-grid rank, distance,
and both Hadamard directions.

The key diagnostic was reconstructing the old weight-14 code in these abelian
coordinates. Replacing its actual Z space by the conventional `[L_b^T|L_a^T]`
gave a different code that lost the regular logical grid. Consequently the
earlier standard abelian pilot was not a positive-control-containing family.
The new folded family does contain the old code exactly and supplied the new
hits. Nonabelian ancestry informed the seed and fold; the successful branch
itself must not be labeled a nonabelian GALA hit.

The fold catalog contains 96 swapping affine folds built from involutive
automorphisms of `C8 x C4`, with fold square either identity or the designated
central translation. Half the fixed inputs use the known-code fold; the rest
cycle through the catalog. Inputs combine random sparse/factor-product seeds
with bounded support deletions and swaps around the known coefficient a.

## Archive audit

All 93 distance-six parent records were revisited: 61 had passed the old
parent-level common-H test, and 32 had not. Ninety-two split into two 64-qubit
components. An explicit right translation identifies the two components in
each pair, so only one representative was used. There were no duplicate X/Z
row-space pairs in the chosen coordinates; this is not a full equivalence
classification.

Among the 92 components:

- 51 have no regular logical `C4 x C4` grid in the tested component-preserving
  right action. Every distinct induced order-four action was considered.
- 39 have a grid but no certified common Hadamard in the bounded inherited-fold
  catalog, including component-exchanging translations used to repair folds.
- Two pass, including the previously certified weight-14 code. The other has
  minimum maximum check weight 16, so neither improves the baseline.

All 33 weight-12 parents were included. Twenty components failed the right-action
grid test; thirteen failed the inherited-fold Hadamard test. Their exact
minimum maximum check weight is 12. A negative result here does not exclude
other physical automorphisms or other Hadamard folds.

For each physical fold tested, the common-basis routine exhausts all 32,768
units of `F2[C4 x C4]` when the first basis does not work. A first-row pairing
condition is only a filter; both full logical action matrices must be the same
permutation matrix before acceptance. The old positive certificate is recovered
as a regression test.

## Bounded pilot results

Counts below are journaled, within-run deduplicated evaluations, not globally
inequivalent codes. Ordinary pilots deduplicate fixed-coordinate X/Z spaces;
the fold-constrained pilots deduplicate code/fold pairs. Counts can overlap
between runs and must not be interpreted as exhaustive coverage.

| Run | Check ceiling | Evaluated | Regular grid | Distance at least six | Full hits |
| --- | --- | --- | --- | --- | --- |
| Nonabelian extensions, first pilot | `8` | `14542` | `1038` | `0` | `0` |
| Nonabelian extensions, second pilot | `10` | `53213` | `3728` | `0` | `0` |
| Nonabelian extensions, interim tier | `12` | `70046` | `3273` | `2` | `0` |
| Standard abelian, first pilot | `8` | `23172` | `0` | `0` | `0` |
| Standard abelian, second pilot | `10` | `81071` | `0` | `0` | `0` |
| Faithful matrix lift, first pilot | `8` | `4275` | `0` | `0` | `0` |
| Faithful matrix lift, second pilot | `10` | `20457` | `0` | `0` | `0` |
| Orthogonal equivariant control | `10` | `103` | `103` | `0` | `0` |
| Orthogonal equivariant control | `12` | `98` | `98` | `0` | `0` |
| Fold-constrained abelian | `10` | `50892` | `2083` | `0` | `0` |
| Fold-constrained abelian | `12` | `72756` | `2959` | `206` | `8` |

The pilots total 390,625 journaled evaluations before cross-run deduplication.
The orthogonal-control runs each generated 4,096 recipes, but most exceeded
the check ceiling and others repeated a row space. The main pilots use 512 or
1,024 fixed inputs and at most 96 or 128 partners per input; configured time
caps and exact seeds are in each run's `summary.json`. All completed their
configured fixed-input ranges before their time caps.

The nonabelian branch tests four order-32 presentations
`z^2=1, z central, [x,y]=z, x^4=z^alpha, y^4=z^beta`, with binary alpha and beta.
The abelian standard branch tests `C8 x C4` and `C4 x C4 x C2`. The faithful
branch uses full matrix commutants in `M2(F2[C4 x C4])`; sampling those commutants
remains heuristic. The orthogonal control allows nonfree check orbits on four
16-site data sheets and supplies exact grid/H compatibility by construction.

Sparse candidates with the required grid were most often rejected by explicit
weight-two, weight-three, or weight-four logical witnesses. No weight-at-most-10
pilot produced a candidate passing the distance threshold. These outcomes
justify a structural change or more targeted sampling, not an impossibility
claim or an unlimited rerun of the same samples.

## Independent certification and artifacts

All eight new hits were independently rechecked with array-based GF(2) algebra,
not just the construction's packed-integer routines. Audits verify ranks
24 and 24, CSS commutation, the full canonical logical pairing, every logical
image under both translations and Hadamard, and a single row-space component.

Exact support enumeration excludes nonzero zero-syndrome errors through weight
five. For the lower-support distance-six code, the weight-six logical witnesses are

```text
Z: [0,2,32,34,36,38]
X: [0,2,12,14,48,50]
```

Hence both distances are exactly six. Independent HiGHS MILPs also returned
infeasible at weight at most five in both sectors for the selected optimized
presentation. The exact enumerations and checked witnesses, not solver statuses,
are the distance certificates. The stronger distance-eight certificates and
their additional weight-seven exclusion are linked in the first section.

- [Best independent presentation](finalists_v1/best_independent/candidate.json),
  [binary checks](finalists_v1/best_independent/checks.npz),
  [explicit supports](finalists_v1/best_independent/supports.json),
  [independent audit](finalists_v1/best_independent/audit.json).
- [Best translation-closed presentation](finalists_v1/best_symmetric/candidate.json),
  [binary checks](finalists_v1/best_symmetric/checks.npz).
- [Final ranking and check-space certificates](finalists_v1/report.json):
  both generator-conversion directions, stabilizer action matrices, load
  balancing, and exact symmetry-orbit optimization. Separate weight files
  contain complete histograms and all low-weight stabilizers for every hit.
- [Archive summary](archive_v1/summary.json) and [component journal](archive_v1/components.jsonl).
- [Successful pilot summary](twisted_w12_v1/summary.json); its `certified/`
  directory retains all eight original hits and their logical bases.

All coordinates are zero-based; integer row bit q labels qubit q. All
permutations map old coordinate to new coordinate. Generator/action bit j
denotes a product including generator j. Logical bases and physical permutations
are exported alongside each selected check presentation.

## Reproduction

From the workspace root, choose output paths that do not already exist:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/gala-sparse-numba.XXXXXX)"
/home/judah_unmuth/gala-code-search/.venv/bin/python \
  experiments/gala_n64_k16/sparse_archive.py --output /tmp/gala-sparse-archive-new
/home/judah_unmuth/gala-code-search/.venv/bin/python \
  experiments/gala_n64_k16/sparse_twisted.py --maxweight 12 \
  --fixed 1024 --per-fixed 128 --seconds 180 --seed 20260924 \
  --output /tmp/gala-sparse-twisted-new
/home/judah_unmuth/gala-code-search/.venv/bin/python -m pytest \
  -o addopts='' experiments/gala_n64_k16/test_sparse_search.py \
  experiments/gala_n64_k16/test_component.py \
  experiments/gala_n64_k16/test_check_weight_reduction.py -q -p no:cacheprovider
```

`sparse_pilot.py` runs the other four branches; the exact parameters are saved
in each run summary. `sparse_finalize.py` performs exhaustive check-weight
analysis and exports the selected presentations. NumPy with `bitwise_count`
is required. Source hashes are saved with runs/certificates. The external
GALA checkout and all prior certificates were left unchanged. The final combined
regression suite passes **32 tests**, including both distance-six and
distance-eight handoffs, scalar reference tests for the weight-seven matcher,
all eight new candidate reconstructions, and the previous code's tests.

## Next decision point

This milestone improves maximum check weight from 14 to 12 and supplies a
distance-eight option, but does not reach the preferred weight-eight or
weight-ten check tiers. The distance-eight code is now my preferred practical
comparison point, alongside the slightly lower-support distance-six code.
For a continuation, prioritize
the now-positive fold-constrained family: targeted sparse-partner enumeration
and witness-guided coupled updates around its distance-six hits, plus broader
row-orbit/protograph structures if the same short-logical mechanism persists.
The current random sparse-kernel sampler is not an exhaustive low-weight search.

No n=66/68 mixed-orbit campaign, full physical-code equivalence classification,
or broader physical-automorphism search was performed. Those remain follow-ups,
not excluded possibilities. This run did not add qubits or gauges to the old
code. Ancilla overhead, extraction schedules, circuit fault distance, noisy
decoding, and STAR injection performance remain untested; lower static check
weights alone do not certify improvements in those quantities.
