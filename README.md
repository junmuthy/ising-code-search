# GALA quantum-code search and STAR simulations

This repository develops and certifies quantum error-correcting codes with
logical symmetries suited to GALA/Ising operations.  It also contains syndrome
schedules, circuit-fault analyses, and transversal multi-rotation (TMR)
simulations.  The original paired-polynomial search targets the self-dual
minimal-GALA family

$$
Q_{4,m}(a)=CSS([A | A^T],[A | A^T])
$$

over `GF(2)[C_4 × C_m]`.  Its first search family is

$$
a(x,y)=1+y^r+xy^s+xy^{s+t}.
$$

The project uses the editable qLDPC checkout at
`/home/judah_unmuth/Documents/multistaq/qLDPC`, but keeps scripts, caches, and
results outside that repository.

## Latest certified results

The newest compact artifacts and result summaries are:

| Area | Latest result | Reproduction and data |
|---|---|---|
| `C4 x C4` logical-support search | A connected `[[64,16,8]]` CSS code with weight-12 checks and a canonical, Hadamard-compatible basis of 32 weight-ten logical Paulis; every physical qubit participates in exactly five logical supports. | [Code and basis](certified_css_grid_codes/n64_k16_d8/README.md), [portable data and exact verifier](certified_css_grid_codes/README.md) |
| `C4 x C4` distance-six alternative | A connected `[[64,16,6]]` CSS code with four weight-eight and twenty weight-twelve independent checks per type, total check support `544`, and the same logical-grid/Hadamard requirements. | [Code, certificates, and both check presentations](certified_css_grid_codes/n64_k16_d6/README.md) |
| `C4 x C8` high-rate search | A nonabelian `[[128,32,6]]` code with weight-12 checks (two 64-qubit components), plus a connected equivariant `[[128,32,7]]` control with weight-14/18 checks and weight-11 logicals. Both have same-basis logical Hadamard and regular translations. | [Nonabelian code](certified_css_grid_codes/n128_k32_d6/README.md), [connected control](certified_css_grid_codes/n128_k32_d7/README.md), [portable verifier](certified_css_grid_codes/verify.py) |
| `C4 x C8` automorphism-fold code | A connected `[[160,32,6]]` CSS code with weight-nine checks, exact `X/Z` distance six, a regular 32-site logical action, and four disjoint symplectic logical pairs. | [Overview](c4xc8_abelian_lp_160_32_6/README.md), [`code.py`](c4xc8_abelian_lp_160_32_6/code.py), [`verify.py`](c4xc8_abelian_lp_160_32_6/verify.py), and [`checks.npz`](c4xc8_abelian_lp_160_32_6/checks.npz) |
| `C4 x C7` automorphism-dual BB code | A `[[56,8,6]]` CSS code with weight-eight checks, a regular `C4 x C2` logical grid, and exact `X/Z` distance six. | [Overview](c4xc7_automorphism_dual_56_8_6/README.md), [`verify.py`](c4xc7_automorphism_dual_56_8_6/verify.py), and [`code.npz`](c4xc7_automorphism_dual_56_8_6/code.npz) |
| Coset-2BGA reference code | The published `[[36,4,6]]` code has a cyclic, pairwise-disjoint weight-seven logical basis; exhaustive enumeration proves the corresponding four-logical weight-six basis is impossible. | [Overview](n36_k4_d6_coset2bga/README.md), [`code.py`](n36_k4_d6_coset2bga/code_data/code.py), and [`presentation.json`](n36_k4_d6_coset2bga/code_data/presentation.json) |
| BB64 two-batch basis | The `[[64,8,8]]` BB code now has minimum-weight `X` and `Z` representatives partitioned into two internally disjoint `4+4` batches while retaining its regular `C4 x C2` action and Hadamard permutation. | [Search summary](bb64_simultaneous_basis_search/README.md), [complete representation](bb64_simultaneous_basis_search/BB64_TWO_BATCH_REPRESENTATION.md), and [`basis_batches2_candidate0.npz`](bb64_simultaneous_basis_search/fewer_batch_search/results/run_001_weight8_60s/basis_batches2_candidate0.npz) |
| BB56/BB64 postselection | Direct noisy ClifT measurements at `p=10^-3`, `M=3`, and `N=1,2,4,8` compare the paper-matching TMR-X rule with strict X/Z rejection at two angles. BB56 has higher acceptance at every matched point. | [Report and plot](postselection_comparison/BB56_BB64_ACCEPTANCE_VS_N.md), [CSV](postselection_comparison/BB56_BB64_ACCEPTANCE_VS_N.csv), and [`plot_acceptance_vs_n.py`](postselection_comparison/plot_acceptance_vs_n.py) |
| Distance-three TMR demonstration | A noiseless `tsim` implementation certifies the `[[9,1,3]]` rotated surface code and verifies that both ideal projection and explicit syndrome extraction produce the accepted logical rotation. | [Overview](surface_d3_rz_tsim/README.md) and [results](surface_d3_rz_tsim/RESULTS.md) |
| BB64 measurement decoder pilot | Across 1,000 histories per configuration, logical-class accuracy is `99.3%` in the hardest one-round `p=0.03` case and `100%` in the other eleven tested cases. | [Pilot CSV](bb64_hybrid_nonclifford/results/measurement_decoder_pilot_1000.csv); implementation is on branch `bb64-hybrid-nonclifford` |

Each linked result documents its certification boundary.  In particular,
"smallest found" statements are not global lower bounds, and the
distance-three TMR demonstration is a noiseless mechanism check rather than a
logical-fidelity estimate.

## Results index

The completed chain findings are summarized in [`RESULTS.md`](RESULTS.md).  The
packed 2D Ising-lattice search is summarized in
[`PACKED_RESULTS.md`](PACKED_RESULTS.md).

The scalable abelian packing search and its certified `[[896,128,7]]` code,
containing four disjoint `C_8 x C_4` logical grids with an explicit optimal
12-layer syndrome schedule, are summarized in
[`MANY_COPY_RESULTS.md`](MANY_COPY_RESULTS.md).

The nonabelian `S_3 x C_8 x C_4` search, including certified
`[[384,194,6]]` and `[[768,388,6]]` representatives and the logical ZX-pairing
limitation, is summarized in [`S3_RESULTS.md`](S3_RESULTS.md).

The candidate-specific enumeration of alternative ZX folds and the complete
`388 x 388` logical symplectic action of the compact code are reported in
[`S3_FOLD_ANALYSIS.md`](S3_FOLD_ANALYSIS.md).

The targeted one-checkerboard search and its self-dual `[[384,200,7]]` code
with 32 canonical, disjoint logical supports are reported in
[`S3_SINGLE_GRID_RESULTS.md`](S3_SINGLE_GRID_RESULTS.md).

The weight-12 active-GALA packing search and its certified
`[[1152,580,6]]` code with four jointly ZX-complete, disjoint `C_8 x C_4`
logical grids are reported in
[`S3_MANY_COPY_RESULTS.md`](S3_MANY_COPY_RESULTS.md).

The exhaustive joint-pairing revisit of all saved weight-16 `L=4` and `L=8`
S3 candidates is reported in
[`S3_REVISIT_RESULTS.md`](S3_REVISIT_RESULTS.md).

The faithful two-dimensional `SL(2,2) ~= S_3` relift, fast falsification of
the saved families, and the bounded `L=10`, `L=14`, and `L=16` resize pilots
are reported in
[`S3_LINEAR_FAST_FALSIFICATION.md`](S3_LINEAR_FAST_FALSIFICATION.md).

The dedicated native `L=6` search, including the exhaustive two-grid screen
and the `[[384,128,6]]` rank-56 ZX near miss, is reported in
[`S3_L6_NATIVE_RESULTS.md`](S3_L6_NATIVE_RESULTS.md).

The seed-first `[[256,32,>=6]]` half-checkerboard search, designed for two
256-qubit red/black blocks, is described in
[`HALF_GRID_SEARCH.md`](HALF_GRID_SEARCH.md).

The geometry-first `n=128` two-batch half-checkerboard search and its exact
faithful-`GL(2,2)` ZX-pairing obstruction are reported in
[`BATCHED_HALF_GRID_RESULTS.md`](BATCHED_HALF_GRID_RESULTS.md).

The isolated natural-`S3`, `n=192` reverse-geometry search—including positive
weight-seven logical geometry, the invariant-triplet diagnosis, and the
bounded weight-at-most-12 generator results—is reported in
[`reverse_geometry/RESULTS.md`](reverse_geometry/RESULTS.md).

The compact `GL(2,2) x C_4`, `L=4`, `J=2` search for a single-row
`[[32,4,6]]` code, including its disjoint logical seed construction and exact
and sampled negative results, is documented in
[`SINGLE_ROW_32_SEARCH.md`](SINGLE_ROW_32_SEARCH.md).

The compact faithful-permutation `D4 x C4`, `L=2`, `J=1`, `n=32` search,
including identity and nonidentity ZX folds, checkpointed exact constraint
spaces, and the persistent distance-two obstruction, is documented in
[`d4_single_row/RESULTS.md`](d4_single_row/RESULTS.md).

The exhaustive paper-faithful abelian `C8`, `C2 x C4`, and `C4 x C4`
single-row searches, including the exact rank and disjoint-ZX-orbit
obstructions, are documented in
[`ABELIAN_SINGLE_ROW_RESULTS.md`](ABELIAN_SINGLE_ROW_RESULTS.md).

The floating-dimension revisit of the self-dual `C4 x C4` BB family, allowing
every `k >= 4` while requiring only one protected four-logical-qubit row, is
reported in
[`c4xc4_floating/RESULTS.md`](c4xc4_floating/RESULTS.md).

The exhaustive one-block `C28 ~= C4 x C7` search for a half-sized
`[[28,4,>=6]]` bicycle-chain analogue, including its positive logical geometry
and exact low-distance obstruction, is documented in
[`c28_one_block/RESULTS.md`](c28_one_block/RESULTS.md).

The broader direct `C4`-invariant `[[28,4,*]]` search, which breaks physical
`C7` translation while retaining the disjoint logical row, exact transversal
Hadamard, and weight-at-most-12 checks, is reported in
[`c4_invariant_lagrangian/RESULTS.md`](c4_invariant_lagrangian/RESULTS.md).

The expanded `n=32` search with one added physical `C4` orbit—including
coupled extensions, all rank-14 module types, directed refinement, and the
best low-weight `[[32,4,5]]` representative—is reported in
[`c4_invariant_n32/RESULTS.md`](c4_invariant_n32/RESULTS.md).

The exhaustive self-dual `C2 x C7` BB search for a `[[28,4,6]]` code with two
independent logical `C2` cycles, covering check weights eight and twelve, is
reported in [`c2xc7_bb/RESULTS.md`](c2xc7_bb/RESULTS.md).

The `C4 x C5`, `n=40` single-row search—including the exact self-dual scan,
the universal BB fold's rank-zero mechanism, and the exhaustive sparse
half-preserving automorphism-dual scan—is documented in
[`c4xc5_search/RESULTS.md`](c4xc5_search/RESULTS.md).

The certified `[[224,32,7]]` code over `C28 x C4` and `[[56,8,7]]` code over
`C7 x C4`, both with disjoint order-seven fibres and exact transversal
Hadamard, are reported in
[`ORDER_SEVEN_FIBRE_RESULTS.md`](ORDER_SEVEN_FIBRE_RESULTS.md).

The relaxed-batching `C4 x C2` half-grid search—including strict identity and
permutation GALA folds, the `L=6` binomial obstruction, compact natural-`S3`
distance-six codes, the saved `[[48,16,3]]` four-batch grid seed, and its exact
radius-two neighborhood, plus the independent-`F/G` half-swap pilots and
their exact logical-action obstruction, and the x-reflected fold that restores
the full translation algebra but exposes a distance-two/three bottleneck, plus
the 6,269-candidate witness-guided beam search that reaches a common
`[[48,16,3]]` plateau, and the 34,957-candidate coordinated beam that identifies
the invariant natural-`S3` three-point logical-fiber obstruction, together
with the faithful `GL(2,2)`, `n=32` follow-up and its exact-commuting
centralizer diagnosis, the `n=64/96` faithful protograph probes, and the
reconstruction of a published self-dual `[[64,8,8]]` BB code satisfying the
regular `C4 x C2` grid, ZX pairing, and four raw disjoint STAR batches—is
documented in
[`batched_c4xc2_search/RESULTS.md`](batched_c4xc2_search/RESULTS.md).

An executable walkthrough of the recommended packed code and its Ising/STAR
acceptance tests is in [`ising_conditions.ipynb`](ising_conditions.ipynb).

Execute and save all notebook outputs with:

```bash
.venv/bin/jupyter execute --inplace --timeout=300 ising_conditions.ipynb
```

## Environment

The local environment has already been created in `.venv` and qLDPC installed
with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e /home/judah_unmuth/Documents/multistaq/qLDPC
```

Set writable cache directories when running several workers:

```bash
export NUMBA_CACHE_DIR="$PWD/results/numba-cache"
export MPLCONFIGDIR="$PWD/results/matplotlib-cache"
```

## Search

Run the algebraic filters and 20 randomized BP+OSD distance trials:

```bash
.venv/bin/python run_search.py --m 7 11 13 --trials 20
.venv/bin/python collect_results.py
```

The search quotients torus translations and independent reflections of the two
cyclic axes.  Pass `--no-symmetry-quotient` to retain every distinct normalized
polynomial support.

The randomized distance result is always recorded as an upper bound.  The known
weight-`m` column logicals are also applied as an analytic upper bound.

## Exact distance

Certify the known bicycle-chain seed:

```bash
.venv/bin/python certify_distance.py --m 7 --r 3 --s 2 --t 2
```

Or certify every promising `m=7` representative from a search result:

```bash
.venv/bin/python certify_results.py --m 7 --min-bound 6 --workers 2
```

To rule out candidates above a target without spending eight ILPs on each one,
use threshold mode.  Every returned `ilp_distance_upper_bound` is backed by an
explicit logical witness, while `certified_distance` remains empty unless all
eight logical constraints were solved:

```bash
.venv/bin/python certify_results.py \
  --input results/weight8-m13-2000.jsonl \
  --m 13 --min-bound 9 --stop-at 8 --workers 4
```

The certification solves eight binary minimum-weight ILPs.  If `L` is the known
orthonormal logical-support matrix, a zero-syndrome vector `e` is nontrivial iff
at least one component of `L @ e` is one.  The distance is consequently the
minimum of the eight problems

```text
min |e|  subject to  H @ e = 0 and L[i] @ e = 1.
```

Only a complete run with every solver status equal to `optimal` is merged into
the report as `certified_distance`.

## Packed `C_8 x C_4` logical lattice

Search the two-pair weight-eight ansatz over `C_56 x C_4`:

```bash
.venv/bin/python run_packed_search.py --family weight8 --m 7 --workers 8
```

This search is exhaustive modulo translations and axis reflections.  The
two-pair family has an unavoidable disconnected-quotient obstruction.  Search
the minimal connected three-pair family, which has weight-12 checks, with:

```bash
.venv/bin/python run_packed_search.py --family weight12 --m 7 --workers 8
```

Certify selected accepted codes against `d >= 6`:

```bash
.venv/bin/python certify_packed.py \
  --input results/packed-weight12.jsonl \
  --output results/packed-weight12-certifications.jsonl \
  --family packed_weight12 --m 7 --target-distance 6 --workers 8
```

The 32 fibre logicals in either physical half are related by exact physical
translations.  Distance certification therefore requires one ILP from each
half rather than 64 redundant translated problems.

## `[[32,4,6]]` Clifford circuit faults

The saved all-weight-eight presentation and simultaneous 12-layer syndrome
schedule now have executable Stim memory circuits and exact low-cardinality
fault searches in `n32_k4_d6_reference_code/stim_fault_distance/`.

Validate the noiseless detectors and noisy detector error models with:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.stim_fault_distance.run_validation
```

Run the exact central-cycle search with periodic checkpoints using:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.stim_fault_distance.run_fault_distance \
  --experiment bulk --basis both --fault-model full --max-faults 6 \
  --heartbeat-seconds 30 --output results/n32-k4-d6-stim-clifford-v1/bulk.json
```

The original depth-12 ordering has exact circuit fault distance four in both
bases.  Searching the other valid ZX folds found schedule `a381a067d3e82750`,
which has exact full-noise circuit fault distance five in both bases for both
three- and eighteen-round memories.  The successful schedule remains
weight-eight, depth twelve, and layerwise `C_4` invariant.  See the local
`stim_fault_distance/RESULTS.md` for the original diagnosis and
`schedule_fault_search/RESULTS.md` for the successful search and certificates.

## Minimum two-spin `C_2` search

The direct search in `inverse_c2_minimum/` targets a `[[n,2,6]]` code with
`n <= 16`, two disjoint logical supports, physical `C_2` exchange symmetry,
and permutation ZX duality. It proves an exact `n = 12` pairing obstruction
and searches the complete symmetry-reduced fold catalogs at `n = 14` and
`n = 16` with checkpointed low-weight logical cuts. The current `n = 16`
frontier is a qLDPC-validated `[[16,2,4]]` code. One- and two-orbit local
refinement evaluated 46,097 exact connected neighbors without finding
distance five or six; this is strong negative evidence, not a no-go proof.

Run the first weight-eight `n = 14` tranche with:

```bash
.venv/bin/python -m inverse_c2_minimum.run_search \
  --output-root results/inverse-c2-minimum --run-name n14-w8-pilot \
  --length 14 --maximum-check-weight 8 --maximum-tasks 20 \
  --checkpoint-every-models 5 --stop-on-hit
```

## Tests

```bash
.venv/bin/python -m pytest
```

The regression suite constructs the `[[56,8,6]]` chain seed, the packed
`[[448,64,7]]` code, and the four-grid `[[896,128,7]]` code. It validates
their canonical logical supports and physical automorphisms and certifies
their exact distances with HiGHS.

## Four-grid abelian packing

Run the complete symmetry-reduced weight-12 sweep over `C_112 x C_4`:

```bash
.venv/bin/python run_many_copy_search.py --copies-per-half 2 --m 7 --workers 8
```

Reproduce the recommended code's exact distance, automorphisms, and optimal
12-layer syndrome schedule:

```bash
.venv/bin/python certify_many_copy.py
```

## Nonabelian `S_3` search

Run the uniform-degree-four polynomial pilots with:

```bash
.venv/bin/python run_s3_ising_search.py --family l4-w16 --accepted-candidates 100
.venv/bin/python run_s3_ising_search.py \
  --family l8-w16 --accepted-candidates 100 --seed 240827
```

The specialized Tanner search exhaustively checks weights one through six up
to bottom translation, then separately certifies disjoint translated logical
grids and their combined ZX pairing.

Reproduce the selected four-grid, weight-12 nonabelian certificate with:

```bash
.venv/bin/python certify_s3_many_copy.py
```
