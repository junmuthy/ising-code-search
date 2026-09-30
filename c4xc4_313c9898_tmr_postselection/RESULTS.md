# First milestone: one and four resource states

Completed 2026-09-14. This study is restricted to the `313c9898` code's
logical `0` and disjoint logical set `{0,2,4,6}`. It includes no N=8 or N=16
resource-preparation run and does not alter the frozen code bundle.

## Code, check weights, and schedule depth

The code is `[[64,16,6]]`, with weight-six canonical logical X and Z
representatives. The preferred schedule `30f5d7baeecc8a92` measures
**24 X checks and 24 Z checks, all weight 12**. It uses 48 dedicated
ancillas, for 112 allocated qubits including the data.

- One full X/Z syndrome round: **576 CNOTs in 12 CNOT layers**.
- Full one- or four-resource preparation: **26 CNOT layers**: 12 for
  Z-only initialization, 2 for the TMR gadgets, and 12 for final X/Z checks.
- One additional physical rotation layer is required. Reset/readout
  durations and hardware routing are not included in these layer counts.
- Total CNOTs per preparation attempt: **870 for one resource**, or
  **888 for four resources**.

The redundant parent measures 32 weight-12 checks per sector, using
768 CNOTs in 12 layers per full syndrome round. A separate lower-weight
basis has 16 weight-8 and 8 weight-12 checks per sector, totaling 448
incidences. That lower-weight basis is **not** used in this study or in
the saved preferred depth-12 circuit. See the
[frozen code description](../c4xc4_64_16_6_313c9898/README.md).

## Main result

The preferred 24-check-per-sector schedule has better measured acceptance
and lower gate cost than the redundant parent in this one-round preparation
comparison. This is not a comparison of their conditional logical fidelity.

At physical error probability `p=0.001`, using three-pair (`M=3`) injections:

| Logical angle | One resource, strict XZ | Four resources, strict XZ | One resource, X-only | Four resources, X-only |
|---|---:|---:|---:|---:|
| `0` (control) | 17.933% | 18.138% | 29.876% | 29.899% |
| `0.001` | 17.568% | 16.813% | 29.254% | 27.806% |
| `pi/128` | 15.387% | 9.688% | 25.581% | 16.038% |
| `pi/32` | 12.302% | 4.051% | 20.471% | 6.685% |

Each row/presentation/N combination has **1,000,000 fixed attempts**. X-only
and strict-XZ estimates come from the same shots, not independent samples.
For the preferred four-resource pi/32 point, the strict 95% Wilson interval
is **[4.013%, 4.090%]**; the X-only interval is **[6.637%, 6.735%]**.

Here, **X-only means `tmr-x`**: reject nonzero terminal X syndromes but
permit nonzero terminal Z syndromes. Strict XZ (`strict-xz`) rejects either.
The four-resource acceptance is for the **whole four-resource block**;
all four candidates are accepted or discarded together. It is not a
per-logical probability and does not describe independent retries.

At logical angle `0.001`, strict acceptance for four is only 4.3% lower
relatively than for one. Producing four candidates on success therefore
gives **3.83 times as many accepted candidates per attempt**, assuming all
four are useful. Small-angle ideal rejection is mild (98.134% ideal
acceptance for one, 92.741% for four), while preparation and syndrome
extraction impose a largely shared rejection cost. Accepted-state noisy
fidelity remains a separate, unmeasured quantity.

At pi/32 under strict XZ, four-resource preparation yields 0.162048 candidate
resources per attempt versus 0.123020 for one-resource preparation: a 31.7%
increase. The corresponding CNOT costs are 5,479.9 versus 7,072.0 per
candidate resource, a 22.5% reduction. A full four-resource block takes
24.684 attempts on average, versus 8.129 attempts for a single resource.
These statements assume all four resources are useful; they do not count
the twelve idle logicals as generated resources.

The redundant parent at the same pi/32 point accepts 11.843% (N=1) and
3.858% (N=4) under strict XZ. Its four-resource CNOT cost is 7,620.7 per
candidate, versus 5,479.9 for the preferred presentation.

## Why acceptance is still limited

For N=4, the ideal pi/32 acceptance is 22.295%, but the preferred circuit's
zero-angle strict acceptance is only 18.138%. Their product predicts about
4.04%, consistent with the measured 4.051%. Across this sweep, acceptance
divided by ideal acceptance and the matching zero-angle control ranges from
0.9944 to 1.0049 for the two reported policies. Thus an approximately
angle-independent noise-survival factor is a useful empirical description
here; it is not an exact theorem and no BB56 fit constants were imported.

The preferred circuit has 870 CNOTs for N=1 or 888 for N=4, with 26 CNOT
layers and 112 allocated qubits. The parent has 1,158 or 1,176 CNOTs,
26 CNOT layers, and 128 qubits. Both add one rotation layer; reset/readout
durations and hardware routing are outside this layer count.

For context, the prior BB56 strict-XZ N=4 pi/32 estimate was 6.62% from
20,000 shots, with 704 CNOTs and depth 20. The new code's lighter logical
rotations therefore do not yet compensate for its larger syndrome-extraction
overhead at this active-set size. This is an acceptance/cost comparison,
not a comparison of accepted-state fidelity or use of all encoded capacity.
See the existing
[BB56 report](../bicycle_chain_l4_paper/tmr_postselection/BB56_POSTSELECTION_SUITE.md).

## Validation and reproducibility

- 17 regression tests pass, including independent Stim/ClifT noisy
  zero-angle checks, wrong-sign negative controls, frozen hashes, schedule
  ordering, initialization feedback, and resumable-run guards.
- Exact enumeration validates both active sets and schedules at every
  sweep angle, identifying the output on all sixteen logicals. The maximum
  normalized amplitude discrepancy is `2.22e-16`.
- 32 noiseless simulation cases, each with 10,000 attempts, check both
  direct projection and physical initialization/extraction. After ideal
  inverse rotations, all sixteen logical X outcomes are correct on all
  248,566 accepted validation trajectories.
- A separate 160,000-shot pilot precedes the 16,000,000-shot production
  suite. Production uses distinct seeds and four sampler threads.
- Production sampling used approximately 23.13 seconds on the current host.
  Active width is 3 for N=1 and 12 for N=4 at nonzero angles. This is
  simulator runtime, not physical preparation time.

The full table, confidence intervals, and costs are in
[the generated report](reports/production/RESULTS.md) and
[the CSV](reports/production/acceptance.csv). The source-hashed configurations,
counts, syndrome histograms, emitted circuits, and exact validation record
are retained under `results/`. See [README.md](README.md) for commands and
the precise historical noise convention.

This study estimates **candidate-state acceptance only**. It does not
establish decoded conditional fidelity, justify every X-only retained
branch, or transfer the saved memory fault-distance bound to preparation.
No runs remain active. No commit or branch change was made.
