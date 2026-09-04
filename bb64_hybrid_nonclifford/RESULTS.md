# BB64 hybrid non-Clifford implementation results

These are milestone results for the decoder and repair machinery.  They are
not yet an end-to-end logical-error estimate for fully decoded circuit noise.

## Exact and noiseless validation

The model reconstructs all `65,536` ideal syndrome classes from the committed
recovery archive.  Exhaustive unit tests verify class-coordinate round trips,
the linear quotient map, alternative-mask recovery, and rejection of displayed
syndromes outside the ideal image.

At `theta=pi/32`, direct BB64 non-Clifford circuits were conditioned on two
representative raw branches, canonically Pauli-corrected, repaired where
needed, and canceled by the ideal target inverse:

| Branch labels | Alternatives | Attempts | Survivors | Measured probability | Exact probability | Final/logical errors |
|---|---:|---:|---:|---:|---:|---:|
| `00000000` | 0 | 20,000 | 1,003 | `0.050150 ± 0.001543` | `0.0497061` | 0 |
| `10000000` | 1 | 20,000 | 153 | `0.007650 ± 0.000616` | `0.00754353` | 0 |

The zero observed errors establish exact behavior for the sampled noiseless
branches; they are not a finite upper bound small enough for a production
logical-error claim.

The local three-qubit TMR convention was also sampled independently with
ClifT and tsim for 10,000 shots per branch.  Both simulators reproduced the
one target and three equal-probability alternative classes within sampling
variation, with no final-detector or logical failures after repair.

## Measurement-noise MAP decoder

The pilot used 1,000 randomly drawn ideal branch classes per configuration.
Every one of the 32 displayed syndrome bits was independently flipped with
probability `p_m` in each round.

| `p_m` | Rounds | Exact-class accuracy | Repair-mask accuracy |
|---:|---:|---:|---:|
| `0.001` | 1, 3, 5, 7 | `1.000` | `1.000` |
| `0.010` | 1, 3, 5, 7 | `1.000` | `1.000` |
| `0.030` | 1 | `0.993` | `0.995` |
| `0.030` | 3, 5, 7 | `1.000` | `1.000` |

This confirms the exact MAP implementation and the benefit of repeated
measurements in its stated independent-readout model.  It does not include
data, preparation, or CNOT faults.

## Scheduled noisy repair component

At physical error probability `p=10^-3`, an ideal selected branch was followed
by scheduled noisy `M=1` repair and one full final syndrome round.  The final
syndrome was postselected.

| Alternative mask | Repairs | Attempts | Survivors | Final-check acceptance | Any-logical infidelity among survivors |
|---|---:|---:|---:|---:|---:|
| `10000000` | 1 | 4,500 | 1,014 | `0.22533 ± 0.00623` | `0.01578 ± 0.00391` |
| `10010000` | 2 | 4,500 | 1,107 | `0.24600 ± 0.00642` | `0.02620 ± 0.00480` |

These numbers characterize the repair component under final-syndrome
postselection.  Their acceptance cannot be multiplied directly by an ideal
branch probability to claim an end-to-end hybrid yield, because the input
branch is idealized and the final syndrome is not decoded.

## Retained syndrome-history diagnostic

Three complete syndrome rounds were retained after the full scheduled `M=3`
preparation.  At `p=0`, all 200 sampled trajectories had X syndromes in the
ideal TMR image in every round, zero Z syndrome, no history drift, and an
unambiguous unit-posterior MAP result.

At `p=10^-3`, the 200-shot diagnostic produced:

| Quantity | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| Raw X syndrome in ideal TMR image | `0.285` | `0.200` | `0.145` |
| Mean X Hamming residual to selected class | `2.90` | `3.25` | `3.85` |
| Mean Z syndrome weight | `3.185` | `4.330` | `5.395` |
| Zero Z-syndrome fraction | `0.295` | `0.230` | `0.160` |

Across the full histories, `75.0%` changed between rounds and `9.5%` were tied
under the persistent-syndrome measurement-only model.  The mean reported MAP
posterior was `0.924`, but it is not calibrated under this mismatched physical
noise model.

This is the central preflight result for decoder development: repeated noisy
measurements cannot be treated merely as independent flips around an ideal TMR
class.  The circuit-level decoder must infer a latent TMR class together with
boundary/data faults and must use the Z history as well.

## Bounded latent-boundary decoder

The first joint decoder enumerates all phenomenological data and measurement
events through weight two in each CSS channel.  With three rounds it contains
288 elementary events and 41,521 distinct bounded history patterns per
channel.  A labeled audit drew 100 cases in each configuration:

| X-channel faults | Z-channel faults | In radius | Class accuracy | Repair-mask accuracy | Ambiguous |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | `1.00` | `1.00` | `1.00` | `0.00` |
| 1 | 0 | `1.00` | `1.00` | `1.00` | `0.00` |
| 0 | 1 | `1.00` | `1.00` | `1.00` | `0.00` |
| 2 | 0 | `1.00` | `0.94` | `0.95` | `0.06` |
| 1 | 1 | `1.00` | `0.99` | `1.00` | `0.01` |
| 0 | 2 | `1.00` | `1.00` | `1.00` | `0.00` |

The two-X-fault failures are not an implementation surprise: maximum joint
probability can favor a different latent branch and lower-cost fault
explanation, and not every wrong result is tied.  The bounded decoder therefore
serves as an exact oracle for its small event model and a fail-closed
controller prefilter.  These results rule out treating it as the final
scheduled-circuit decoder without a calibrated confidence/reset rule.

## Full scheduled feasibility point

One preliminary trajectory includes noisy initialization, all eight scheduled
`M=3` rotations, the existing syndrome schedule, conditioning on the exact
all-target raw branch, and a final syndrome round at `p=10^-3`.  It reached 20
postselected survivors in 3,000 attempts:

\[
p_{\rm pass}=0.00667\pm0.00149.
\]

Sampling took `86.9 s`, or about `34.5` attempts/s.  No logical errors were
seen among only 20 survivors, which is insufficient to estimate infidelity.
This run establishes that full scheduled branch continuations are executable
at the expected cost; it is not the final hybrid simulation.

## Scheduled circuit-location decoder

The frozen three-round circuit contains 3,016 independent noisy locations.
Expanding depolarizing channels into their mutually exclusive Pauli outcomes
gives 31,064 elementary single-fault mechanisms.  Exact propagation compresses
these to 15,433 distinct tuples containing:

- all 196 detector changes;
- the final 64-data-qubit X and Z Pauli;
- its canonical physical correction and eight logical X/Z frame bits; and
- the subset of 24 non-Clifford rotations whose signs reverse.

The complete single catalog took `61.0 s` to build and compresses to a 193 KB
local NPZ archive.  The first important-pair pilot combines the 64 most
probable single-signature groups, producing 2,017 distinct pair signatures in
`0.20 s`.

At `p=10^-3`, the exact global fault-count masses are:

| Included fault locations | Probability mass |
|---|---:|
| 0 | `0.0489229` |
| exactly 1 | `0.147699` |
| exactly 2, all distinct-location pairs | `0.222880` |
| 0 through 2, all pairs | `0.419502` |
| 0, 1, and the selected pair pilot | `0.198935` |

Thus `0.801065` remains an adversarial omitted-probability bound for the
current pilot.  The decoder does not renormalize that mass away.

A labeled audit gave:

| Audit | Cases | Result |
|---|---:|---:|
| Ideal histories: best action equals no-fault truth | 32 | 32 |
| Random single signatures: best action equals injected truth | 64 | 52 |
| Direct two-fault propagation equals composed signatures | 100 | 100 |
| Ideal histories accepted at rigorous posterior threshold `0.99` | 32 | 0 |

The 12 single-signature disagreements are Bayesian/action ambiguities: another
latent-branch-plus-fault explanation has greater aggregate probability.  They
are not propagation mismatches.  This is exactly why the decision rule sums
probability by complete repair action and exposes a reset instead of treating
the injected label as observable truth.

ClifT supplied two independent non-Clifford checks:

- all 144 combinations of 24 rotations, three Pauli axes, and before/after
  insertion were sampled for 200 shots each; all 28,800 histories returned to
  the ideal persistent syndrome image after the symbolic detector delta;
- all 32 local combinations of eight rotation-sign patterns and four syndrome
  labels were sampled for 5,000 attempts; after the derived frame and angle
  repair there were zero final-detector and zero logical errors.  Acceptance
  probabilities agreed with the analytic values within `2.19` standard errors
  at worst.

## Stage-D3 arbitrary-weight decoder prototype

The exact scheduled model has now been converted into a categorical factor
graph without splitting a depolarizing location into mutually compatible
binary faults.  The frozen graph contains:

| Quantity | Count |
|---|---:|
| Physical categorical variables | 3,016 |
| Four-state TMR branch variables | 8 |
| Binary symplectic/quotient components | 9,568 |
| Detector equations | 196 |
| Variable-detector edges | 54,846 |

The variables retain 920 one-qubit depolarizing locations, 1,872 two-qubit
depolarizing locations, 128 `X_ERROR` locations, and 96 `Z_ERROR` locations.
Building and validating the complete graph took `56.6 s`; loading its 161,074
byte compressed archive took about `0.2 s`.

The decoder first runs sparse categorical BP, then constructs a finite OSD
list whose candidates satisfy the complete observed detector history exactly.
It evaluates each candidate with the original categorical likelihood and sums
equivalent explanations by a canonical action modulo stabilizers.  On 20
labeled `p=10^-3` trajectories:

| Diagnostic | Result |
|---|---:|
| Mean physical fault locations | `2.90` |
| BP converged | `8/20` |
| BP hard decision already satisfied all detectors | `6/20` |
| Leading finite-list action matched the label | `12/20` |
| Literal injected state occurred in the 128-entry list | `6/20` |
| Mean leading finite-list probability | `0.8796` |

The `0.8796` value is normalized only over the finite list.  Its disagreement
with the 60% labeled accuracy is direct evidence that it must not be reported
or thresholded as a posterior.

A detector-nullspace Metropolis sampler was added for full-posterior
refinement.  Its stationary target is the exact categorical distribution
conditioned on the detector history, but the first ten-history audit found
poor action-sector mixing.  The strengthened controller reset all ten cases:

| Reset reason | Cases |
|---|---:|
| No retained transition between distinct repair actions | 4 |
| Insufficient effective samples | 6 |
| Accepted actions | 0 |

An earlier smoke setting would have falsely accepted two wrong actions because
a constant chain was assigned an optimistic effective-sample count.  That
failure was retained as a design lesson and fixed: a chain with constant
action has zero empirical mixing information, and the production-facing
controller requires at least one action transition.  D3 is therefore an
implemented, tested, fail-closed prototype—not yet a decoder with useful
acceptance at `p=10^-3`.

## Current conclusion

The non-Clifford algebra, exact branch correction, residual-angle action,
scheduled single-fault signatures, selected-pair composition, and independent
simulator convention are validated.  The new decoder is already a correct
fail-closed circuit-location kernel, but it is not yet an accepting production
decoder at `p=10^-3`: the schedule has an expected fault count near three, the
bounded catalog omits too much probability, and the first scalable posterior
proposal mixes poorly across action sectors.  The next required scientific
milestone is improving and calibrating that posterior engine before the
thresholded end-to-end comparison specified in `DECODER_PLAN.md`.
