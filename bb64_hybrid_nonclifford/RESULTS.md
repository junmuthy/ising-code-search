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

## Current conclusion

The non-Clifford algebra, exact branch correction, residual-angle action, and
independent simulator convention are validated.  Retained scheduled histories
are now available, and they rule out using the cheap persistent-syndrome MAP
decoder as the final circuit decoder.  The next required scientific milestone
is the latent-boundary decoder for circuit faults, followed by the thresholded
end-to-end comparison specified in `DECODER_PLAN.md`.
