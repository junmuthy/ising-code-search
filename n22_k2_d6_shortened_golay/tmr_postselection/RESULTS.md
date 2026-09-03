# `[[22,2,6]]` post-selection-only results

## Algebraic and circuit validation

The stored `2+2+3` partitions have partial-syndrome rank

\[
r=4
\]

and kernel dimension

\[
6-r=2.
\]

Those two kernel directions are exactly the complete three-piece products for
logical 0 and logical 1.  Thus both disjoint logical rotations may be applied
in parallel before one final stabilizer projection.  Seven automated tests
also verify the frozen hashes, CSS logical pairing, exact encoded-plus state,
initialization right inverse, operation counts, and zero-noise survival.

## Ideal validation

For target logical angle `theta` and `M=3`, let `theta_star` be the physical
TMR angle.  The ideal per-logical acceptance is

\[
p_1=\cos^6(\theta_\star/2)+\sin^6(\theta_\star/2),
\]

and two disjoint logicals accept with probability `p_1^2`.  The ideal sampler
agrees with this prediction:

| `theta` | Logicals | Shots | Measured acceptance | Analytic acceptance | Candidate yield |
|---|---:|---:|---:|---:|---:|
| `pi/128` | 1 | 20,000 | `0.85485 ± 0.00249` | `0.856104` | `0.85485` |
| `pi/128` | 2 | 20,000 | `0.73375 ± 0.00313` | `0.732914` | `1.46750` |
| `pi/32` | 1 | 50,000 | `0.68464 ± 0.00208` | `0.687150` | `0.68464` |
| `pi/32` | 2 | 50,000 | `0.47072 ± 0.00223` | `0.472175` | `0.94144` |
| `pi/8` | 1 | 100,000 | `0.43186 ± 0.00157` | `0.431325` | `0.43186` |
| `pi/8` | 2 | 200,000 | `0.18573 ± 0.000870` | `0.186041` | `0.37146` |

## Circuit-level result at `p=10^-3`

At `N=2`, `M=3`, and `theta=pi/32`, with the same circuit-level depolarizing
noise assumptions used for BB64:

| Attempts | Accepted | Block acceptance | 95% Wilson interval | Candidate yield |
|---:|---:|---:|---:|---:|
| 1,000,000 | 294,727 | `0.294727 ± 0.000456` | `[0.293834,0.295621]` | `0.589454` |

The noisy circuit retains `62.42%` of the ideal TMR survivors.  Equivalently,
one accepted two-resource candidate block takes about `3.39` attempts on
average.  The run completed in well under one second of reported sampling
time on this host; compilation took approximately four milliseconds.

For comparison, the earlier eight-resource BB64 point at the same `theta` and
`p` accepted about `0.978%` of whole blocks.  The much larger `29.47%` here is
primarily the expected benefit of requiring only two resource states to pass
jointly, together with the smaller physical circuit.  It is not yet a logical
infidelity comparison: teleportation and output-error measurements remain to
be performed.

## Full-circuit angle sweep

The full two-logical experiment was repeated at 13 logarithmically spaced
target angles from `1` through `10^-3` radians, with one million attempts at
each angle.  The noise model remains `p=10^-3` on preparation, measurement,
rotation, CNOT, and entangling-layer idle locations.

![Post-selection acceptance versus target logical angle](ANGLE_SWEEP_P1E3.svg)

Measured block acceptance increases from
`0.048888 ± 0.000216` at `theta=1` to
`0.601928 ± 0.000490` at `theta=10^-3`.  The ratio between noisy and ideal
TMR-only acceptance stays in the narrow interval `[0.62453,0.62645]` across
the sweep.  The complete table and machine-readable data are in
`ANGLE_SWEEP_P1E3.md` and `ANGLE_SWEEP_P1E3.csv`.
