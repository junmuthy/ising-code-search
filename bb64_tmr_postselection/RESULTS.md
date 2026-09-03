# BB64 postselection-only results

## Algebraic certificate

The stored balanced `M=3` partition has batch partial-syndrome ranks

\[
(r_0,r_1)=(8,8)
\]

and combined rank

\[
r_{\mathrm{all}}=16.
\]

The corresponding kernel dimensions are `4`, `4`, and `8`, exactly generated
by selecting either none or all three pieces of each logical Z operator.  No
additional syndrome-free partial products occur.  This certifies applying both
four-logical TMR batches before one final stabilizer projection.

## Ideal validation sweep

The 12-point ideal-projection sweep covered
`N in {1,2,4,8}` and `theta in {pi/128,pi/32,pi/8}`.  Every measured
acceptance agrees with

\[
p_{\mathrm{ideal}}(N,\theta)=
\left(\cos^{6}\theta+\sin^{6}\theta\right)^N
\]

within its short-run statistical uncertainty.  Representative `N=8` results
are:

| `theta` | Shots | Measured acceptance | Analytic acceptance | Candidate yield |
|---|---:|---:|---:|---:|
| `pi/128` | 20,000 | `0.29055 ± 0.00321` | `0.288543` | `2.3244` |
| `pi/32` | 100,000 | `0.04989 ± 0.000689` | `0.0497062` | `0.39912` |
| `pi/8` | 1,000,000 | `0.001220 ± 0.0000349` | `0.00119795` | `0.00976` |

This validates the intended TMR angle convention and, together with the
kernel certificate, the use of one final projection for both logical batches.

## Initial circuit-level result at `p=10^-3`

Both protocols below use `N=8`, `M=3`, and `theta=pi/32`.  The noise model
includes preparation, measurement, one-qubit rotation, two-qubit gate, and
entangling-layer idle faults.  It does not yet include teleportation or a
logical-fidelity measurement.

| Protocol | Attempts | Accepted | Acceptance | 95% Wilson interval | Candidate yield |
|---|---:|---:|---:|---:|---:|
| one final check | 41,000 | 401 | `0.009780 ± 0.000486` | `[0.008873,0.010780]` | `0.078244` |
| check after each batch | 200,000 | 1,122 | `0.005610 ± 0.000167` | `[0.005292,0.005947]` | `0.044880` |

The one-final-check run retained about `19.7%` of the ideal TMR survivors;
the two-check pilot retained about `11.3%`.  This is physically plausible:
the second protocol adds one complete syndrome round, increasing the CNOT
count from `848` to `1,360` and the entangling depth from `24` to `32`.

The one-final-check ClifT trajectory has peak active width `24` under noise and
runs at about `38.4` attempts/s using 16 worker threads.  At the observed
acceptance, expected wall times are roughly:

| Statistical target | Expected attempts | Expected wall time |
|---|---:|---:|
| 100 survivors | `10,200` | `4.4 min` |
| 400 survivors | `40,900` | `17.7 min` |
| 2,500 survivors | `255,600` | `1.84 h` |

The two-projection program is much faster computationally (about 0.43 million
attempts/s on this host) because its intermediate postselection collapses the
ClifT active width, despite containing more physical gates.  The difference is
a simulator-state-width effect and not a claim that the physical protocol is
faster.

These are initial postselection estimates.  Raw and generated summary results
live under the ignored `results/` directory; source, manifests, and the frozen
algebraic certificate are version controlled.
