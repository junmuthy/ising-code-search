# `[[56,8,6]]` TMR post-selection cross-check

## Benchmark

The benchmark follows Appendix C and Fig. 17 of Ismail et al.:

\[
\ell=4,\quad m=7,\quad N=1,\quad M=3,\quad
\theta=\pi/32,\quad p=10^{-3},\quad f=0.
\]

The full circuit uses:

- 56 data qubits and 56 dedicated syndrome ancillas;
- the published depth-eight Table-II schedule;
- one Z-only initialization round with virtual X feedback;
- three partial-product rotations on a `2+2+3` partition of a weight-seven
  logical;
- one simultaneous full X/Z syndrome round;
- 680 physical CNOTs in 20 entangling layers and 3 physical rotations.

One million attempts required 2.16 seconds of ClifT sampling for the principal
diagnostic run on this machine.

## Results

| Experiment | Measured acceptance | Reference |
|---|---:|---:|
| Ideal TMR projection | `0.687142 +/- 0.000464` | `0.6871495483` exact |
| Full noisy, TMR-X policy | `0.294159 +/- 0.000456` | `0.2950609519` Eq. (C6) fit |
| Full noisy, TMR-X policy, `theta=0` | `0.427712 +/- 0.000495` | `0.4293984514` Eq. (C7) fit |
| Full noisy, strict raw XZ policy | `0.203179 +/- 0.000402` | not the fitted curve |
| Full noisy, strict raw XZ policy, `theta=0` | `0.295418 +/- 0.000456` | not the fitted curve |

For the principal paper point,

\[
p_{\rm TMR}=0.6871495483,
\]

\[
p_{\rm init}=\exp[-0.93(32\cdot4\cdot7+13)10^{-3}]
=0.4293984514,
\]

and

\[
s_{\rm paper}=p_{\rm init}p_{\rm TMR}=0.2950609519.
\]

Our `0.294159` estimate is lower by `0.00090195`, or 0.31% relative. The
difference is 1.98 simulation standard errors. One accepted single-resource
candidate therefore costs about `3.3995` attempts at this point.

The `theta=0` control independently estimates the state-preparation factor as
`0.427712 +/- 0.000495`, 0.39% below the fitted `0.42939845`. The fit is an
empirical model rather than an exact probability, so this level of agreement
is satisfactory.

## Acceptance convention

There is a real convention issue worth preserving. The paper's Fig. 11 labels
the terminal condition as a full X/Z syndrome condition, but its fitted
Eqs. (C6)--(C7) numerically agree with the TMR-sensitive X-check marginal of
this reconstructed circuit. If every nonzero raw X **or** Z outcome is instead
treated as an automatic rejection, the same shots accept with probability
`0.203179`, well below the fit.

The simulator therefore exposes both policies:

- `--postselection-policy tmr-x` reproduces the published acceptance curve;
- `--postselection-policy strict-xz` applies the stricter rule used in the
  earlier BB64 post-selection study.

This result does not by itself show that every retained nonzero Z-syndrome
branch has good conditional logical fidelity. Those branches require the
paper's decoder/teleportation analysis. The cross-check here is deliberately
limited to acceptance, as requested.
