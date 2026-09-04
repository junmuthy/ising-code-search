# Four-logical TMR post-selection

The four disjoint weight-seven logical Z fibres are each partitioned as
`2+2+3`.  The twelve partial operators have X-syndrome rank eight and kernel
dimension four.  That kernel is generated exactly by the four intended full
logical products, so all four preparations can use one final syndrome
projection without admitting an unintended partial product.

## Circuit

The full post-selection circuit contains:

- 28 data qubits and 28 reusable syndrome ancillas;
- one scheduled Z-only initialization round and virtual X-frame correction;
- 12 partial rotations implemented by 32 TMR-ladder CNOTs;
- one simultaneous full X/Z syndrome round;
- 368 physical CNOTs in 20 entangling layers;
- 12 physical one-qubit rotations;
- 28 final post-selection outcomes.

Preparation, measurement, rotation, CNOT, and entangling-layer idle noise are
all set to the same probability `p`, matching the previous BB64 and shortened
Golay experiments.

## Primary result

At

\[
p=10^{-3},\qquad \theta=\frac{\pi}{32},\qquad M=3,\qquad N=4,
\]

the analytic ideal four-logical TMR acceptance is

\[
p_{\mathrm{ideal}}=0.2229487601.
\]

One million attempts per mode gave:

| Mode | Acceptance | Standard error | Noise/ideal | Candidate yield `4s` |
|---|---:|---:|---:|---:|
| Ideal projection sampler | `0.221812` | `0.000415` | `0.99490` | `0.887248` |
| Scheduled noisy TMR only | `0.158584` | `0.000365` | `0.71130` | `0.634336` |
| Full circuit | `0.123622` | `0.000329` | `0.55449` | `0.494488` |

The full-circuit 95% Wilson interval is

\[
[0.1229783,0.1242686].
\]

Thus one accepted block containing all four candidate resource states takes

\[
\frac{1}{p_{\mathrm{accept}}}=8.089
\]

attempts on average.  Marginal diagnostic pass rates were `0.147737` for the
final X-syndrome conditions and `0.670611` for the final Z-syndrome
conditions; their joint acceptance was `0.123622`.  The X projection is the
dominant rejection channel, largely because it contains the intended TMR
projection.

## Angle dependence

The fixed-`M=3` one-million-attempt sweep from `theta=1` to `10^-3` radians is
stored in `ANGLE_SWEEP_P1E3.csv` and plotted in `ANGLE_SWEEP_P1E3.svg`.
Full-circuit acceptance rises from

\[
0.003441\pm0.0000586\quad(\theta=1)
\]

to

\[
0.512838\pm0.000500\quad(\theta=10^{-3}).
\]

Across the sweep, noisy acceptance is approximately 55.3--56.3% of the
ideal TMR-only acceptance.

These accepted blocks are candidate resource states.  Acceptance alone does
not measure conditional logical infidelity; that requires the later
teleportation/observable experiment.
