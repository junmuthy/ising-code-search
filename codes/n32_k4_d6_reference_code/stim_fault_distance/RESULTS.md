# Stim circuit-fault results for the `[[32,4,6]]` code

## Outcome

The saved all-weight-eight, simultaneous 12-CNOT-layer stabilizer schedule has

\[
\boxed{d_{\mathrm{fault}}^X=d_{\mathrm{fault}}^Z=4.}
\]

This is an exact Clifford-circuit result, not a graphlike approximation.  For
every reported experiment, direct enumeration proves that no set of one, two,
or three elementary circuit faults can produce zero detector syndrome and a
nonzero logical observable.  Stim independently supplies a concrete set of
four circuit faults that does exactly that.

## Certified experiments

| Experiment | Fault model | Syndrome rounds | Distinct effects | Detectors | `d_fault(X)` | `d_fault(Z)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Isolated bulk cycle | CNOT only | `3` including guards | `1,984` | `108` | `4` | `4` |
| Isolated bulk cycle | Full | `3` including guards | `1,984` | `108` | `4` | `4` |
| Full memory | Full | `1` | `791` | `36` | `4` | `4` |
| Full memory | Full | `2` | `3,196` | `72` | `4` | `4` |
| Full memory | Full | `3` | `5,088` | `108` | `4` | `4` |
| Full memory | Full | `6` | `10,752` | `216` | `4` | `4` |
| Full memory | Full | `18=3d` | `33,408` | `648` | `4` | `4` |

The full model includes preparation depolarization, measurement flips,
two-qubit depolarization after each CNOT, and single-qubit depolarization on
qubits idle during entangling layers.

## Paper-scale temporal certificate

For each basis of the 18-round memory circuit, the exact three-fault exclusion
examined

\[
\binom{33408}{2}=558{,}030{,}528
\]

fault pairs.  The `X` search required 113.5 seconds and the `Z` search 108.7
seconds on the current workstation.  Timestamped heartbeat records were
written throughout the searches.

## Interpretation

The extra preparation, measurement, and idle locations do not cause the
distance reduction: the CNOT-only central-cycle experiment already has a
four-fault logical witness.  The limitation is therefore in the hook-error
geometry of this particular 12-layer CNOT ordering.

The result does not invalidate the static `[[32,4,6]]` code, its ZX fold, its
four disjoint logical supports, its even syndrome parity, or its exact
collision-depth optimum.  It says that collision depth and clean
cross-ancilla back-action are insufficient to guarantee the desired circuit
fault distance.  A schedule search must explicitly optimize the propagated
fault signatures.

The immediate next target is another clean 12-layer ordering with

\[
d_{\mathrm{fault}}\geq5,
\]

while preserving `C4`-invariant layers, fold-plus-time-reversal symmetry, and
the redundant all-weight-eight presentation.
