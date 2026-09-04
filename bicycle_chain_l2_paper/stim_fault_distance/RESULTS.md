# Circuit fault distance

The exact result for the paper's Table-II schedule is

\[
\boxed{d_{\mathrm{fault}}=5}.
\]

This holds in both logical bases for:

- a noisy CNOT-only extraction round between ideal guard rounds;
- a three-round full-noise memory experiment;
- a five-round full-noise memory experiment.

The full model includes noisy data/ancilla preparation, CNOTs, measurements,
and entangling-layer idles.  Exact enumeration excludes every undetected
logical mechanism made from one, two, three, or four elementary faults.
Five final-data measurement faults on a saved weight-five logical provide
the matching upper bound.  For the bulk circuit, Stim also independently
finds five-fault circuit witnesses.

The five-round circuits contain 56 qubits, 160 detectors, and four logical
observables.  Their exact four-fault meet-in-the-middle tables contained
about 28.9 million X-basis pairs and 28.5 million Z-basis pairs.

Because the static code distance is five, no syndrome ordering can exceed
this result.  The paper schedule is therefore fault-distance optimal for the
literal `ell=2` specialization, so an alternative-ordering search is not
needed to improve `d_fault`.

The complete machine-readable certificate is written to the ignored file
`results/paper_schedule_r5.json`.
