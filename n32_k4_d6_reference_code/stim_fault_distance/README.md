# Clifford circuit-fault analysis for the `[[32,4,6]]` code

This directory turns the saved all-weight-eight stabilizer presentation and
optimal simultaneous 12-layer syndrome schedule into executable Stim memory
circuits.  It deliberately excludes the non-Clifford TMR rotations.  The first
goal is to determine whether the stabilizer circuit retains circuit fault
distance five or better.

## Inputs

The scripts load, without modifying:

- `../schedule/all_weight8_translation_symmetric_v1/presentation.json`;
- `../followups/fold_rescan_and_second_translation_v1/optimal-clean-joint-12-layer.json`;
- the four saved disjoint weight-seven logical representatives in the selected
  `start-010.json` result.

The circuit uses 32 data qubits, 16 dedicated `X` ancillas, and 16 dedicated
`Z` ancillas.  Every full syndrome cycle contains 256 CNOTs in 12 simultaneous
layers.

## Experiments

Two complementary definitions are reported:

\[
d_{\mathrm{fault}}^{\mathrm{bulk}}
\]

allows faults only in the middle syndrome cycle between two ideal guard
cycles.  It isolates hook errors caused by the CNOT ordering.

\[
d_{\mathrm{fault}}^{\mathrm{memory}}
\]

allows faults in the complete preparation, repeated-syndrome, and logical
readout circuit.  Both `X`- and `Z`-basis memories are analyzed independently.

## Detector construction

The circuits include temporal syndrome-difference detectors, compatible
preparation and readout boundary detectors, and both redundant-check relations
in every round.  The latter expose the even syndrome-parity information needed
by the one-round STAR postselection argument.  Four `OBSERVABLE_INCLUDE`
instructions record the saved logical representatives.

## Fault models

The command-line tools support three nested models:

| Model | Included locations |
| --- | --- |
| `cnot` | two-qubit depolarizing faults after scheduled CNOTs |
| `no-idle` | preparation, measurement, and CNOT faults |
| `full` | preparation, measurement, CNOT, and entangling-layer idle faults |

The `full` model follows the circuit-level depolarizing convention of the
bicycle-chain STAR work.  The numerical probability does not affect fault
cardinality as long as it is nonzero.

## Validation

From the repository root:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.stim_fault_distance.run_validation \
  --rounds 3 \
  --output results/n32-k4-d6-stim/validation.json
```

The validation requires zero detector and observable samples in the noiseless
circuits and successful undecomposed detector-error-model construction for
both bases.  Use `--write-circuits DIR` to save human-readable `.stim` files.

## Exact fault-distance search

The exact search extracts every distinct hypergraph detector/observable
signature from Stim.  Cardinalities one through three are exhausted directly;
Stim's hypergraph search supplies a concrete circuit-level upper-bound witness.
When the witness is at cardinality four, the two sides together constitute an
exact certificate.  A Z3 XOR model is retained as the fallback for higher
cardinalities.  The calculation does not use Stim's graphlike approximation.

```bash
.venv/bin/python -m n32_k4_d6_reference_code.stim_fault_distance.run_fault_distance \
  --experiment bulk \
  --basis both \
  --fault-model full \
  --max-faults 6 \
  --heartbeat-seconds 30 \
  --output results/n32-k4-d6-stim/bulk-full.json
```

For a complete three-round memory circuit, replace `--experiment bulk` by
`--experiment memory --rounds 3`.

Every long solver call prints a timestamped heartbeat.  The current basis,
fault bound, solver status, and elapsed time are also atomically written to a
`.checkpoint.json` file.  Interrupted runs therefore leave a valid progress
record.  Re-run the same command with `--resume` to retain completed basis
results; an interrupted active basis restarts its current certification.

## Acceptance target

The static distance is six, so the search stops after fault cardinality six.
The initial target is

\[
d_{\mathrm{fault}}^{X},d_{\mathrm{fault}}^{Z}\geq 5.
\]

If the depth-12 schedule falls below five, the saved fault signatures identify
the responsible hook mechanisms and can rank alternative valid CNOT orderings.

## Initial result

The saved depth-12 schedule has

\[
\boxed{d_{\mathrm{fault}}^X=d_{\mathrm{fault}}^Z=4.}
\]

This is already true in the central-cycle CNOT-only model.  Adding preparation,
measurement, and idle faults does not lower the result.  Complete full-noise
memory circuits with `1`, `2`, `3`, `6`, and `18=3d` syndrome rounds all retain
the same exact distance four in both bases.  In the 18-round experiment, the
one-, two-, and three-fault lower-bound search covers 33,408 distinct fault
signatures and exhausts 558,030,528 pairs per basis.

Thus the circuit is valid and symmetric, but the current ordering misses the
distance-five target achieved by the bicycle-chain schedule.  The next useful
search is over alternative clean 12-layer orderings, using this fault engine as
the ranking and certification stage.
