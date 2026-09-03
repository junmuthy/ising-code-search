# `[[22,2,6]]` teleportation and RUS analysis

This package implements the resource-state fidelity estimator and stochastic
repeat-until-success analysis of Appendix C and Figs. 9, 10, and 19 of the
High-Rate STAR paper for the saved `[[22,2,6]]` shortened-Golay presentation.
It leaves the preceding post-selection package unchanged.

## Circuit convention

The 66-qubit estimator circuit uses:

- qubits `0`--`21`: noisy TMR resource block and eventual output;
- qubits `22`--`43`: resource-block syndrome ancillas;
- qubits `44`--`65`: ideal inverse-rotation reference block.

The resource is the control of a transversal 22-CNOT layer and the reference
is the target.  The reference is measured transversally in Z and the resource
in X.  On logical outcome `Z_L=+1`, the two rotations cancel; the fraction of
these shots decoded as `X_L=-1` estimates resource-state infidelity.

The estimator tail is ideal unless `--teleportation-probability` is nonzero.
The operational calibration sets both preparation and teleportation
probabilities to `0.001`, thereby including the transversal-CNOT and terminal
readout floor in every RUS level.

## Decoder

`decoder.py` decodes the terminal Z and X strings jointly as a single Pauli
error.  It enumerates corrections by increasing symplectic weight and resolves
only syndromes observed by the sampler.  For a fixed observed syndrome this is
an exact minimum-weight decoder through `--max-decoder-weight`; minimum-weight
logical-class ties are resolved by degeneracy and counted in the result JSON.

This has the same joint terminal-readout role as the paper's Gurobi MLE
decoder, but its likelihood model is residual i.i.d. Pauli weight rather than a
full circuit-location likelihood model.  Every Pauli error of weight at most
two is tested exhaustively and decoded correctly.

## Validation

Run the ClifT tests with:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m unittest discover \
  -s n22_k2_d6_shortened_golay/teleportation/tests -v
```

Run the independent zero-noise `tsim` check with:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  -m n22_k2_d6_shortened_golay.teleportation.run_tsim_crosscheck \
  --theta 0.09817477042468103 --shots 2000
```

Both simulators produce all four two-logical sign patterns with probability
near `1/4` and zero errors on the ideal cancellation branches.

## Running the estimator

The runner checkpoints atomically and prints progress after every shot batch:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m n22_k2_d6_shortened_golay.teleportation.run_estimator \
  --theta 0.09817477042468103 \
  --partition-count 3 --logical-count 2 \
  --preparation-probability 0.001 \
  --teleportation-probability 0.001 \
  --max-shots 20000000 --checkpoint-shots 1000000 \
  --output n22_k2_d6_shortened_golay/teleportation/results/example.json
```

The two committed manifests reproduce the 14 circuit-level calibration
points.  Raw checkpoint JSON and emitted circuits are ignored; the compact CSV
and SVG summaries are version controlled.

## RUS model

`build_injection_model.py` converts completed estimator results into an
angle- and active-set-dependent injection table.  `run_rus.py` then simulates
the two independent sign ladders while preserving blockwise post-selection
costs and correlated logical-error patterns.

At each level it:

1. repeats preparation geometrically until the whole block is accepted;
2. samples the calibrated logical error pattern;
3. terminates each active logical with probability `1/2`;
4. doubles the angle for the remaining logicals;
5. terminates for free when the correction reaches `R_Z(pi)=Z`, a Pauli-frame
   update.

The model assumes that the error distribution measured on the Fig. 10
cancellation branch also applies to the wrong-sign branch.  This is the
stochastic resource-state approximation used to turn injection data into a
full RUS estimate; it is not a monolithic non-Clifford simulation of every RUS
level in one circuit.  When only logical 1 remains active, it also uses the
single-active-logical calibration measured on logical 0; this relies on the
code's order-two logical symmetry.

See `RESULTS.md` for the measured baseline and `RUS_P1E3.svg` for the initial
angle sweep.
