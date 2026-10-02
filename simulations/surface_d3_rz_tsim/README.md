# Distance-three rotated-surface-code TMR demo

This directory is a deliberately small, direct `tsim` demonstration of the
STAR transversal multi-rotation (TMR) resource-state preparation mechanism.
It prepares

\[
|m_\theta\rangle_L=R_{Z,L}(\theta)|+\rangle_L
\]

in the `[[9,1,3]]` rotated surface code and verifies the result by conditioning
on a trivial stabilizer syndrome.

It is not yet a complete STAR factory simulation. There is no physical noise,
decoder, atom loss, inverse-reference patch, or RUS teleportation in this first
version.

## Code and physical labels

The canonical data grid is

```text
 1   3   5
 8  10  12
15  17  19
```

with syndrome ancillas `2, 9, 11, 13, 14, 16, 18, 25`. The chosen logicals are

\[
Z_L=Z_1Z_3Z_5,
\qquad
X_L=X_1X_8X_{15},
\]

and

\[
Y_L=iX_LZ_L=Y_1Z_3Z_5X_8X_{15}.
\]

`surface_code.py` exactly verifies the check ranks, commutation relations,
logical anticommutation, encoder state, and distance.

## TMR convention

The logical-Z support is partitioned into three single-qubit pieces. With
Tsim's convention

\[
R_Z(\alpha)=\exp(-i\alpha\pi Z/2),
\]

the common physical angle is chosen so that

\[
(-i)^3\tan^3(\theta_*/2)=-i\tan(\theta/2).
\]

For a positive target angle this requires a negative physical angle,

\[
\theta_*=-2\arctan\!\left(\tan^{1/3}(\theta/2)\right).
\]

The ideal trivial-syndrome probability is

\[
p_{\mathrm{TMR}}
=\cos^6(\theta_*/2)+\sin^6(\theta_*/2).
\]

## Two circuit modes

`projection` uses ideal MPP measurements of the eight stabilizer generators.
It is the smallest circuit that exposes the TMR mechanism.

`scheduled` replaces the direct projection by two initialization and two
post-TMR extraction rounds. Each round uses the eight canonical ancillas and
the four CNOT layers emitted by Stim's rotated-surface-code generator. It has
32 explicitly labeled detector outcomes: 16 initialization detectors and 16
post-TMR detectors.

Both modes begin with an exact Clifford encoder for the fixed all-`+1`
`|+>_L` stabilizer frame. This is intentional: a raw transversal `RX`
initialization gives random Z-check signs, which would require a Pauli-frame
correction or sign-adapted physical rotations. Measurement-based initialization
is the next extension after this ideal mechanism is validated.

## Run

Use the installed tsim environment:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  -m unittest discover -s tests -v
```

Run the compact ideal projection experiment:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  run_ideal_demo.py \
  --mode projection \
  --angles 0 0.01 0.39269908169872414 \
  --shots 20000 \
  --batch-size 5000 \
  --save-circuits
```

Exercise the explicit scheduled circuit at one paper-scale angle:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  run_ideal_demo.py \
  --mode scheduled \
  --angles 0.01 \
  --shots 5000 \
  --batch-size 1000 \
  --output results/scheduled-theta-0.01.json \
  --save-circuits
```

Every compile and sample batch prints a checkpoint. Result JSON contains the
analytic and sampled acceptance, conditional logical observables, syndrome
histograms, environment versions, and timing.

Print one generated circuit and its qubit/check metadata with:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  inspect_circuit.py --mode scheduled --theta 0.01 --basis X
```

## Acceptance and readout

The runner samples detector outcomes directly and performs transparent manual
filtering. A shot is accepted exactly when all selected stabilizer detectors
are zero. It then estimates, in separate commuting-measurement circuits,

\[
\langle X_L\rangle=\cos\theta,
\qquad
\langle Y_L\rangle=\sin\theta,
\qquad
\langle Z_L\rangle=0.
\]

Terminal logical measurements are never included in the acceptance condition.
This follows the STAR convention that final transversal readout is diagnostic,
not additional postselection information.

## References

- `references/Ismail et al. - 2026 - Transversal architecture for megaquop-scale quantum simulation with neutral atoms.pdf`, especially Fig. 5 and Appendix G.
- `references/Ismail et al. - 2026 - Fast and Parallel High-Rate STAR Architecture for Megaquop Quantum Simulation.pdf`, especially Appendix C.
- `references/Toshio et al. - 2025 - Practical Quantum Advantage on Partially Fault-Tolerant Quantum Computer.pdf`, especially Sec. III.
