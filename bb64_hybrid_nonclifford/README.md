# BB64 hybrid non-Clifford recovery simulation

This package is the first executable circuit-level layer built on the exact
`[[64,8,8]]` TMR recovery certificate in `bb64_syndrome_recovery`.  It keeps
the preferred logical basis, the two disjoint logical-Z batches, and the
existing syndrome schedule frozen.

The implementation currently provides:

1. an exact table decoder for all `65,536` ideal TMR syndrome classes;
2. a two-bit linear quotient coordinate for each logical resource, separating
   its target branch from its three equivalent alternative branches;
3. an exact MAP reference decoder for a persistent TMR syndrome observed
   through repeated, independently flipped check measurements;
4. a bounded latent-boundary oracle for data and measurement events through
   weight two in each CSS channel;
5. fail-closed, frame-aware reset-or-repair actions for thresholds zero through
   eight;
6. direct non-Clifford circuits that validate exact branch correction and
   residual-angle repair;
7. scheduled noisy `M=1` repair-component circuits;
8. a full scheduled `M=3` preparation followed by one branch-conditioned
   `M=1` continuation, used only as a feasibility calibration;
9. retained repeated X/Z syndrome histories for circuit-decoder development;
10. an exact fault ledger for all 3,016 scheduled noisy locations and all
    31,064 elementary Pauli outcomes;
11. compressed detector, boundary-frame, and rotation-sign signatures for
    every single fault, with selected distinct-location fault pairs;
12. an action-posterior decoder that sums physically different explanations
    whenever they prescribe the same correction and fails closed when its
    posterior lower bound is insufficient; and
13. independent ClifT checks of every before/after Pauli fault at all 24
    rotations and of all signed local TMR branch angles.

At `theta=pi/32`, each logical has one target class with probability

\[
p_0=0.6871495483161995
\]

and three alternative classes, each with probability

\[
p_1=0.10428348389460015.
\]

All three alternative classes have the same logical angle.  If that angle is
`alpha`, the scheduled deterministic repair applies an encoded `M=1`
rotation through

\[
\Delta=\theta-\alpha.
\]

The sign of `Delta` is reversed when an existing logical-X frame is present.
Logical-Z frame changes commute with the repair rotation and are tracked.

## Scientific boundary

The exact algebra, noiseless branch recovery, and scheduled single-fault
signature engine are complete.  The action-posterior decoder is deliberately
fail closed: it makes no confidence claim from probability mass that its
catalog does not contain.

The retained-history runner demonstrates this distinction directly: scheduled
circuit faults frequently move raw syndromes outside the ideal TMR image.  It
records all repeated X/Z outcomes for decoder development but does not assign
a ground-truth ideal branch to a noisy coherent trajectory.

The bounded decoder remains exact only for its phenomenological event catalog.
The new scheduled decoder replaces that catalog with actual preparation,
CNOT, idle, rotation, and measurement locations, including correlated
two-qubit Pauli outcomes and rotation-sign flips.  At `p=10^-3`, however, zero
and one fault contain only about `19.66%` of total probability after the small
selected-pair extension.  Consequently, a strict `0.99` posterior-lower-bound
policy currently resets every audited history.  This is the correct behavior,
not a production-yield result: scalable higher-fault inference is still
needed before end-to-end repair decisions can be accepted.

Similarly, the noisy `M=1` calibration begins in a selected ideal branch and
postselects the final syndrome.  The raw-branch calibration trusts one exact
first-round syndrome and also postselects the final syndrome.  Neither number
is a final end-to-end hybrid logical-error rate.  The remaining circuit-level
decoder work and its acceptance criteria are specified in
`DECODER_PLAN.md`.

## Environments

Algebra, decoder tests, and measurement-noise pilots use the repository
environment:

```bash
.venv/bin/python -m unittest discover -s bb64_hybrid_nonclifford/tests -v
```

ClifT circuits use:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python
```

and the independent tsim fixture uses:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python
```

## Reproducible commands

Run the repeated-measurement MAP pilot:

```bash
.venv/bin/python -m bb64_hybrid_nonclifford.run_decoder_pilot \
  --shots 1000 \
  --batch-size 16 \
  --output bb64_hybrid_nonclifford/results/measurement_decoder_pilot_1000.json
```

Audit every ideal decoder output, policy threshold, and logical-X-frame sign:

```bash
.venv/bin/python -m bb64_hybrid_nonclifford.run_action_audit
```

Audit the bounded latent-boundary oracle on labeled zero-, one-, and two-fault
phenomenological histories:

```bash
.venv/bin/python -m bb64_hybrid_nonclifford.run_bounded_decoder_audit
```

Validate an all-target branch and a one-alternative branch with direct exact
non-Clifford circuits:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_hybrid_nonclifford.run_ideal_validation \
  --backend clifft \
  --shots 20000 \
  --checkpoint-shots 2000 \
  --output bb64_hybrid_nonclifford/results/ideal_validation_clifft_20000.json
```

Cross-check the local TMR convention independently:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_hybrid_nonclifford.run_toy_crosscheck \
  --backend clifft \
  --shots 10000 \
  --output bb64_hybrid_nonclifford/results/toy_clifft.json

/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-tsim/bin/python \
  -m bb64_hybrid_nonclifford.run_toy_crosscheck \
  --backend tsim \
  --shots 10000 \
  --output bb64_hybrid_nonclifford/results/toy_tsim.json
```

Retain three complete syndrome rounds after the scheduled `M=3` preparation:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_hybrid_nonclifford.run_history_pilot \
  --probability 0.001 \
  --syndrome-rounds 3 \
  --shots 200 \
  --checkpoint-shots 50 \
  --output bb64_hybrid_nonclifford/results/syndrome_history_p1e3_200.json
```

Every long runner prints progress checkpoints.  Decoder and validation runners
refuse to overwrite an existing output unless `--overwrite` is explicitly
given.  Raw run JSON remains ignored by the repository-wide results policy;
the compact, reviewed findings are versioned in `RESULTS.md` and
`manifests/validation_summary.json`.  Exact software revisions and frozen-input
hashes are in `manifests/environment.json`.

Build the complete scheduled single-fault catalog and the first selected-pair
extension (about one minute on the recorded workstation):

```bash
.venv/bin/python -m bb64_hybrid_nonclifford.run_scheduled_catalog \
  --pair-top-groups 64 \
  --output bb64_hybrid_nonclifford/results/scheduled_fault_catalog_p1e3_pairs64.npz
```

Audit ideal classes, injected single signatures, and direct pair composition:

```bash
.venv/bin/python -m bb64_hybrid_nonclifford.run_scheduled_decoder_audit \
  --catalog bb64_hybrid_nonclifford/results/scheduled_fault_catalog_p1e3_pairs64.npz
```

Cross-check every physical rotation boundary in ClifT:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_hybrid_nonclifford.run_rotation_fault_validation \
  --shots 200 --max-cases 144 \
  --output bb64_hybrid_nonclifford/results/rotation_fault_validation_all_200.json
```

The companion `run_signed_toy_crosscheck` runner verifies the phase-sensitive
angle and Pauli-frame convention for all eight rotation-sign patterns and all
four local syndrome branches.
