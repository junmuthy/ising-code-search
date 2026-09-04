# BB64 hybrid decoder development plan

## Objective

The decoder must decide whether to reset or repair an eight-resource BB64
block after noisy `M=3` preparation.  On acceptance it must return:

- one of the `65,536` ideal TMR syndrome classes;
- the eight target-versus-alternative angle labels;
- a canonical physical Pauli correction;
- logical X and Z frame updates;
- signed residual `M=1` repair angles; and
- a confidence value that can drive a reset threshold.

The decoder must use all 32 displayed X-check outcomes.  The 16 independent
coordinates are an algebraic parameterization, not permission to discard the
redundant measurements.

## Frozen algebraic model

For logical `j`, let the three TMR piece-selection bits be
`(q[j,0],q[j,1],q[j,2])`.  Complementing all three bits multiplies by the
complete logical Z and does not change the local syndrome.  The quotient
coordinates are

\[
u_j=q_{j,0}+q_{j,2},\qquad
v_j=q_{j,1}+q_{j,2}\pmod 2.
\]

The target branch is `(u_j,v_j)=(0,0)`; the other three coordinate values are
the three alternative branches.  Across eight logicals these form a linear
16-bit coordinate system on the complete ideal syndrome image.  The mapping
from the stored 16 independent syndrome coordinates to these quotient bits is
derived and exhaustively verified by `model.py`.

## Decoder stages

### Stage A: exact ideal oracle — implemented

`NoiselessTableDecoder` verifies that a displayed 32-bit syndrome lies in the
ideal 16-dimensional image and performs the exact class lookup.  All 65,536
classes are covered by tests.

### Stage B: persistent-syndrome MAP oracle — implemented

`MeasurementMapDecoder` assumes one latent ideal syndrome `s` persists through
`R` readout rounds and independently flips each displayed bit with probability
`p_m`.  It evaluates

\[
\log P(s\mid y)
=
\log P_\theta(s)
+
\sum_{t=1}^{R}\log P(y_t\mid s)
\]

for every class.  `P_theta` is the exact angle-dependent TMR branch prior.
The result includes the runner-up likelihood, likelihood gap, normalized MAP
posterior, and an ambiguity flag.  This is the exact small-state-space oracle
against which approximate decoders must be checked.

### Stage C1: retained circuit-fault histories — implemented

`syndrome_history.py` extends the scheduled circuit so every repeated
full-syndrome round is retained rather than postselected.  Each trajectory
currently records all displayed X and Z syndromes by time.  The pilot reports
ideal-image membership, drift between rounds, Z-syndrome activity, and the
behavior of the Stage-B decoder.

### Stage C2: labeled circuit-fault data — next

For supervised decoder calibration, each trajectory must additionally record:

- the true injected TMR branch in labeled calibration data;
- the physical and logical Pauli frame at the repair boundary;
- final stabilizer syndromes; and
- eight final logical observables after the ideal inverse target rotation.

Generate separate training, validation, and held-out test seeds.  Store only
compact sufficient statistics and explicitly requested samples, not every raw
trajectory.  A noisy coherent trajectory does not carry a simulator-provided
classical “true TMR branch,” so labels must be defined through a conditioned
branch experiment or a verified fault ledger rather than inferred from the
same noisy syndrome being evaluated.

### Stage D: latent-boundary circuit decoder — next

Model the first post-TMR syndrome as the sum of an ideal TMR class and a
circuit-fault boundary syndrome.  Subsequent time differences constrain data
and measurement faults.  The reference implementation should jointly score

\[
(s_{\rm TMR},e_{\rm boundary},e_{1},\ldots,e_R)
\]

subject to the BB64 detector equations.  The exact 65,536-class Stage-B oracle
remains practical as an outer latent-state enumeration for small pilots;
later acceleration can shortlist candidates from the quotient map and solve
the Pauli portion with matching, belief propagation plus ordered statistics,
or integer programming.

The first correctness criterion is not logical infidelity.  It is exact
agreement with an exhaustive or maximum-likelihood oracle on small enumerated
fault sets through at least total fault weight two.  Any tie must be surfaced
as ambiguity rather than resolved silently.

### Stage E: frame-aware action — partially implemented

`repair_action` already converts a decoded ideal class into reset or repair,
including X-frame-dependent angle signs.  Connect Stage D's inferred physical
correction and logical frames to this action.  Before simulation, verify for
every retained trajectory that:

- the physical Pauli correction returns the data to the codespace;
- the reported logical frames reproduce the residual Pauli action;
- repaired logical angles equal the target angle in the decoded frame; and
- applying the same history twice produces identical actions.

### Stage F: end-to-end policy comparison — next

For thresholds `b_max=0,1,2,3,8`, simulate the complete sequence:

1. noisy initialization and `M=3` rotations;
2. repeated syndrome acquisition;
3. decode and either reset or apply a frame-aware canonical correction;
4. scheduled `M=1` repairs for at most `b_max` alternative logicals;
5. final error-correction rounds and decoding; and
6. ideal inverse target rotations followed by stabilizer and logical readout.

Report, with binomial standard errors or confidence intervals:

- block acceptance/reset probability;
- produced logical resources per attempt;
- repair-count distribution;
- per-logical infidelity;
- any-logical infidelity; and
- the full cross-logical error covariance matrix.

Compare thresholds at equal physical noise and angle.  A hybrid policy is
useful only if its accepted-resource yield at a fixed output-error target is
better than zero-syndrome STAR postselection.

## Required safeguards

- Never infer an angle branch from a single violated check independently of
  the global 16-dimensional syndrome image.
- Never treat a postselected calibration as a decoded end-to-end estimate.
- Preserve and report decoder ambiguity and reset events.
- Keep logical X and Z frame conventions explicit at every non-Clifford
  command.
- Validate noiseless behavior before enabling each new noise location.
- Print checkpoints during compilation, generation, decoding, and sampling.
- Refuse accidental result overwrite; use new filenames or explicit
  `--overwrite` for deliberate reruns.
