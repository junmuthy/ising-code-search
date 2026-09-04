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

### Stage C2: labeled circuit-fault data — implemented for the exact Pauli model

For supervised decoder calibration, each trajectory must additionally record:

- the true injected TMR branch in labeled calibration data;
- the physical and logical Pauli frame at the repair boundary;
- final stabilizer syndromes; and
- eight final logical observables after the ideal inverse target rotation.

`circuit_decoder/labeled_noise.py` now samples the native categorical channel
at every scheduled location and the eight ideal TMR branch variables.  It
composes the exact detector, boundary-frame, rotation-sign, and repair-action
label without inferring any label from the syndrome being decoded.  Direct
four-location composition is covered by a propagation regression test.

Separate large training, validation, and held-out test seeds are still needed
before any acceptance threshold can be calibrated.  Store only compact
sufficient statistics and explicitly requested samples, not every raw
trajectory.  A noisy coherent trajectory does not carry a simulator-provided
classical “true TMR branch,” so ClifT end-to-end labels must continue to use a
conditioned branch experiment or the verified fault ledger.

### Stage D1: bounded latent-boundary oracle — implemented

Model the first post-TMR syndrome as the sum of an ideal TMR class and a
circuit-fault boundary syndrome.  Subsequent time differences constrain data
and measurement faults.  `BoundedLatentBoundaryDecoder` exactly maximizes the
joint class-and-explanation score through phenomenological fault weight two in
each CSS channel:

\[
(s_{\rm TMR},e_{\rm boundary},e_{1},\ldots,e_R)
\]

The X history is decomposed into a repeated ideal TMR syndrome plus persistent
data-Z and transient check-measurement events.  The Z history is decomposed
into data-X and measurement events around zero.  The decoder reports
out-of-radius and likelihood-tied cases explicitly; `boundary_repair_action`
resets both rather than guessing.

The first correctness criterion is not logical infidelity.  It is exact
agreement with an exhaustive or maximum-likelihood oracle on small enumerated
fault sets through at least total fault weight two.  Any tie must be surfaced
as ambiguity rather than resolved silently.  The implemented labeled audit
covers zero, one, and two injected phenomenological events.

### Stage D2: scheduled circuit-location decoder kernel — implemented

`circuit_decoder` now enumerates every elementary Pauli outcome of the actual
preparation, CNOT, idle, rotation, and measurement locations.  It propagates
each outcome exactly through the scheduled Clifford circuit, records all 196
detector deltas, decomposes the final data Pauli into a canonical physical
correction and logical frame, and records every non-Clifford rotation whose
sign is reversed.  Correlated two-qubit depolarizing outcomes are individual
15-way mechanisms, not independent one-qubit approximations.

Mechanisms with identical detector/frame/sign transformations are compressed.
Selected pairs are composed only across distinct physical locations.  At
decode time the ideal 65,536-class model remains an outer latent variable, but
only classes compatible with a given signature and the complete repeated X/Z
history are scored.  Explanation probabilities are summed by complete repair
action rather than maximized by literal fault.

All 144 before/after `X`, `Y`, and `Z` cases at the 24 rotation pivots were
checked with ClifT.  A separate phase-sensitive fixture checked all 32 signed
local TMR branch cases.

### Stage D3: scalable higher-fault inference — prototype implemented

At `p=10^-3`, the circuit has 3,016 noisy locations.  The exact no-fault mass
is only `0.0489`; all zero-, one-, and two-location terms together contain
`0.4195`.  The initial top-64-signature pair pilot models `0.1989`.  The
implemented posterior therefore assigns every omitted term adversarially
against the leading action and correctly resets all cases at a `0.99`
threshold.

The categorical factor graph now absorbs arbitrary-weight combinations without
enumerating location subsets.  Its 3,024 variables comprise 3,016 physical
locations and eight four-state TMR branches.  The physical alphabets retain
the true channel exclusivity: 4 states for `DEPOLARIZE1`, 16 for
`DEPOLARIZE2`, and 2 for `X_ERROR` or `Z_ERROR`.  Their 9,568 binary
symplectic/quotient components couple sparsely to all 196 detector equations.

The implemented inference path is:

1. categorical log-domain belief propagation;
2. ordered-statistics construction of exact-syndrome candidates;
3. exact probability summation by canonical complete repair action; and
4. detector-nullspace MCMC targeting the full categorical posterior.

The MCMC target is exact, but finite-chain estimates are not automatically
calibrated.  The controller therefore resets unless the leading action clears
the posterior lower-bound threshold, the effective sample count is adequate,
chains agree, and at least one retained transition between actions occurred.
This last condition prevents a chain trapped in one wrong action basin from
claiming unit confidence.

The first `p=10^-3` audit is diagnostic rather than production-ready.  A
20-history BP+OSD run selected the labeled action in 12 cases, while its mean
finite-list probability was `0.880`; this mismatch confirms that list
probabilities are not calibrated posteriors.  A ten-history MCMC audit safely
reset every case: four lacked action mixing and six had insufficient effective
samples.  The next D3 milestone is a global proposal or importance sampler
that mixes across action sectors, followed by larger held-out calibration.

### Stage E: frame-aware action — implemented for bounded and scheduled decoders

`repair_action` converts a decoded ideal class into reset or repair, including
X-frame-dependent angle signs.  `boundary_repair_action` adds the bounded
decoder's physical X/Z corrections and fails closed on ambiguity or an
out-of-radius history.  `ScheduledActionDecoder` supplies canonical physical
X/Z corrections, logical X/Z frames, and signed residual angles.  It groups
posterior weight by that complete action and refuses low-confidence output.
Before end-to-end simulation, still verify for every retained trajectory that:

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
