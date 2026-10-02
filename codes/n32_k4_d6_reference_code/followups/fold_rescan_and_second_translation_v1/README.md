# Fold rescan and second-translation follow-ups

## Outcome

Both requested cheap follow-ups are complete for the saved `[[32,4,6]]`
code and its all-weight-eight, 16-row presentation.

The fold rescan found 24 admissible ZX folds.  Every one independently gives

\[
[[32,4,6]],
\]

and every one admits a clean simultaneous 12-layer `X/Z` syndrome schedule.
The saved schedule uses the original fold, so changing the code or logical
Hadamard is unnecessary.  The improvement comes from the newer redundant
16-row check presentation and the joint scheduling search.

The second-translation search is negative: the exact Tanner automorphism
group of the 16-row `H_X` presentation has order four and consists only of the
known physical `C_4` translation and its powers.  No automorphism moves one of
the four check orbits to another, and no order-16 translation is present in
this presentation.

## 1. Exhaustive affine fold rescan

The scan exhausts every involution in the full affine normalizer of the known
physical `C_4` action that has the form

\[
P(a,t)=(\epsilon a+s_t,\pi(t)),
\qquad
\epsilon\in\{+1,-1\},
\]

where `a` is the logical `C_4` coordinate, `t` is any of the eight thickness
coordinates, `pi` is an arbitrary involution of all eight fibres, and the
shifts satisfy the exact condition `P^2=1`.  Unlike the earlier separate
scans, this simultaneously allows thickness permutation and
thickness-dependent logical shifts, and it does not fix the spectator fibre.

The complete counters are:

| Quantity | Count |
| --- | ---: |
| Affine involutive folds tested | 2,036,992 |
| Permutation logical pairings | 622,208 |
| Folded CSS pairs | 24 |
| Connected exact-distance-six codes | 24 |

Independent qLDPC validation gives `n=32`, `k=4`, and `d_X=d_Z=6` for all 24
survivors.  Every survivor has the same combined data-degree histogram:

\[
4^4\,8^{24}\,12^4.
\]

Thus every fold has the same collision lower bound of 12 CNOT layers.  No
alternative fold improves that lower bound.

## 2. Optimal joint schedule

For each of the 24 folds, the exact finite-domain search assigns one time to
each of the 32 free `C_4` Tanner-edge orbits.  It imposes:

- collision freedom across both CSS types;
- exact physical-`C_4` invariance of every time layer;
- exchange of `X` and `Z` gates by `P` plus time reversal;
- even cross-ancilla back-action for every `X/Z` check pair.

All 24 folds are satisfiable at depth 12.  The saved certificate uses the
original fold and has:

| Quantity | Value |
| --- | ---: |
| CNOT depth | 12 |
| CNOT count | 256 |
| Dedicated syndrome ancillas | 32 |
| Check weight | 8 |
| Lower bound from maximum combined data degree | 12 |

The depth is therefore optimal for this measured presentation.  This
supersedes the previously saved 16-layer sequential schedule as the best
known ideal clean schedule.  It requires both CSS ancilla sets concurrently;
the old 16-layer schedule remains useful when only 16 reset-and-reused
ancillas are available.

Circuit fault distance is still not certified.  Collision freedom and clean
cross-ancilla propagation do not replace the separate hook-fault/Stim search.

## 3. Second physical translation

The second search exhaustively enumerates every color-preserving automorphism
of the bipartite Tanner graph whose check vertices are the 16 saved `X`
generators and whose data vertices are the 32 physical qubits.  There are
exactly four automorphisms.  They are the four powers of the already known
translation `T` and each preserves all four check orbits individually.

This rules out a second translation, and hence an exposed regular order-16
BB/2BGA action, for this exact 16-row presentation.  It does not prove that no
different redundant presentation of the same stabilizer row space could
reveal additional symmetry, nor is it a general code-equivalence
classification.

## Files

- `fold-scan-summary.json`: complete affine-fold counters.
- `distance-six-folds.jsonl`: all 24 fold certificates.
- `joint-schedule-summary.json`: exact scheduling result for every fold.
- `optimal-clean-joint-12-layer.json`: full best schedule certificate.
- `tanner-automorphisms.json`: complete Tanner automorphism result.
- `validation.json`: independent qLDPC and structural validation.
- `rescan_folds.py`: reproducible exhaustive fold scanner.
- `optimize_joint_schedules.py`: exact joint-schedule solver.
- `check_automorphisms.py`: exact Tanner automorphism enumerator.
- `validate_followups.py`: independent validator.
