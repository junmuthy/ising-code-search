# BB64 Clifford circuit-fault analysis

This directory validates and fault-screens the simultaneous all-weight-eight
basis of the Liang--Chen `[[64,8,8]]` BB code. It deliberately covers only the
Clifford stabilizer-measurement circuit. Non-Clifford STAR resource preparation
and postselection are outside this calculation.

## Circuit under test

| Property | Value |
| --- | ---: |
| Data qubits | 64 |
| Logical qubits | 8 |
| Static distance | 8 |
| Displayed checks | 32 `X` + 32 `Z` |
| Independent checks | 28 `X` + 28 `Z` |
| Check weight | 8 |
| Dedicated syndrome ancillas | 64 total |
| CNOT count per round | 512 |
| Simultaneous CNOT depth | 8, exactly optimal |

The memory circuit measures all 64 displayed checks. Four independent row
relations per type are emitted as redundant detectors; this retains the
single-measurement-fault information carried by the dependent checks. The
three-round circuit has 128 Stim qubits, 256 measurements, 216 detectors, and
eight logical observables.

The selected schedule uses the identity ZX fold plus time reversal. Global
Hadamard exchanges the `X` and `Z` extraction problems, and the saved logical
basis records the associated logical permutation. The schedule itself is a
perfect matching in every entangling layer and has clean cross-ancilla
backaction.

## Noise models

The scripts support:

- `cnot`: two-qubit depolarizing faults after CNOTs only;
- `no-idle`: preparation, measurement, and CNOT faults;
- `full`: the preceding faults plus idle depolarization.

There are no entangling-layer idles in this depth-eight schedule, so `no-idle`
and `full` have the same elementary signatures during those layers. Fault
distance counts undecomposed elementary Stim detector-error mechanisms, after
deduplicating identical detector/observable signatures where appropriate.

## Validation stages

`run_validation.py` verifies both logical bases with ideal sampling, checks the
detector error models, and can save human-readable `.stim` circuits.

`run_direct_certificate.py` exactly excludes fault cardinalities one through
three, or one through four using a sorted pair table. For the three-round full
memory, the four-fault calculation contains 51,770,400 pair records in a
2,277,897,600-byte table.

`run_exact_five_mitm.py` answers only the requested five-fault question. A
minimal five-fault witness has a fault with nonzero logical component. The free
32-element physical translation action moves that physical fault to cell zero.
Stim sometimes merges identical signatures arising at several physical
locations, so the anchor construction inspects every merged circuit location.
For each complete anchor, the remaining four effects are divided into two
pairs. Two sorted pair-syndrome groups are compatible precisely when

\[
D(p_1)\oplus D(p_2)=D(a),
\qquad
O(p_1)\oplus O(p_2)\oplus O(a)\neq 0,
\]

with all five effect indices distinct. Exhausting every anchor is therefore an
exact five-fault UNSAT certificate, given the separately saved exact lower
bound of five.

The optional `run_bounded_sat.py`, `run_maxsat_distance.py`, and
`run_xor_sat.py` provide independent SAT formulations. They require
`python-sat` and/or `pycryptosat`; the NumPy meet-in-the-middle certificate is
the primary path and does not require those optional packages.

## Current certified results

| Experiment | Basis | Exact result |
| --- | --- | --- |
| Guarded noisy-middle-round, CNOT faults | `X` | `d_fault = 6` |
| Guarded noisy-middle-round, CNOT faults | `Z` | `d_fault = 6` |
| Three-round full memory | `X` | `d_fault = 6` |
| Three-round full memory | `Z` | `d_fault = 6` |

For each full-memory basis, exact direct and pair meet-in-the-middle searches
exclude mechanisms of cardinality one through four. The translation-anchored
five-fault search then exhausts 262 complete anchors, 51,770,400 pair records,
and 50,913,624 pair-syndrome groups without finding a witness. A separately
saved Stim search supplies a concrete six-fault mechanism. The lower and upper
bounds therefore meet at exactly six. In accordance with the target for this
study, no attempt is made to exclude six-fault mechanisms.

## Useful commands

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.stim_fault_distance.run_validation \
  --schedule \
  searches/bb64_simultaneous_basis_search/schedule_fault_search/baseline_schedule.json \
  --rounds 3 --shots 256 \
  --output searches/bb64_simultaneous_basis_search/stim_fault_distance/results/validation.json

.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.stim_fault_distance.run_direct_certificate \
  --schedule \
  searches/bb64_simultaneous_basis_search/schedule_fault_search/baseline_schedule.json \
  --experiment memory --rounds 3 --basis both --fault-model full \
  --max-faults 4 \
  --output searches/bb64_simultaneous_basis_search/stim_fault_distance/results/direct4.json

.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.stim_fault_distance.run_exact_five_mitm \
  --schedule \
  searches/bb64_simultaneous_basis_search/schedule_fault_search/baseline_schedule.json \
  --experiment memory --rounds 3 --basis X --fault-model full --workers 12 \
  --output searches/bb64_simultaneous_basis_search/stim_fault_distance/results/exact5-X.json
```

All long runs write atomic JSON checkpoints and print periodic progress. Raw
results remain ignored; compact certificates and conclusions can be promoted
to tracked artifacts after validation.
