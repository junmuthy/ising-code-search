# Optimal-depth `[[160,32,6]]` syndrome schedule

## Result

The selected simultaneous bare-ancilla schedule has provably minimum CNOT
depth `12`.  Every one of its twelve layers contains exactly `96` CNOTs, so it
also attains the minimum possible peak layer load at that depth.

| Property | Value |
|---|---:|
| Stabilizer weight | exactly `9` |
| X checks / Z checks | `64 / 64` |
| Dedicated syndrome ancillas | `128` |
| CNOTs per syndrome round | `1,152` |
| CNOT depth | `12` |
| CNOTs in every layer | `96` |
| Translation symmetry | `C4 x C8` |
| Guarded CNOT-only circuit fault distance | `5 <= d_fault <= 6` in both bases |

The fault-distance lower bound requested for this analysis therefore passes.
The present certificate does not distinguish exact distance five from exact
distance six.

## Why depth twelve is optimal

Every qubit in physical block `4` (zero based) is incident on six X-check and
six Z-check CNOTs.  A data qubit can participate in at most one CNOT per
layer, so any simultaneous one-ancilla-per-check extraction needs at least
`12` CNOT layers.  The saved schedule attains that bound.

A round contains `1,152` CNOTs, so a depth-12 schedule averages `96` CNOTs per
layer.  The selected schedule has exactly `96` in every layer and hence also
minimizes the peak layer load.

The schedule is invariant under all `C4 x C8` translations, collision free,
and mapped to itself by the ZX fold plus time reversal.  Exact tests of all
overlapping X/Z check pairs verify even cross-ancilla back-action parity.

## Compact schedule

Write `(f,b;u,v)` for the translation orbit containing, for every group
element `g`, the interaction from check family `f` at `g` to data block `b` at
`g+(u,v)`.  Coordinates are modulo `(4,8)`.  Every displayed orbit represents
`32` parallel CNOTs.

| Layer | X edge orbits | Z edge orbits |
|---:|---|---|
| 1 | `(0,0;0,0)`, `(1,1;0,0)` | `(1,4;1,0)` |
| 2 | `(0,4;0,0)` | `(0,1;0,0)`, `(1,3;3,2)` |
| 3 | `(1,3;2,4)` | `(0,4;1,2)`, `(1,2;3,7)` |
| 4 | `(0,2;2,4)`, `(1,3;0,0)` | `(0,4;1,5)` |
| 5 | `(1,4;0,0)` | `(0,0;3,4)`, `(1,2;3,4)` |
| 6 | `(0,2;3,0)`, `(1,4;2,0)` | `(0,0;3,7)` |
| 7 | `(0,0;3,3)` | `(0,1;3,2)`, `(1,4;2,4)` |
| 8 | `(0,0;3,6)`, `(1,1;3,6)` | `(1,4;0,0)` |
| 9 | `(0,4;1,1)` | `(0,1;2,0)`, `(1,3;0,0)` |
| 10 | `(0,4;1,4)`, `(1,1;3,3)` | `(1,3;2,0)` |
| 11 | `(0,2;0,0)`, `(1,3;3,0)` | `(0,4;0,0)` |
| 12 | `(1,4;1,6)` | `(0,0;0,0)`, `(1,2;0,0)` |

For an X entry, apply `CX(X ancilla, data)`.  For a Z entry, apply
`CX(data, Z ancilla)`.  Prepare X ancillas in `|+>` and Z ancillas in `|0>`,
then measure them in X and Z respectively.

## Circuit-fault analysis

The certified experiment places one noisy syndrome round between two ideal
guard rounds.  Every one of the fifteen nonidentity two-qubit Pauli outcomes
following a CNOT is an elementary fault.  Preparation, measurement, boundary,
and idle noise are disabled in this isolated scheduling test.

Stim produces `9,184` distinct detector/logical fault effects in either
logical memory basis.  An exact pair meet-in-the-middle calculation checks
all `42,168,336` pairs in each basis and excludes logical failures containing
one through four faults.  This proves

```text
d_fault^X >= 5,   d_fault^Z >= 5.
```

Explicit six-fault witnesses are made by inserting a data-only Pauli outcome
after the final CNOT touching each qubit of a stored weight-six logical.  Thus
both circuit distances are at most six.  The resulting certified interval is

```text
5 <= d_fault^X,d_fault^Z <= 6.
```

This is a CNOT-hook analysis, not a full phenomenological or hardware-noise
threshold calculation.  Schedule depth and peak utilization are proven
optimal; global optimization of fault distance over every depth-12 ordering
has not been claimed.

## Files and verification

- `optimal_depth12_schedule.json`: all `1,152` indexed CNOTs.
- `build_and_verify.py`: canonical reconstruction and structural checks.
- `search_schedule.py`: Z3 search for depth-12, balanced, clean schedules.
- `stim_circuit.py`: guarded-bulk Stim circuit construction.
- `circuit_fault_certificate.json`: lower and upper fault-distance evidence.
- `export_fault_effects.py` and `pair_certify.cpp`: complete pair certificate.
- `verify_certificate.py`: schedule, checksum, and six-fault-witness checks.

Run the inexpensive verification with:

```bash
../.venv/bin/python schedule/build_and_verify.py \
  --verify schedule/optimal_depth12_schedule.json
../.venv/bin/python schedule/verify_certificate.py
```

Recompute the full one-through-four fault exclusion (about `1.1 GB` temporary
RAM for its pair table) with:

```bash
../.venv/bin/python schedule/verify_certificate.py --full-pair
```
