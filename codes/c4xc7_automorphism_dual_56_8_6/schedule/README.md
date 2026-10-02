# Depth-optimal BB56 syndrome schedule

This directory records the selected simultaneous bare-ancilla syndrome
schedule for the parent `[[56,8,6]]` automorphism-dual BB code.  It is not the
Table-II schedule of the STAR bicycle-chain paper: the parent code has
different `a` and `b` polynomials and a nonidentity ZX fold.

## Compact schedule

Write the two data halves as `L` and `R`, and place one X-check ancilla and one
Z-check ancilla at every `g` in `C4 x C7`.  An entry such as `L x*y^2` means
the data qubit `L[g+(1,2)]`, with arithmetic modulo `(4,7)`.

| Layer | X-check interaction | Z-check interaction |
| ---: | --- | --- |
| 1 | `R x` | `L x^-1*y^2` |
| 2 | `L x` | `R x^-1*y^-2` |
| 3 | `L 1` | `R y^-1` |
| 4 | `R y^-1` | `L 1` |
| 5 | `R 1` | `L y` |
| 6 | `L y` | `R 1` |
| 7 | `L x*y^2` | `R x^-1` |
| 8 | `R x*y^-2` | `L x^-1` |

For every cell `g`, the X entry is the CNOT

```text
CX(X-check ancilla g, listed data qubit),
```

and the Z entry is

```text
CX(listed data qubit, Z-check ancilla g).
```

Prepare the 28 X ancillas in `|+>` and the 28 Z ancillas in `|0>`, execute the
eight layers, and measure them in the X and Z bases respectively.

## Optimality and structural checks

Each of the 56 displayed stabilizers has weight eight, so any one-ancilla-per-
check extraction has CNOT depth at least eight.  The saved schedule attains
that bound.  Every layer is a 56-gate perfect matching: every data qubit and
every dedicated syndrome ancilla participates exactly once.  One round uses
448 CNOTs and 56 syndrome ancillas.

The layers are translation invariant over `C4 x C7`, have no CNOT collisions,
and have even cross-ancilla back-action parity for every overlapping X/Z check
pair.  The Z order is the time-reversed image of the X order under

```text
data:  (half,x,y) -> (1-half,-x,-y)
check: (x,y)      -> (-x,-y).
```

`optimal_depth8_schedule.json` contains all 448 indexed CNOTs.  Its stable
schedule identifier is `8c613cb8afa703e4`.  Rebuild it from the eight orbit
colors and verify all structural claims with:

```bash
python build_and_verify.py --verify optimal_depth8_schedule.json
```

## Circuit-fault certificate

The schedule was also tested in an isolated guarded-bulk experiment: one noisy
syndrome round lies between two ideal guard rounds, and every nonidentity
two-qubit Pauli outcome after a CNOT is an elementary fault.

Exact enumeration excludes one through five faults.  A concrete six-fault
Stim witness has zero detector syndrome and nonzero logical observable, hence

```text
d_fault^X = d_fault^Z = 6
```

for this CNOT-only model.  Counts, methods, symmetry transfer, and the explicit
X-basis six-fault witness are stored in `circuit_fault_certificate.json`.
Validate the certificate metadata and the detector/logical cancellation of
that witness with `python verify_certificate.py`.
This certificate does not claim a full hardware-noise threshold or optimize a
continuous neutral-atom transport trajectory.
