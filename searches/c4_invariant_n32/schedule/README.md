# `[[32,4,5]]` syndrome schedule

## Status

This is a verified collision-free schedule for the saved direct `C_4`-invariant
`[[32,4,5]]` code.  It uses a better-balanced basis of the same stabilizer
row space; therefore the code, logical representatives, and exact distance are
unchanged.

- `14` independent checks and one redundant check are measured per CSS type.
- Check weights are one weight `4`, two weight `6`, and twelve weight `8`.
- Every data qubit participates in either two or four measured checks per type.
- The redundant row is the XOR of the independent rows, so the parity of every
  nonzero syndrome is even.  This restores the syndrome-parity condition used
  by the one-round TMR test.
- `qldpc.codes.CSSCode(H,H)` verifies `[[32,4,5]]` exactly.

## Safe sequential schedule

For layers `L1` through `L8`, use the table entries as
`data q -> Z-check ancilla S` CNOTs.  Then repeat the same eight layers as
`X-check ancilla S -> data q` CNOTs.  This gives `16` CNOT layers, excluding
ancilla preparation and readout.

| Layer | Collision-free check--data pairs |
| --- | --- |
| `L1` | `S0-q18 S1-q14 S2-q10 S3-q7 S4-q26 S5-q5 S6-q3 S7-q6 S8-q29 S9-q21 S10-q17 S11-q27 S12-q23 S14-q22` |
| `L2` | `S0-q20 S1-q5 S2-q28 S3-q23 S4-q25 S5-q9 S6-q13 S7-q4 S8-q21 S9-q30 S10-q19 S11-q22 S12-q15 S13-q8 S14-q10` |
| `L3` | `S0-q30 S1-q24 S2-q16 S3-q15 S4-q18 S5-q10 S6-q22 S7-q11 S8-q27 S9-q14 S10-q21 S11-q29 S12-q6 S14-q9` |
| `L4` | `S0-q23 S1-q6 S2-q1 S4-q29 S5-q2 S6-q20 S7-q24 S8-q8 S9-q5 S10-q30 S11-q17 S12-q19 S13-q10 S14-q13` |
| `L5` | `S0-q9 S1-q7 S2-q5 S4-q28 S5-q12 S6-q0 S7-q8 S8-q18 S9-q19 S10-q25 S11-q1 S12-q30 S13-q22 S14-q26` |
| `L6` | `S0-q8 S1-q15 S2-q11 S4-q21 S5-q17 S6-q14 S8-q0 S9-q3 S10-q2 S11-q13 S12-q4 S13-q24 S14-q25` |
| `L7` | `S0-q26 S1-q27 S2-q24 S3-q31 S4-q1 S5-q13 S6-q15 S8-q4 S9-q25 S10-q18 S11-q6 S12-q29 S13-q20 S14-q11` |
| `L8` | `S0-q31 S1-q12 S2-q19 S4-q4 S5-q20 S6-q23 S7-q26 S8-q14 S9-q9 S10-q1 S11-q11 S12-q16 S13-q27 S14-q17` |

The machine-readable JSON is the authority if the generator is rerun.

## Eight-layer simultaneous target

The balanced checks have maximum check weight `8` and maximum combined `X/Z`
data degree `8`, so the incidence graph admits an optimal collision-free
eight-layer coloring.  That fact alone does not make it a valid simultaneous
CSS syndrome circuit: the relative ordering of overlapping `X`- and `Z`-check
CNOTs must also cancel ancilla back-action, and hook faults must preserve the
desired circuit fault distance.  Those conditions have not yet been certified.

Thus `16` is the current safe CNOT depth; `8` is the schedule-search target.

## Clifford and STAR caveats

The code has exact transversal Hadamard, transversal inter-block CNOT, four
disjoint weight-seven logical supports, and the required logical `C_4` shift.
Plain `S^tensor32` is not code-preserving because the stabilizer contains
weight-six elements.  A fixed Pauli-frame correction repairs it:

`Z({1,2,4,5,6,7,14,15,16,24}) S^tensor32`

acts as logical `S` on each of the four selected logical qubits.

The remaining open certification is circuit-level: effective fault distance,
memory/CNOT logical error rates, and the TMR postselection simulation.  The
code distance is `5`, rather than the bicycle-chain code's `6`.
