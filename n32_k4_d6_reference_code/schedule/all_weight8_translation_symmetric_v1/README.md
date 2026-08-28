# All-weight-eight translated presentation of `[[32,4,6]]`

## Status

This directory preserves a new stabilizer presentation of the previously saved
folded `[[32,4,6]]` code.  It does not replace or modify the earlier minimum,
20-layer, or 24-layer presentations.

The underlying stabilizer spaces, logical operators, distance, translation
`T`, and ZX fold `P` are unchanged.  Only the measured generating frame and
its syndrome schedule are new.

## Presentation

Each CSS type is measured with 16 checks arranged into four complete `C4`
orbits:

\[
(0,1,2,3),\quad(4,5,6,7),\quad(8,9,10,11),\quad(12,13,14,15).
\]

All checks have weight eight.  The 16 rows have rank 14 and exactly two saved
dependencies:

\[
S_0+S_2+S_4+S_6+S_8+S_{10}+S_{12}+S_{14}=0,
\]

\[
S_1+S_3+S_5+S_7+S_9+S_{11}+S_{13}+S_{15}=0.
\]

Consequently the XOR of all 16 rows is zero.  Every data qubit participates in
an even number of checks of each CSS type: 24 qubits have degree four, four
have degree two, and four have degree six.  This retains the even-syndrome
parity used by one-round STAR postselection.

## Schedule

The translated Tanner edges divide into 32 free `C4` orbits.  A saved exact
eight-coloring assigns each orbit to one layer, so every one-type layer:

- contains 16 CNOTs;
- contains no data or ancilla collision;
- is fixed as a set by the physical `C4` translation.

The safe full cycle executes eight `Z` layers followed by eight `X` layers.
The `Z` layers are the `P`-images and time reverse of the `X` layers.  Therefore
the full 16-layer circuit is exchanged with itself by the ZX fold plus time
reversal.  Sequential CSS extraction also guarantees clean cross-ancilla
back-action.

| Quantity | Value |
| --- | ---: |
| Independent checks per CSS type | 14 |
| Measured checks per CSS type | 16 |
| Dedicated ancillas | 32 |
| Reusable ancillas with sequential reset | 16 |
| Check weight | 8 |
| CNOTs per full cycle | 256 |
| CNOT depth per CSS type | 8 |
| Safe full-cycle CNOT depth | 16 |
| Combined collision-only lower bound | 12 |

Preparation, measurement, reset, and physical routing layers are not included
in the CNOT depth.

## Validation and remaining caveat

`build_and_verify.py` reconstructs the presentation from the saved survivor and
independently verifies with qLDPC:

\[
n=32,\qquad k=4,\qquad d_X=d_Z=6.
\]

It also verifies equality with the saved `H_X/H_Z` row spaces, the two
dependencies, CSS orthogonality, translation invariance, even data degree,
edge coverage, collision freedom, clean cross-ancilla back-action, and folded
time symmetry.

Circuit fault distance under hook faults remains uncertified.  This is a valid
ideal syndrome circuit and an improved scheduling presentation, but it still
requires the separate Stim/symbolic fault search used for the bicycle-chain
schedule.

## Files

- `presentation.json`: complete `X/Z` check supports and presentation metadata.
- `safe-fold-symmetric-16-layer.json`: complete machine-readable CNOT schedule.
- `validation.json`: independent structural and qLDPC validation.
- `build_and_verify.py`: reproducible builder and verifier.
