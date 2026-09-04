# Paper Table-II syndrome schedule

The Appendix-B schedule specializes without collisions at `ell=2`, `m=7`.
Each of its eight layers is a perfect matching containing 14 X-check CNOTs
and 14 Z-check CNOTs.  The syndrome types always touch opposite data halves.

| Quantity | Value |
|---|---:|
| Check weight | `8` |
| Displayed X checks | `14` |
| Displayed Z checks | `14` |
| Independent checks per type | `12` |
| Dedicated X ancillas | `14` |
| Dedicated Z ancillas | `14` |
| Total syndrome ancillas | `28` |
| Simultaneous CNOT depth | `8` |
| CNOTs per layer | `28` |
| CNOTs per full round | `224` |

Symbolic propagation verifies even X-to-Z ancilla backaction for every
overlapping check pair.  The schedule is therefore a clean measurement of
the intended stabilizers.  Its certified circuit fault distance is recorded
in `../stim_fault_distance/RESULTS.md`.
