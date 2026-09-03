# Optimal simultaneous syndrome schedule

The cyclic presentation contains eleven displayed `X` checks and eleven
displayed `Z` checks.  Every check has weight eight.  Each physical data qubit
has total degree eight across the two check types, so any simultaneous
bare-ancilla extraction requires at least eight CNOT layers.

The exhaustive translation-invariant search considered all `8! = 40,320`
assignments of the eight Tanner-edge orbits to time layers.  After identifying
global time reversal, it tested 20,160 representatives and found 2,880
collision-free schedules satisfying exact even cross-ancilla backaction
parity.

The selected `baseline_schedule.json` therefore has provably optimal CNOT
depth:

| Quantity | Value |
| --- | ---: |
| Simultaneous CNOT depth | `8` |
| CNOTs per layer | `22` |
| CNOTs per round | `176` |
| Dedicated `X` ancillas | `11` |
| Dedicated `Z` ancillas | `11` |
| Total syndrome ancillas | `22` |
| Idle data qubits during CNOT layers | `0` |
| Idle syndrome ancillas during CNOT layers | `0` |

Every layer is a perfect matching.  The `Z` ordering is the time reverse of
the `X` ordering under the universal bicycle ZX fold
`L_i <-> R_{-i}`.  The resulting ancilla-to-ancilla propagation cancels with
even parity for every overlapping `X/Z` check pair.

## Circuit fault distance

For the saved baseline, exact meet-in-the-middle enumeration excludes every
undetected logical mechanism made from one through four elementary faults.
Stim finds explicit five-fault witnesses.  Consequently,

\[
d_{\mathrm{fault}}=5
\]

for both logical bases in both of the following experiments:

- one noisy CNOT-only extraction round between two ideal guard rounds;
- a three-round memory circuit with noisy preparation, CNOTs, measurements,
  and idle locations.

This certifies the selected schedule at the same circuit fault distance as the
accepted schedule for the `[[32,4,6]]` reference code.  The result proves
optimal interaction depth and certifies fault distance five; it does not claim
that fault distance six is impossible for every one of the other 2,879 clean
orderings.

