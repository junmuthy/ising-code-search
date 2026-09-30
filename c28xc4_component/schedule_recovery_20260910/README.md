# Recovered syndrome-schedule analysis for `[[112,16,7]]`

Recovered and independently checked on 2026-09-10 following an interrupted
Emacs/Codex session. The code is the exact-self-dual `C28 x C4` component in
the parent directory. The stabilizer row space, logical fibres, and static
distance seven are unchanged by the choice of measured check basis.

## Established results

| Schedule | CNOT layers for both CSS types | Circuit fault distance, X memory | Circuit fault distance, Z memory |
| --- | ---: | ---: | ---: |
| Original saved one-type ordering, completed sequentially as Z then X | `32` | `2` | `2` |
| Recovered compact-window schedule | `24` | `3` | `3` |
| Recovered candidate selected by the hook-error search | `32` | `4` | `4` |

All distances in this table are exact **in the guarded bulk CNOT-only model**:
one noisy syndrome round between two ideal syndrome rounds, ideal data
preparation and final readout, and any of the 15 nonidentity two-qubit Pauli
faults after each middle-round CNOT. The table is not a claim about full
preparation/readout/idle noise, arbitrary temporal boundaries, or long-running
memory. Static code distance seven is a different quantity.

Every schedule measures 48 independent weight-16 X checks and 48 independent
weight-16 Z checks, using 96 dedicated ancillas, 112 data qubits, and 1,536 CNOTs
per round. The new bases have maximum data degree eight per CSS type; the old
saved basis has maximum degree nine. These are measurements with bare
ancillas, without flags or cat states. Depth counts CNOT layers only and
assumes the required connectivity is available.

The 24-layer construction uses a 16-color single-type Tanner-edge coloring
`c(row,data)` with incident colors spanning at most seven at every data qubit.
Z gates occur at `c`; X gates occur at `c+8`. Consequently every Z gate on a
given data qubit precedes every X gate on that qubit. This ensures clean
cross-ancilla propagation as well as collision freedom. Layer loads are
`48` for eight layers, `96` for eight layers, and `48` for eight layers.
It reduces the sequential CNOT depth by 25% and improves the assessed fault
distance from two to three.

The 32-layer hook-search candidate gives a better fault distance, four. The
old scratch file records `upper: 5`: that was a heuristic five-fault witness,
not a lower-bound certificate. Recovery analysis exhaustively excluded all
failures through three faults, then found four-fault witnesses in both bases.
Both four-fault witnesses flip logical fibres 8 through 15 with zero detectors.
The exact search enumerates 83,301,778 distinct-index pairs from 12,908
distinct fault signatures in each memory basis. Hashes only index pairs;
full detector signatures are compared before accepting a collision.

## What did not survive as a verified result

No verified depth-18 schedule was found among the saved code artifacts,
temporary search files, or project staging directories. Eighteen was a search
target; the earlier solver run returned `unknown` after its time limit.

The depth-16 scratch edge coloring in `recovered/c28_joint_schedule.json` is
collision-free but fails 148 cross-ancilla parity checks, so it is not an
accepted syndrome-extraction circuit. The sampled fixed-X searches and timed
out general searches do not rule out clean schedules at depths 16 through 23.
Weight 16 gives a CNOT-depth lower bound of 16 in this measurement model;
24 is the best verified full-round depth recovered here, not a proven optimum.

The check bases and schedules are not required to be translation-invariant.
The code's logical `C4 x C4` translations and transversal Hadamard remain
properties of its stabilizer space.

## Files and reproduction

- `recovered_depth24.json`: validated checks, CNOT layers, and resource counts.
- `recovered_depth32_hook_candidate.json`: validated checks and the distance-four schedule.
- `original_sequential32.json`: the explicit baseline constructed from the originally saved single-type ordering.
- `*_X.stim`, `*_Z.stim`: guarded bulk circuits for the three schedules.
- `*_X.json`, `*_Z.json`: initial fault audits and explicit witnesses. For the hook-search candidate, combine these initial bounds with `pair_results.json`.
- `pair_results.json`: exhaustive-search result establishing distance four.
- `pair_witness_validation.json`: independently reconstructed four-fault mechanisms.
- `RESULTS.json`: consolidated final conclusions.
- `analyze.py`: self-contained schedule validation, order-preserving compaction, circuit construction, signature export, and preliminary fault audit.
- `verify_pair_results.py`, `pair_certify.cpp`: exact four-fault certificate reconstruction.
- `component_code.py`: snapshot of the parent's code construction.
- `recovered/`: unchanged copies of the surviving scratch records and scripts for provenance and resumption. Treat their old claim fields as provisional; some scratch tools have hard-coded `/tmp` paths and overly broad optimality labels. Use the root validation tools for accepted results.

With the `gala-code-search` virtual environment, from this directory:

```bash
../../.venv/bin/python analyze.py
../../.venv/bin/python verify_pair_results.py
```

`analyze.py` checks the stabilizer row spaces against the canonical construction,
all Tanner edges, qubit collisions, and every cross-ancilla parity. It also
checks that earliest-start compaction preserving each qubit's complete CNOT
order does not shorten any of these three schedules.

For reconstruction of the ideal circuits, fault signatures, and initial bounds:

```bash
../../.venv/bin/python analyze.py --audit
```

For the complete exhaustive fault search:

```bash
../../.venv/bin/python verify_pair_results.py --full
```

The full check builds the C++ program with `g++`, uses about 2 GB RAM, and
processes one memory basis at a time. The default verifier regenerates all
signatures, checks their saved hashes, and independently explains the four
faults; `--full` additionally redoes the exhaustive lower-bound computation.
Dependencies are Python, NumPy, Stim, and (for `--full`) a C++20 compiler.
No web search or new external theorem is needed to reproduce these results.

## Search continuation

The two distinct starting points are the verified depth-24/distance-three
schedule and the depth-32/distance-four schedule. Further work should jointly
optimize the measured check basis, hook ordering, and depth. A useful next
target is a shorter schedule retaining circuit distance at least four,
followed by attempting a certified lower bound of five. A depth-18 or
fault-distance-five claim requires a new explicit schedule and corresponding
verification.
