# BB64 simultaneous syndrome schedules

This directory constructs and screens bare-ancilla syndrome schedules for the
all-weight-eight simultaneous basis of the Liang--Chen `[[64,8,8]]` BB code.
The presentation has 32 displayed `X` rows and 32 displayed `Z` rows, each of
weight eight and rank 28.

## Exact depth-eight schedule family

Every data qubit meets four `X` checks and four `Z` checks. A collision-free
simultaneous bare-ancilla schedule therefore has the lower bound

\[
d_{\mathrm{CNOT}}\geq 4+4=8.
\]

The saved schedules attain this bound. Each of their eight layers is a perfect
matching containing 64 CNOTs and using all 64 data qubits and all 64 dedicated
syndrome ancillas. Consequently the entangling layers have no idle qubits.

`enumerate_schedules.py` uses the full 32-element twisted-torus translation
action. The 256 Tanner edges of one check type form eight free edge orbits. It
enumerates all `8! = 40,320` orbit colorings, identifies time reversal, and
requires:

- collision freedom for simultaneous `X` and `Z` extraction;
- the `Z` order to be the reverse of the `X` order under the identity ZX fold;
- even cross-ancilla backaction parity for every `X`/`Z` check pair;
- exactly one use of every Tanner edge.

The exhaustive identity-fold run found:

| Stage | Count |
| --- | ---: |
| Raw orbit colorings | 40,320 |
| Time-reversal representatives | 20,160 |
| Collision-free colorings | 4,608 |
| Clean depth-eight schedules | 32 |

The stable baseline is `baseline_schedule.json`, schedule ID
`67e327dc7cb35335`.

## Reproduction

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.enumerate_schedules \
  --output-dir \
  bb64_simultaneous_basis_search/schedule_fault_search/results/identity-fold-run-001
```

Generated search logs and complete schedules live under the ignored `results/`
tree. The stable baseline is kept outside that tree so downstream validation is
reproducible from tracked inputs.
