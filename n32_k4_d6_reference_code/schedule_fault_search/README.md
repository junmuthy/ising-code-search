# Alternative 12-layer schedule fault search

This directory searches CNOT orderings for the saved `[[32,4,6]]` code without
changing its checks, logical grid, or static distance. The strict search space
preserves:

- physical-`C4`-invariant layers;
- the selected ZX fold and time-reversed `X`/`Z` schedules;
- collision-free use of 32 dedicated syndrome ancillas;
- exact even cross-ancilla backaction parity;
- the optimal 12-CNOT-layer depth.

## Staged search

Each candidate is first run through Stim's CNOT-only, three-round guarded bulk
circuit in the `X` basis. A concrete witness of cardinality four or less rejects
the schedule immediately. Only heuristic survivors receive exact exclusion of
one through four faults. Survivors advance through bulk `Z`, CNOT-only
three-round memory in both bases, and finally full-noise three-round memory in
both bases. The exact four-fault step sorts all two-fault detector signatures
and searches for disjoint pairs with canceling detectors and a nonzero logical
effect.

The default pilot samples schedules at orbit-color Hamming distance at least
four. It is a diverse sample, not an exhaustive enumeration. Set
`--minimum-color-changes 1` for literal enumeration.

## Pilot command

From `/home/judah_unmuth/gala-code-search`:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.schedule_fault_search.search \
  --output results/n32-k4-d6-schedule-fault-search-v1 \
  --fold-indices 0 \
  --max-candidates 1000 \
  --minimum-color-changes 4 \
  --checkpoint-every 100 \
  --heartbeat-seconds 30
```

Interrupted searches can be continued by adding `--resume` with the identical
arguments. Compact candidate records are appended to `candidates.jsonl`, while
`progress.json` and `summary.json` report cumulative histograms. Complete
schedule JSON is saved only under `survivors/` when both CNOT-only bulk bases
and both full-noise three-round memory bases have an exact lower bound of five.
Use `--exclude-records PATH/TO/candidates.jsonl` when beginning a new pipeline
version so the sampled color neighborhoods from an older run are not repeated.

## Full-noise finalist verification

```bash
.venv/bin/python -m n32_k4_d6_reference_code.schedule_fault_search.verify_survivor \
  --schedule results/n32-k4-d6-schedule-fault-search-v1/survivors/ID.schedule.json \
  --output results/n32-k4-d6-schedule-fault-search-v1/survivors/ID.full-verification.json \
  --memory-rounds 3,18 \
  --heartbeat-seconds 30
```

This runs the full preparation, measurement, CNOT, and idle noise model for the
bulk circuit and the three- and eighteen-round memories in both bases. Before
exact four-fault exclusion, the verifier estimates the compact two-fault table.
The default one-GiB ceiling admits the bulk and three-round calculations but
records the eighteen-round case as deferred instead of attempting a tens-of-
gigabytes allocation. A later temporal/windowed certifier is needed for that
paper-scale lower bound.

The exact temporal certifier is now available separately. It proves that every
elementary signature spans at most one detector interval and uses endpoint
parity plus minimal-witness connectedness to reduce every witness of at most
four faults to three adjacent detector-time slices:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.schedule_fault_search.temporal_certificate \
  --schedule PATH/TO/SURVIVOR.schedule.json \
  --three-round-records PATH/TO/candidates.jsonl \
  --output PATH/TO/SURVIVOR.temporal-r18.json \
  --rounds 18 \
  --heartbeat-seconds 30
```

Normalized time translation reduces the 17 windows of an eighteen-round
memory to five exact window classes per basis. The saved three-round five-fault
witness supplies an upper bound because all of its faults occur during
preparation and the first syndrome round, so it remains a logical witness when
ideal rounds are appended.

## Versioned final schedule

`survivor_a381a067d3e82750.json` contains the fold and 32 orbit colors needed to
reconstruct the successful schedule without relying on ignored search-result
directories. Materialize and validate all 256 gates with:

```bash
.venv/bin/python -m n32_k4_d6_reference_code.schedule_fault_search.materialize_survivor \
  --output /tmp/a381a067d3e82750.schedule.json
```
