# Alternative 12-layer schedule-search results

## Outcome

The alternate-fold search found a complete survivor:

\[
\boxed{
[[32,4,6]],\qquad
d_{\mathrm{fault}}^X=d_{\mathrm{fault}}^Z=5
}
\]

for the full circuit-level noise model with both three and eighteen syndrome
rounds. The saved schedule has identifier `a381a067d3e82750` and uses ZX fold
index one.

All candidates preserve the saved `[[32,4,6]]` code, its four disjoint logical
supports, the physical `C4` action, an allowed ZX fold, fold plus time reversal,
collision-free simultaneous extraction, and exact clean cross-ancilla
backaction.

## First pilot

The first run sampled 140 alternative schedules for fold zero, each at orbit-
color Hamming distance at least four from the saved schedule, its time reverse,
and the other sampled schedules.

| Outcome | Count |
| --- | ---: |
| Bulk `X` heuristic witness with 3 faults | 19 |
| Bulk `X` heuristic witness with 4 faults | 118 |
| Exact CNOT-only bulk distance 5 in both bases | 3 |

The three bulk-distance-five schedules are:

- `7dfe9cb851d3082c`;
- `71d1e1b848c12b6e`;
- `7dc7cdb9cbf77f2b`.

For `7dfe9cb851d3082c`, exact pair meet-in-the-middle exclusion proves no set of
four faults causes an undetected logical error in either full-noise guarded bulk
circuit, while Stim supplies five-fault witnesses. Thus

\[
d_{\mathrm{fault}}^{\mathrm{bulk},X}
=d_{\mathrm{fault}}^{\mathrm{bulk},Z}=5.
\]

Its full-noise three-round memory has exact distance four in both bases. The
other two schedules have the same exact three-round result. These mechanisms
use CNOT faults in more than one extraction round and are invisible to the
single-noisy-cycle guarded bulk experiment.

## Memory-aware follow-up

The second run sampled 500 additional schedules, excluding all 140 color
neighborhoods from the first pilot. Its staged acceptance pipeline was:

1. CNOT-only guarded bulk `X` and `Z`, exact through four;
2. CNOT-only three-round memory `X` and `Z`, exact through four;
3. full-noise three-round memory `X` and `Z`, exact through four.

| Rejection stage | Count |
| --- | ---: |
| Bulk `X`, heuristic 2-fault witness | 17 |
| Bulk `X`, heuristic 3-fault witness | 66 |
| Bulk `X`, heuristic 4-fault witness | 413 |
| Bulk `X`, exact 4-fault witness missed by the heuristic | 2 |
| CNOT-only three-round memory `X`, heuristic 4-fault witness | 2 |
| Full-memory survivors | 0 |

The two schedules reaching the memory stage passed exact bulk screening in both
bases but failed immediately at four faults in the CNOT-only `X` memory. No
candidate needed the full-noise memory stage.

Across both runs, five of 640 completed schedules reached exact bulk distance
five, and all five failed the three-round memory requirement at four faults.
This is a negative sampled result, not an exhaustive no-go theorem for all
strictly `C4`-invariant depth-12 schedules.

## All-fold search and final survivor

The final run sampled folds 1 through 23 in round-robin order, using the full
memory-aware pipeline. It stopped at candidate 346 when fold one produced the
first complete survivor. At that point fold one had 16 sampled schedules and
every other alternate fold had 15.

| Outcome before stopping | Count |
| --- | ---: |
| Bulk `X`, heuristic 2-fault witness | 15 |
| Bulk `X`, heuristic 3-fault witness | 61 |
| Bulk `X`, heuristic 4-fault witness | 268 |
| Bulk `X`, exact 4-fault witness missed by the heuristic | 1 |
| Full-memory survivor | 1 |

The survivor retains all desired structural properties:

- 32 data qubits and four disjoint logical qubits;
- uniformly weight-eight `X` and `Z` stabilizer presentations;
- 32 dedicated syndrome ancillas;
- 256 CNOTs in the exact minimum of 12 simultaneous layers;
- `C4`-invariant layers and the logical four-cycle;
- fold plus time reversal and clean cross-ancilla backaction;
- ZX duality through an involutive fold with logical pairing permutation
  `(0 2)(1)(3)`.

Independent qLDPC validation reproduces `[[32,4,6]]`, verifies every Tanner
edge exactly once, collision freedom, clean backaction, fold/time reversal, and
the depth-12 collision lower bound.

For the guarded bulk, CNOT-only three-round memory, and full-noise three-round
memory, both bases have a concrete five-fault upper-bound witness and exact
exclusion of all one- through four-fault logical mechanisms.

### Exact eighteen-round certificate

The full eighteen-round circuits have 648 detectors and 33,364 distinct fault
effects per basis. A monolithic four-fault pair table would be prohibitively
large, so the final certificate uses temporal locality:

1. every elementary detector signature spans at most one syndrome interval;
2. a minimal undetected logical witness is connected in detector overlap;
3. detector parity requires at least two effects at its earliest time and two
   at its latest time;
4. with at most four effects, temporal diameter three or greater would split
   those effects into disconnected endpoint groups.

Therefore every witness of at most four faults is contained in three adjacent
detector-time slices. The 17 windows reduce exactly to five normalized window
classes per basis. All ten classes exhaustively exclude cardinalities one
through four. The saved five-fault three-round witnesses occur entirely during
preparation and the first syndrome round, so they remain valid when ideal
rounds are appended. Consequently,

\[
\boxed{
d_{\mathrm{fault}}^{X}(r=18)
=d_{\mathrm{fault}}^{Z}(r=18)=5.
}
\]

The alternative-generator and flag-ancilla branches were not run because the
alternate ZX fold solved the bare-ancilla problem without adding qubits or
increasing circuit depth.

## Exact engine and checkpoints

The exact four-fault calculation represents every two-fault combination by its
detector and logical masks. Two disjoint pairs form a four-fault logical when
their detector masks agree and their logical masks differ. For the bulk circuit,
1,984 distinct effects produce

\[
\binom{1984}{2}=1{,}967{,}136
\]

pair records in a 55 MB table. The full three-round memory uses approximately
4,976 effects and a 347 MB table. The optimized bucket scan is cross-checked
against brute force on randomized small instances.

Candidate summaries are appended and flushed to `candidates.jsonl` after every
completed schedule. A newly enumerated schedule is also atomically saved as
`pending-candidate.json` before Stim screening, so future interruptions retain
the in-progress candidate. Console and JSON checkpoints report long heuristic,
table-build, sort, and scan phases.

## Saved data

- First pilot: `results/n32-k4-d6-schedule-fault-search-v1/`
- Memory-aware run: `results/n32-k4-d6-schedule-fault-search-memory-v2/`
- All-fold run: `results/n32-k4-d6-schedule-fault-search-all-folds-v3/`
- Exact first-survivor verification:
  `results/n32-k4-d6-schedule-fault-search-v1/survivors/7dfe9cb851d3082c.staged-full-bulk-r3.json`
- Final schedule:
  `results/n32-k4-d6-schedule-fault-search-all-folds-v3/survivors/a381a067d3e82750.schedule.json`
- Exact eighteen-round certificate:
  `results/n32-k4-d6-schedule-fault-search-all-folds-v3/survivors/a381a067d3e82750.exact-temporal-r18.json`
- Independent static and schedule validation:
  `results/n32-k4-d6-schedule-fault-search-all-folds-v3/survivors/a381a067d3e82750.static-schedule-validation.json`
