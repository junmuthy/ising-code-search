# `run_001_weight8_60s`: two-batch BB64 basis

The follow-up found two all-weight-eight logical bases with two internally
disjoint batches.  Candidate `0` is preferred because it retains the same
ordered logical-class basis as the earlier three-batch result; only its
stabilizer dressings change.

## Preferred candidate

| Property | Result |
|---|---|
| Code | `[[64,8,8]]` Liang--Chen BB code |
| CSS presentation | `H_X = H_Z` |
| Check rank | `28` for each CSS type |
| Check weights | all `8` |
| Logical Z weights | all `8` |
| Logical X weights | all `8` |
| Z batch sizes | `4+4` |
| X batch sizes | `4+4` |
| Logical translations | regular `C_4 x C_2` |
| Transversal Hadamard | logical H plus `(0 5)(1 4)(2 7)(3 6)` |

The exact Z cover is

\[
\{0,3,4,7\}\;\sqcup\;\{1,2,5,6\}.
\]

With

\[
p=(0\;5)(1\;4)(2\;7)(3\;6),
\]

the saved basis uses `X_i` on exactly the same physical support as `Z_{p(i)}`.
The corresponding exact X cover is therefore

\[
\{1,2,5,6\}\;\sqcup\;\{0,3,4,7\}.
\]

Every logical occurs exactly once in each cover, and representatives within
each displayed batch have no physical support overlap.  Thus the refinement
does not relax X disjointness.

The physical grid generators induce

\[
x=(0\;2\;4\;6)(1\;3\;5\;7),
\qquad
y=(0\;1)(2\;3)(4\;5)(6\;7),
\]

on both the X and Z logical bases.  These commuting actions are regular and
transitive on the eight `C_4 x C_2` sites.

## Complete solver outcome

All eight deduplicated affine logical-class candidates were tested at logical
weight eight.  The one-batch phase returned `unsat` for all eight candidates.
The two-batch phase returned `sat` for candidates `0` and `1`, and `unsat` for
candidates `2` through `7`.  There were no timeouts.  Although the configured
limit was 60 seconds per case, the complete 16-case run finished in about
32.5 seconds because every exact result returned early.

The one-batch result is an exact negative within this search boundary:

- every logical representative is required to have weight eight;
- the eight saved affine class candidates are covered;
- non-affine grid folds and heavier representatives are not covered.

For one batch, the weight-eight condition makes disjointness equivalent to an
exact partition of all 64 data qubits, which is the stronger formulation used
by the solver.

## Files

- `basis_batches2_candidate0.npz`: preferred complete matrix artifact;
- `basis_batches2_candidate1.npz`: second independently found witness;
- `results.json`: exact per-case outcomes and the stabilizer-dressing witnesses;
- `summary.json`: compact run summary;
- `checkpoint.json`: resumable full state;
- `../../search.py`: reproducible solver.

The preceding three-batch artifact remains untouched in
`../../../results/run_003_low_weight`.
