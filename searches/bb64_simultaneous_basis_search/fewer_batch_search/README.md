# BB64 one-/two-batch logical-basis search

This follow-up asks whether the preferred self-dual `[[64,8,8]]` BB code can
retain its exact logical Hadamard map while reducing the STAR injection cover
from three batches to one or two.

The search deliberately preserves all of the current class-level structure:

- the eight logical classes and their regular `C_4 x C_2` action;
- the permutation ZX pairing
  `(0 5)(1 4)(2 7)(3 6)`;
- the representative-level rule that physical transversal Hadamard sends each
  `Z_i` support exactly to the support of `X_{p(i)}`;
- disjointness for both Z and X batches.  The X batches are the images of the Z
  batches under `p`, so this introduces no independent relaxation.

Only stabilizer dressings of the eight saved logical-class bases are changed.
All eight deduplicated affine candidates from `../results/run_002` are tested.
One-batch cases are exhausted before two-batch cases.

## Default minimum-weight formulation

Every representative is constrained to weight eight, the code distance.  For
one batch, eight disjoint weight-eight supports on 64 qubits necessarily
partition all data qubits, and the solver uses this exact-cover equality.  For
two batches, a Boolean batch color is assigned to every logical and supports
with the same color are constrained not to overlap.  Each batch contains at
least two logicals.

Consequently:

- `sat` supplies a certified minimum-weight basis and saved NPZ artifact;
- `unsat` rules out that candidate and batch count at weight eight;
- `unknown` is a timeout, not a negative result;
- an all-weight-eight negative result does not rule out a heavier two-batch
  basis.

## Running and resuming

From the `gala-code-search` root:

```bash
.venv/bin/python -m searches.bb64_simultaneous_basis_search.fewer_batch_search.search \
  --output-dir searches/bb64_simultaneous_basis_search/fewer_batch_search/results/run_001_weight8_60s \
  --seconds-per-case 60
```

The process emits a heartbeat every 15 seconds while Z3 is active and writes
`checkpoint.json` after every case.  Resume an interrupted run with the same
arguments plus `--resume`.  Existing result directories are otherwise never
overwritten.
