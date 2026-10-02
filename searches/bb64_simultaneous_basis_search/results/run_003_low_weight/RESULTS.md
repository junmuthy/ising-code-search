# `run_003_low_weight`: all-weight-eight simultaneous basis

The targeted refinement of the `\{2,3,5\}` batch succeeded.  Each of those
three logical classes admits a weight-eight representative, and the solver
found representatives that are mutually disjoint within the batch.

The resulting single basis retains every previously certified property:

- the code remains the self-dual `[[64,8,8]]` Liang--Chen BB code;
- every stabilizer row has weight eight;
- physical code permutations induce a regular logical `C_4 x C_2` action;
- transversal Hadamard implements logical Hadamard plus
  `(0 5)(1 4)(2 7)(3 6)`;
- the Z and X representatives each have an exact-cover partition into
  internally disjoint batches of sizes `3+2+3`;
- every logical representative in both bases now has weight eight.

The Z exact cover remains

\[
\{0,1,7\}\;\sqcup\;\{4,6\}\;\sqcup\;\{2,3,5\},
\]

and the Hadamard-permuted X cover remains

\[
\{2,4,5\}\;\sqcup\;\{1,3\}\;\sqcup\;\{0,6,7\}.
\]

Every logical occurs exactly once in the appropriate cover.  Disjointness is
required within each batch, not between different batches.

## Search result

| Query | Result | Solver time |
|---|---:|---:|
| Logical `2` has a weight-eight representative | `sat` | `0.022127 s` |
| Logical `3` has a weight-eight representative | `sat` | `0.014678 s` |
| Logical `5` has a weight-eight representative | `sat` | `0.002538 s` |
| Disjoint pattern `(8,8,8)` for `\{2,3,5\}` | `sat` | `0.022690 s` |

Because the published code distance is eight, none of these nontrivial logical
representatives can have weight below eight.  Thus every saved representative
is individually minimum weight, and the total weight `64` of one representative
for each of eight logicals is also minimum.  This statement does not imply that
all eight representatives are mutually disjoint at once; the operational cover
uses the three displayed batches.

## Preferred files

- `simultaneous_basis_weight64.npz`: complete preferred matrix artifact;
- `weight64_certificate.json`: machine-readable independent certificate;
- `success.json`: low-weight solver witness;
- `summary.json` and `pattern_results.json`: search outcomes;
- `../../search_low_weight_batch.py`: reproducible targeted search;
- `../../certify_weight64.py`: certification and artifact writer.
