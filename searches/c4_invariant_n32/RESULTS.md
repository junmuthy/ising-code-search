# Direct `C4`-invariant `[[32,4,*]]` search

## Geometry and acceptance conditions

Physical qubits form a `4 x 8` array.  The four logical representatives are
the first seven positions in each column; positions `7`, `15`, `23`, and `31`
form one additional physical `C4` orbit.  Thus the logical supports remain
pairwise disjoint and odd-weight:

\[
L_i=\{8i,8i+1,\ldots,8i+6\},\qquad LL^T=I_4.
\]

The search constructs a 14-dimensional, self-orthogonal, `C4`-invariant
stabilizer space `S` inside `L^perp` and uses `H_X=H_Z=S`.  Every structurally
accepted candidate is therefore a `[[32,4,*]]` code with:

- exact transversal Hadamard;
- four disjoint logical supports and identity ZX pairing;
- physical `C4` translation acting as the desired logical four-cycle;
- transversal inter-block CNOT;
- a connected Tanner graph;
- stabilizer generators of weight at most 12.

Distances are exact.  The program enumerates all `2^14=16384` stabilizers in
each of the 15 nonzero logical cosets.

## Strict extensions cannot help

Embedding a complete `[[28,4,5]]` stabilizer and merely adding a rank-two
orbit has a unique result modulo the embedded stabilizer: two isolated
weight-two checks on the four new qubits.  This follows because the old
rank-12 stabilizer is already maximal isotropic inside the old logical
complement.  The extension is Tanner-disconnected and retains distance five.

Consequently, a useful extension must modify old stabilizer orbits while
coupling in the new physical orbit.

## Coupled extension search

For each saved `[[28,4,5]]` seed, the search retained two of its three
rank-four translated stabilizer orbits and jointly regenerated:

- the third rank-four orbit;
- the new rank-two orbit.

Across 75,000 checkpointed trials from 75 seeds:

- 25,553 candidates were constructed;
- 2,357 connected distinct candidates were exactly certified;
- 498 had `d=3`, 675 had `d=4`, and 1,184 had `d=5`;
- none had `d>=6`.

## Fresh module search

The unrestricted constructor scanned all 47 cyclic `C4` module partitions of
rank 14.  A 100-trial-per-type pilot found ten constructible types.  Only
module type `(4,4,4,2)` produced distance-five candidates, so the confirmation
run focused on that component.

The 20,000-trial `(4,4,4,2)` run certified 3,001 connected codes:

| Exact distance | Count |
| --- | ---: |
| `2` | 481 |
| `3` | 336 |
| `4` | 1,541 |
| `5` | 643 |
| `>=6` | 0 |

## Directed refinement

The final search selected the best 100 fresh `d=5` representatives and
replaced complete rank-four or rank-two translated generator orbits.  Across
25,000 neighborhoods it certified:

- 480 `d=2` neighbors;
- 3,132 `d=3` neighbors;
- 3,197 `d=4` neighbors;
- 14,713 `d=5` neighbors;
- no `d>=6` neighbor.

## Best code found

The leading result is `[[32,4,5]]`.  Its minimum-weight stabilizer basis has:

- one weight-four row;
- four weight-six rows;
- nine weight-eight rows.

Only four weight-five logical representatives remain across all 15 nonzero
logical cosets, so it lies close to the requested distance-six boundary.
Independent construction as `qldpc.codes.CSSCode(H,H)` returned `n=32`,
`k=4`, maximum check weight 8, and `get_distance_exact()=5`.

## Conclusion and scope

Adding a complete spectator orbit produces many more `d=5` codes and permits
a particularly low-weight `[[32,4,5]]` representative, but this search did not
find `[[32,4,6]]`.  The result combines 75,000 coupled trials, 20,000 focused
fresh trials, and 25,000 directed refinements.

This is strong randomized negative evidence for the weight-at-most-12,
orbit-generated ansatz, not an exhaustive impossibility proof for every
`C4`-invariant rank-14 stabilizer space.  The best remaining route at fixed
`n=32` is a solver-guided search that explicitly forbids the four residual
weight-five logical operators and learns new exclusions iteratively.

## Primary artifacts

- `results/c4-invariant-n32/coupled-all75x1000-v1/summary.json`
- `results/c4-invariant-n32/fresh-modules-pilot-100-v1/summary.json`
- `results/c4-invariant-n32/fresh-4442-20k-v1/summary.json`
- `results/c4-invariant-n32/local-4442-100x250-v1/summary.json`
- `results/c4-invariant-n32/local-4442-100x250-v1/best-candidates.jsonl`
