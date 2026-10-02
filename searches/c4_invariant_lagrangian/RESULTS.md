# Direct `C4`-invariant `[[28,4,*]]` search

## Objective

This search drops the unnecessary physical `C7` translation imposed by the
scalar-circulant `C28 ~= C4 x C7` search.  It retains only the required Ising
row translation.  Physical coordinates are a `4 x 7` array and the four
canonical logical representatives are its disjoint weight-seven columns,

\[
L_i=\{(i,t):t\in\mathbb Z_7\},\qquad i\in\mathbb Z_4.
\]

Because seven is odd, `L L^T = I_4`.  We search for a 12-dimensional,
self-orthogonal, `C4`-invariant stabilizer space `S` inside `L^perp` and set

\[
H_X=H_Z=S.
\]

Every accepted structural candidate is therefore a `[[28,4,*]]` CSS code with:

- exact ZX self-duality and physical transversal Hadamard;
- four disjoint logical supports;
- physical `C4` translation acting as the desired logical four-cycle;
- transversal inter-code-block CNOT;
- a reported stabilizer basis of weight at most 12.

For every connected candidate, distance is exact: the program enumerates all
`2^12 = 4096` stabilizers in each of the 15 nonzero logical cosets.

## Free-module search

The first family takes `S` to be a free rank-three `F2[C4]` module, generated
by three independent rank-four translated stabilizer orbits.

The checkpointed 100,000-trial run produced 97,238 connected distinct codes:

| Certified distance | Count |
| --- | ---: |
| `2` | 18,789 |
| `3` | 17,190 |
| `4` | 53,340 |
| `5` | 7,919 |
| `>=6` | 0 |

The best result was therefore `[[28,4,5]]`, not the requested
`[[28,4,>=6]]`.

The leading saved representative has a minimum-weight stabilizer basis with
eleven weight-eight rows and one weight-ten row.  An independent construction
as `qldpc.codes.CSSCode(H,H)` returned `n=28`, `k=4`, check weight 10, and
`get_distance_exact()=5`, agreeing with the coset enumeration.

## Local refinement of the `d=5` frontier

The search retained 75 representative `d=5` starts.  A local search repeatedly
removed one complete translated generator orbit and regenerated it subject to
all structural constraints.  Across 37,500 replacements it observed:

- 30 improvements in the low-weight-logical spectrum;
- 10,028 connected `d=5` neighbors;
- no `d>=6` neighbor.

This makes a merely isolated `d=6` basin less likely, but is not an exhaustive
enumeration of the free-module component.

## Non-free module types

Over `F2`, a `C4` representation is unipotent.  A 12-dimensional invariant
space can decompose into translated cyclic orbits of ranks 1, 2, 3, and 4.
The second constructor enumerated all 33 integer partitions of 12 with parts
at most four, excluding the already sampled `(4,4,4)` type, and deliberately
sampled vectors from the appropriate kernels of `(T+I)^r`.

In the 1,000-attempt-per-type confirmation run:

- 33,000 constructions were attempted;
- 5,040 stabilizer spaces were constructed;
- 3,439 connected distinct candidates were certified;
- the exact distribution was `d=2`: 2,168, `d=3`: 210, `d=4`: 1,061;
- no non-free candidate reached `d=5` or `d>=6`.

Nine module types were constructible under this randomized weight-12 ansatz;
the rest consistently failed during constrained orbit extension.  The full
list is recorded in the machine-readable summary.

## Conclusion and scope

Breaking the physical `C7` symmetry helped: the broader free-module search
found many `d=5` codes, whereas the exhaustive scalar-circulant search topped
out at `d=4`.  It did not find the requested distance six.

This is strong negative evidence for low-weight, orbit-generated,
`C4`-invariant self-dual `[[28,4,>=6]]` codes, not an impossibility proof for
every `C4`-invariant stabilizer space.  The randomized ansatz still requires a
cyclic-module decomposition with generators of weight at most 12.  A complete
answer would require an exact SAT/CP-SAT classification or a provable distance
bound for invariant Lagrangians.

## Reproduction

```bash
.venv/bin/python -m searches.c4_invariant_lagrangian.run_search \
  --run-name random-free-module-100k-v1 \
  --batches 100 --trials-per-batch 1000 --workers 8

.venv/bin/python -m searches.c4_invariant_lagrangian.run_local_search \
  --input results/c4-invariant-lagrangian/random-free-module-100k-v1/near-misses.jsonl \
  --run-name orbit-replacement-all75x500-v1 \
  --starts 75 --iterations 500 --workers 8

.venv/bin/python -m searches.c4_invariant_lagrangian.run_module_type_search \
  --run-name nonfree-module-types-1000-v1 \
  --trials-per-type 1000 --workers 8 \
  --maximum-check-weight 12 --attempts-per-orbit 200
```

Primary artifacts:

- `results/c4-invariant-lagrangian/random-free-module-100k-v1/summary.json`
- `results/c4-invariant-lagrangian/orbit-replacement-all75x500-v1/summary.json`
- `results/c4-invariant-lagrangian/nonfree-module-types-1000-v1/summary.json`
