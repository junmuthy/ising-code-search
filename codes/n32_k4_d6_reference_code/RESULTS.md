# Targeted folded-ZX `[[32,4,6]]` result

## Result

The targeted search found and independently certified a folded CSS code with

\[
\boxed{[[32,4,6]]}.
\]

It satisfies the single-row Ising requirements:

- four pairwise-disjoint logical `Z` representatives forming one `C4` orbit;
- four pairwise-disjoint folded logical `X` representatives;
- exact distances `d_X=d_Z=6`;
- physical `C4` translation acting as the logical four-cycle;
- distinct check spaces related by one fixed physical involution `P`;
- logical ZX duality implemented by `P H^{\otimes32}`;
- CSS structure, hence transversal inter-block CNOT;
- connected Tanner graph;
- maximum stabilizer-generator weight eight.

The four logical supports in each Pauli basis have weight seven.  They occupy
28 qubits and leave the physical orbit `{7,15,23,31}` as the spectator fibre.

## Why the targeted search was fast

A direct rank-14 Z3 synthesis would be more expensive than the preceding
rank-13 `n=30` search.  Instead, the search reused the saved self-dual
`[[32,4,5]]` frontier in its productive `(4,4,4,2)` cyclic-module component.

The first scan tested all 232 involutions of the seven occupied thickness
fibres, with both orientations of the logical coordinate.  Across 100 saved
seeds it tested 46,300 pairs in 0.49 seconds.  Ten were folded CSS codes, but
all ten preserved the original stabilizer row space.

The second scan fixed the thickness fibres and exhaustively varied every
thickness-dependent logical-`C4` shift compatible with an involutive fold and
permutation logical pairing.  It tested

\[
3,353,500
\]

seed/fold pairs in 36.6 seconds and found:

- 1,100 connected folded CSS seeds;
- 800 with genuinely distinct `H_X` and `H_Z` row spaces.

The local search then replaced complete translated stabilizer orbits while
preserving the folded CSS bilinear form.  Twenty diverse starts and at most 250
iterations per start produced:

| Exact neighbor distance | Count |
| --- | ---: |
| `2` | 51 |
| `3` | 502 |
| `4` | 708 |
| `5` | 3,339 |
| `6` | 1 |

There were also 240 disconnected neighbors and eight failed orbit
replacements.  The hit appeared after 1,099 aggregate iterations; all workers
finished after 4,849 iterations in 43.8 seconds.

## Stabilizers

Both check spaces have rank 14 and are different:

\[
\operatorname{row}(H_X)\ne\operatorname{row}(H_Z).
\]

The physical involution exchanges them:

\[
\operatorname{row}(P H_X)=\operatorname{row}(H_Z),\qquad
\operatorname{row}(P H_Z)=\operatorname{row}(H_X).
\]

A minimum-weight generating basis has row weights

\[
(4,8,8,8,8,8,8,8,8,8,8,8,8,8).
\]

Thus the maximum displayed stabilizer weight is eight, not merely the search
ceiling of 12.

## Logical supports and gates

The logical `Z` representatives are

\[
\begin{aligned}
z_0&=\{0,1,2,3,4,5,6\},\\
z_1&=\{8,9,10,11,12,13,14\},\\
z_2&=\{16,17,18,19,20,21,22\},\\
z_3&=\{24,25,26,27,28,29,30\}.
\end{aligned}
\]

The folded logical `X` representatives are

\[
\begin{aligned}
x_0&=\{0,1,2,4,19,21,22\},\\
x_1&=\{11,13,14,24,25,26,28\},\\
x_2&=\{3,5,6,16,17,18,20\},\\
x_3&=\{8,9,10,12,27,29,30\}.
\end{aligned}
\]

Each list is internally pairwise disjoint.  The ZX pairing matrix is

\[
\begin{pmatrix}
0&0&1&0\\
0&1&0&0\\
1&0&0&0\\
0&0&0&1
\end{pmatrix}.
\]

It is a permutation matrix corresponding to the logical reflection

\[
i\longmapsto2-i\pmod4.
\]

Accordingly, `P H^{\otimes32}` implements logical Hadamard together with this
known reflection.  The physical fold normalizes the translation `T` as

\[
P T P^{-1}=T^{-1},
\]

so the translated nearest-neighbor Ising operations remain consistent.

The fold uses logical-coordinate reflection and thickness-dependent shifts

\[
(0,0,0,2,0,2,2,0).
\]

## Independent validation

The survivor was reconstructed from the saved `H_X` and `H_Z` matrices as a
`qldpc.codes.CSSCode`.  qLDPC independently returned:

\[
n=32,\qquad k=4,\qquad d_X=6,\qquad d_Z=6.
\]

The validator also independently checked:

- rank 14 for both check spaces;
- exact CSS orthogonality;
- distinct check row spaces;
- bidirectional exchange of the check spaces by `P`;
- `C4` invariance of both check spaces;
- fold involution and translation normalization;
- four independent logicals modulo stabilizers in both bases;
- disjoint weight-seven supports in both bases;
- permutation ZX pairing and fold mapping from every stored `Z` support to its
  stored `X` support;
- Tanner connectivity.

Every required validation flag passed.

## Primary artifacts

- `results/folded-local-20x250-260828-v1/start-results/start-010.json`: full
  winning task, including fold metadata and check matrices.
- `results/folded-local-20x250-260828-v1/survivors.jsonl`: compact survivor.
- `results/folded-local-20x250-260828-v1/validation-v1.json`: independent qLDPC
  and structural certification.
- `results/folded-local-20x250-260828-v1/summary.json`: local-search totals.
- `results/shifted-identity-thickness-all100-260828-v1/summary.json`: exhaustive
  shifted-fold scan totals.

The code is a general `C4`-invariant folded CSS code.  This search did not
require it to have a BB or GALA group-polynomial presentation; deriving a
compact algebraic presentation is a separate follow-up task.
