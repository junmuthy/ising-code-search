# Minimum-size `C4` Ising-code synthesis results

## Outcome

The search has an exact obstruction at `n=24` and a broad distance-five
frontier at `n=28`.

- `n=24`: no code can have four disjoint logical representatives of weight at
  least six and exchange them with four logical representatives through
  transversal Hadamard followed by any physical permutation.
- `n=28`: no `[[28,4,>=6]]` code was found under the tested `C4`-invariant,
  permutation-ZX-folded CSS ansatz with a stabilizer generating basis of weight
  at most 12.
- `n=28`: many `[[28,4,5]]` codes were found. Two representatives were saved
  and independently checked with qLDPC.

The `n=28` result is a scoped computational result, not a universal no-go
theorem for every `C4`-invariant CSS code or every physical permutation.

## Hard operational constraints

Every synthesized model satisfies the following before its distance is
measured:

1. Four pairwise-disjoint logical `Z` supports form one physical `C4` orbit.
2. A physical permutation `P` is an involution and normalizes the translation
   `T`:

   \[
   PTP^{-1}=T^{\pm1}.
   \]

3. The logical `X` representatives are the folded logicals `Pz_i`, and their
   pairing with the logical `Z` representatives is a permutation matrix.
4. The stabilizer spaces obey

   \[
   \operatorname{row}(H_Z)=\operatorname{row}(H_XP).
   \]

   Consequently, the physical logical gate is

   \[
   P H^{\otimes n}.
   \]

   The search allows `H_X != H_Z`; it does not allow unrelated check spaces.
5. The code is CSS, has `k=4`, has a connected Tanner graph, and has no isolated
   qubits.
6. The displayed `C4`-orbit stabilizer basis has check weight at most 12.

## Exact `n=24` obstruction

Four disjoint logical supports at distance at least six occupy at least 24
qubits. At `n=24`, they therefore partition the whole physical support and each
has weight exactly six. For any physical permutation `P`, define

\[
M_{ij}=z_i\mathbin{\cdot}Pz_j.
\]

Since the four `Pz_j` also partition all qubits,

\[
\sum_j M_{ij}
=z_i\mathbin{\cdot}P\mathbf 1
=\operatorname{wt}(z_i)
=0\pmod 2.
\]

Thus

\[
M(1,1,1,1)^T=0,
\]

so `M` is singular. It cannot be the nonsingular pairing required for four
independent logical qubits. This proof permits every physical permutation; it
is not restricted to the search fold catalog.

Every run saves this proof as `n24-no-go-certificate.json` before solving an
`n=28` task.

## `n=28` construction

The physical qubits are seven `C4` fibres. The canonical logical `Z` supports
use six fibres:

\[
z_t=\{4f+t:f=0,1,\ldots,5\},\qquad t\in\mathbb Z_4.
\]

They are disjoint and have weight six. A fold exchanges an occupied fibre with
the seventh, slack fibre. Optional internal fibre swaps and a reflection of the
`C4` coordinate give the six stored fold geometries. Each fold is checked
directly for the normalization and logical-pairing conditions above.

Z3 chooses a complete rank-12 `C4`-invariant `X`-stabilizer space. The
`Z`-stabilizer space is its physical folded image. Exact CSS orthogonality,
rank, logical-kernel, coverage, and weight conditions are symbolic constraints.
Distance is then computed exactly by enumerating 4,096 stabilizers in each of
the 15 nonzero logical cosets in both Pauli sectors.

When a model has distance below six, every retained low-weight logical operator
is added as a necessary detection constraint. An `UNSAT` result after these
counterexample cuts is therefore exact for that specific fold/module task. A
timeout or model limit remains bounded and is never reported as a no-go proof.

## Aggregate search record

Across the implemented inverse-fold runs:

| Exact code distance | Connected models |
| --- | ---: |
| `2` | 31 |
| `3` | 29 |
| `4` | 75 |
| `5` | 36 |
| `>=6` | 0 |
| Total | 171 |

The runs learned 9,032 low-distance operator cuts. The two focused
continuations alone tested 80 new connected codes after replaying their earlier
cuts; 34 had exact distance five.

The eight remaining module types that had been constructible in the earlier
randomized self-dual search were tested under both strongest fold geometries.
Fourteen of those 16 tasks became exact `UNSAT`; the two `(4,4,2,2)` tasks hit
the five-model bound with best distance four. Thus only the free `(4,4,4)` and
the `(4,4,3,1)` decompositions reached distance five in these folded searches.

## Independently validated `[[28,4,5]]` representatives

Two saved frontier codes were rebuilt as `qldpc.codes.CSSCode` objects. qLDPC
returned `n=28`, `k=4`, `d_X=5`, and `d_Z=5` for both. Independent matrix
checks also established:

- `H_X != H_Z` and their row spaces are distinct;
- `P(H_X)=H_Z` row for row and `P(H_Z)` has the `H_X` row space;
- both check spaces are invariant under the physical `C4` translation;
- all four logical `Z` supports and all four logical `X` supports are mutually
  disjoint within their respective bases and have weight six;
- both logical bases are independent modulo stabilizers;
- the fold maps every stored logical `Z` representative to its stored logical
  `X` representative.

The orientation-preserving representative uses module type `(4,4,3,1)` and has
logical pairing `I_4`. Its row weights are

\[
(11,11,11,11,12,12,12,12,12,12,12,8).
\]

The reflected representative uses module type `(4,4,4)` and pairs logical
indices as `0 -> 0`, `1 -> 3`, `2 -> 2`, `3 -> 1`. Its row weights are

\[
(11,11,11,11,12,12,12,12,10,10,10,10).
\]

The first fold commutes with `T`; the second conjugates `T` to `T^{-1}`. Both
therefore retain an easy logical Hadamard and a consistent cyclic Ising row.

## Scope of the negative evidence

The current exact task results cover a stored catalog of six zero-shift,
involutive normalizer folds and selected cyclic `F_2[C4]` module types. The two
strongest folds have now been tested on all ten module types that produced
structural candidates in the previous randomized search. Some tasks terminated
exactly; the best free and `(4,4,3,1)` tasks remain bounded at distance five.

The computation does not yet classify:

- every normalizer permutation, including arbitrary allowed shifts within
  fibres;
- every rank-12 invariant module type under every fold;
- non-involutive physical permutations that could still implement a valid
  logical ZX duality;
- stabilizer generating bases above weight 12.

Therefore the justified conclusion is that `n=28`, weight at most 12 has a
strong distance-five frontier in the most natural folded geometries—not that a
`[[28,4,6]]` code is impossible in full generality.

## Primary artifacts

- `results/remaining-folds-free-4431-260828-v1/validation-v1.json`: independent
  qLDPC and structural validation of the two saved distance-five codes.
- `results/resume-f2-4431-after424-260828-v1/summary.json`: 40-model continuation
  of the orientation-preserving `(4,4,3,1)` frontier.
- `results/resume-f3-444-after562-260828-v1/summary.json`: 40-model continuation
  of the reflected free-module frontier.
- `results/viable-nonfree-f2-f3-260828-v1/summary.json`: the remaining 16
  viable-type/fold tasks.

All run directories are immutable by convention: the runner refuses to start
if its requested output directory already exists. Progress is written
atomically and models, counterexample cuts, and task summaries are appended or
saved as the run proceeds.
