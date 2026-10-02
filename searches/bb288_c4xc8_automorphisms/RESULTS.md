# `[[288,32,8]]` BB candidate: Tanner-automorphism result

## Outcome

The published code reconstructs correctly and has an exact BB ZX fold, but its
canonical translated-check Tanner presentation does **not** support a regular
logical \(C_4\times C_8\) grid through Tanner-graph automorphisms.

The main structural finding is that this is not a connected packed code:

\[
[[288,32,8]]
\cong
[[72,8,8]]^{\oplus 4}
\]

at the Tanner-component level.  The four components are isomorphic and are
related by the native \(y\)-translations.

## Reconstructed code

The source is class `j` in Table 2 of Cruz-Benito *et al.*,
[arXiv:2606.02418](https://arxiv.org/abs/2606.02418):

\[
A=1+xy^3+x^5y^3,
\qquad
B=1+x^3y+x^3y^5
\]

over \(\mathbb F_2[C_{12}\times C_{12}]\), with

\[
H_X=[A\mid B],
\qquad
H_Z=[B^T\mid A^T].
\]

The independent reconstruction gives

\[
\operatorname{rank}H_X=operatorname{rank}H_Z=128,
\qquad
k=288-128-128=32.
\]

All X and Z checks have weight six, and \(H_XH_Z^T=0\).  The published exact
MILP distance is \(d=8\); the scripts record that published certification but
do not rerun the distance MILP.

The standard BB physical permutation consisting of group inversion plus an
exchange of the two physical halves satisfies

\[
H_X P \sim H_Z,
\qquad
H_Z P \sim H_X,
\]

where \(\sim\) denotes equality of row spaces.  The permutation is an
involution, so the code has the required GALA-style ZX duality.

## Tanner decomposition

The colored Tanner graph has 576 vertices and four connected components, each
containing

- 72 data qubits;
- 36 X-check vertices;
- 36 Z-check vertices.

Within every component,

\[
\operatorname{rank}H_X^{(c)}
=
\operatorname{rank}H_Z^{(c)}
=32,
\qquad
k_c=72-32-32=8.
\]

Because all four components are isomorphic and the full code has published
distance eight, every component is a `[[72,8,8]]` code.

Nauty gives the connected-component Tanner automorphism group

\[
|G|=288.
\]

The full Tanner automorphism group is the wreath product

\[
G\wr S_4,
\qquad
|G\wr S_4|
=288^4\,4!
=165{,}112{,}971{,}264.
\]

This agrees exactly with nauty's direct calculation on the full graph.

## Exhaustive logical-action screens

### One component

All 288 connected-component Tanner automorphisms were enumerated.  Their
physical-order histogram is

| Physical order | Number |
|---:|---:|
| 1 | 1 |
| 2 | 55 |
| 3 | 8 |
| 4 | 104 |
| 6 | 80 |
| 12 | 40 |

On the eight-dimensional logical quotient there are 72 order-four actions and
84 order-two actions.  All 144 physically commuting order-four/order-two pairs
were tested.  None generated eight distinct logical `C4 x C2` actions, so the
simple four-component twisted-cycle construction is impossible.

### Full wreath product: cyclic order-four top action

For a normalized order-eight element whose component permutation is a
four-cycle, all 84 possible logical order-two twists and their complete
centralizers were tested:

| Quantity | Count |
|---|---:|
| Normalized twists | 84 |
| Centralizer candidates | 96,768 |
| Valid centralizer elements | 10,368 |
| Logical order-four candidates | 2,016 |
| Faithful `C4 x C8` pairs | 0 |

### Full wreath product: transitive Klein-four top action

The other transitive abelian component action is the Klein four group.  All 72
normalized logical order-four twists were tested:

| Quantity | Count |
|---|---:|
| Normalized twists | 72 |
| Centralizer pairs | 4,608 |
| Logical order-four candidates | 1,152 |
| Faithful `C4 x C8` pairs | 0 |

These two cases exhaust the relevant Tanner-automorphism search.  A regular
32-dimensional module for the 2-group \(C_4\times C_8\) is the regular module
over \(\mathbb F_2[C_4\times C_8]\), hence is indecomposable.  Its action on
the four Tanner components must therefore be transitive.  The only transitive
abelian subgroups of \(S_4\) are cyclic `C4` and the Klein four group, and both
were covered above.

## Interpretation and scope

This is a negative result for **Tanner-graph automorphisms of the canonical
translated-check presentation**.  It does not rule out a more general physical
permutation that preserves the stabilizer row spaces while mapping individual
checks to products of checks, nor does it rule out an alternative check
presentation of the same stabilizer code.

Architecturally, the decomposition already makes the candidate less attractive
than four copies of the previously analyzed `[[64,8,8]]` code:

\[
[[256,32,8]]
=
[[64,8,8]]^{\oplus4}
\]

uses 32 fewer data qubits and has the desired component-weaving construction.
The `[[288,32,8]]` candidate's compensating advantage is its weight-six rather
than weight-eight stabilizers.

## Reproduction

The saved runs are:

- `results/run_002`: reconstruction, ZX fold, connectivity, and full nauty
  group size;
- `results/components_001`: exact connected-component group and local screen;
- `results/wreath_c4_top_all_twists_001`: exhaustive cyclic-top screen;
- `results/wreath_v4_top_001`: exhaustive Klein-four-top screen.

Every command refuses to overwrite prior output and prints progress
checkpoints.
