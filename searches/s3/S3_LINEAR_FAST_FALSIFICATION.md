# Faithful 2D `S_3` fast-falsification campaign

This campaign revisits the saved `S_3 x C_8 x C_4` polynomials using the
faithful two-dimensional binary representation

\[
S_3 \cong \operatorname{SL}(2,2)=\operatorname{GL}(2,2).
\]

It is additive to the earlier searches: every output was written under
`results/s3-linear-fast-falsification/`, and no earlier candidate or result
file was replaced.

## Representation and ZX fold

The new top lift has dimension two rather than the natural permutation
representation's dimension three. A physical block therefore has

\[
2\lvert C_8\times C_4\rvert=64
\]

coordinates. The lift is faithful but not orthogonal, so the group-ring
inverse is not itself the binary matrix transpose. The checks are consequently
built explicitly as

\[
H_X=[F\mid G],\qquad H_Z=[G_{\rm bin}^{T}\mid F_{\rm bin}^{T}].
\]

The fixed alternating form

\[
J=\begin{bmatrix}0&1\\1&0\end{bmatrix}
\]

satisfies

\[
\rho(g^{-1})=J\rho(g)^T J.
\]

Combining the internal `J` swap with a protograph reflection produces the
physical ZX fold. For a relation `G_(i+s)=F_i^T`, the simple fold requires a
self-inverse shift: `s=0`, or `s=h/2` when the half-protograph size `h` is
even. The failed arbitrary-shift pilot was retained because it establishes
this restriction empirically as well as algebraically.

## Saved-family relift

The first cheap filters were CSS orthogonality, the expected ZX fold, and the
absence of logical operators of weight at most four. Exact sparse searches
then tested weights five and six. A graph-supported weight-`d` seed was
translated through `C_8 x C_4`, after which all support-disjoint grid
combinations were tested for both logical independence and joint ZX-pairing
rank.

| Saved family | Relifted | Survived weight `<=4` | Joint-grid conclusion |
|---|---:|---:|---|
| `L=4`/`L=8` folded and `L=4` self-dual | 203 | 74 | Five distance-6 grid candidates; exhaustive joint screen found no valid grid set |
| `L=12` exact self-dual | 1,225 | 2 | Both survivors have a weight-5 logical; negative |
| `L=12` edge/vertex folded | 500 | 215 | Eight `[[768,384,6]]` codes have a valid pair of disjoint grids; none has three |

For the five small grid candidates, every graph-supported weight-6 orbit was
exhausted. The `L=4` grids have self-ZX rank zero. The `L=8` candidates either
have only one grid or have two candidate grids whose internal physical fibres
overlap, so off-diagonal pairing cannot rescue them.

Among the weight-at-most-16 `L=12` folded cases, 153 candidates exposed an
initial grid. Enumeration found 436 grid orbits. It was exhaustive for 126
codes; 27 negative searches reached the node limit. Eight codes have a
full-rank two-grid pairing, accounting for 18 valid pairs. Eleven of those
pairs have a permutation pairing matrix. Six positive codes have maximum
stabilizer weight 14 and two have weight 15.

A representative with permutation ZX pairing is
`s3-l12-j3-w12-vertex-fold-0465d4cc4dff1e87`, with parameters
`[[768,384,6]]`, maximum stabilizer weight 14, and two disjoint 32-qubit
logical grids. Its grid enumeration is exhaustive. This is a useful rescue of
the two-grid direction, but its useful packing is only 64 logicals in 768
physical qubits, and it does not improve the many-copy target.

## Resized pilots

Saved six-entry polynomials were resized without changing their monomials:

- `L=10,J=2`: delete one entry; target three grids.
- `L=14,J=3`: insert one zero entry; target four grids.
- `L=16,J=3`: insert two zero entries; target five grids.
- `L=16,J=4`: insert two zero entries; target five grids.

Only self-inverse fold shifts were retained. One hundred algebraic survivors
per pilot were sampled deterministically. The accepted binary check-weight
ceiling was 16.

| Pilot | Algebraic survivors in transformed pool | Structural survivors out of 100 | Distance/grid result |
|---|---:|---:|---|
| `L=10,J=2` | 931 | 0 | All have a logical of weight at most four |
| `L=14,J=3` | 1,634 | 8 | Three distance-5 failures; four distance-6 grid codes; one distance-6 code without a grid |
| `L=16,J=3` | 12,226 | 0 | 87 low-weight failures and 13 check-weight failures |
| `L=16,J=4` | 3,451 | 33 | Eleven distance-6 grid codes, two `d>=7` weight-7 grid codes, and several node-limited cases |

The 17 resized codes with known grids were subjected to joint enumeration.
Fifteen enumerations were exhaustive and two weight-7 enumerations were
node-limited. None reaches its target of four or five jointly ZX-complete,
support-disjoint grids. In fact, none has even one self-ZX-complete grid or a
full-rank lower-cardinality grid combination. Some codes contain two
independent disjoint grids, but their combined ZX-pairing ranks are zero or
32 rather than 64.

The representative resized parameters are `[[896,512,6]]` for the `L=14`
codes and `[[1024,512,6]]` or `[[1024,512,>=7]]` for the `L=16,J=4` codes,
with maximum stabilizer weights between 13 and 16.

## Conclusions

1. The faithful 2D representation genuinely reduces block length and can
   repair a rank-zero grid through off-diagonal ZX pairing, as demonstrated
   by eight two-grid `[[768,384,6]]` codes.
2. The same change does not automatically increase useful packing. The
   tested sparse resize ansatz fails because zero protograph entries create
   low-weight modes and because its surviving grids do not form sufficiently
   rich ZX-paired modules.
3. Arbitrary fold shifts are not valid. Future searches should impose the
   self-inverse-shift condition before lifting.
4. A serious four- or five-grid search should sample native `L=14`/`L=16`
   polynomial patterns rather than insert zeros into an `L=12` pattern. It
   should score the full off-diagonal grid-pairing matrix during candidate
   generation, not only after distance screening.
5. These resize results are bounded randomized pilots, not exhaustive no-go
   theorems for the complete `L=14` or `L=16` families. The individual
   graph-orbit conclusions are exhaustive wherever explicitly stated above.

## Reproduction

The main new result files are:

- `legacy-l4-l8-grid-sets.jsonl`
- `legacy-l12-folded-grid-sets-w16.jsonl`
- `resized-l10-l14-l16-structural.jsonl` (the deliberately retained
  arbitrary-shift falsification)
- `resized-l10-l14-l16-corrected-structural.jsonl`
- `resized-l14-l16-distance.jsonl`
- `resized-l14-l16-grid-sets.jsonl`

The corresponding `*.summary.json` files sit beside the JSONL outputs. The
new runners refuse to overwrite either an output or its summary.
