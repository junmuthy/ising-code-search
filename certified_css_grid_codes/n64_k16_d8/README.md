# Connected [[64,16,8]] with weight-ten logicals

The latest saved basis has all 16 X and all 16 Z logicals of weight ten,
with canonical pairing, regular `C4 x C4` translations, and a common-basis
Hadamard fold. Each physical qubit occurs in exactly five of the 32 logical
supports: total support is 320. The maximum pairwise support overlap is
four. Checks are unchanged: 24 independent weight-12 rows per Pauli type,
with exact X and Z distances both eight.

The lower-total-check-support distance-six comparison is also backed up as
[`[[64,16,6]]`](../n64_k16_d6/README.md), including both its independent and
translation-closed check presentations.

## Construction and actions

Use two sheets over `C8 x C4`, flattening physical `(s,i,j)` to
`32*s+4*i+j`. The little-endian group-algebra coefficients are
`a=2164531200`, `b=23815`. With binary left-regular lifts,
`H_X=[L_a | L_b]`; the Z space is its image under the involution

```text
P(s,i,j) = (1-s, i+4*j mod 8, 2*i+3*j mod 4).
```

The saved check arrays are independent row bases of these spaces.
Physical translations have orders eight and four, but their **logical**
orders are four and four. In logical coordinates modulo four, `P H^64`
acts as individual Hadamards with permutation
`(i,j) -> (i, 2*i+3-j)`.

Relative to the original basis preserved under `provenance/`, the new
basis multiplies three neighboring logicals, then dresses by stabilizers:

```text
X'_(i,j) = X_(i,j) X_(i,j+1) X_(i,j+2)  modulo X stabilizers
Z'_(i,j) = Z_(i,j) Z_(i,j+1) Z_(i,j+2)  modulo Z stabilizers.
```

Use `candidate.json` and `logical_bases.npz` here for the new basis, not the
archived original logicals. Explicit distance-eight witnesses are
`X: [21,28,32,35,38,41,45,58]` and
`Z: [10,16,17,20,27,29,53,62]`.

## Logical-support search scope

The archived independent audit reports minimum maximum logical weight ten
for pure canonical CSS grids with the saved translations and common-basis
Hadamard. Its catalog contains 2,048 translation orbits of unit basis
changes, 256 compatible with the saved fold, with exact coset scans for
those 256 cases. The all-weight-eight grid obstruction and the original
weight-eight logical words are retained separately: distance-eight
logical operators exist even though they cannot supply the required
all-weight-eight canonical grid in that scope.

`alternatives/lower_overlap/` preserves another all-weight-ten basis with
maximum pairwise overlap three and peak physical load six. Overlap
optimality is **not** proved. The default balanced basis attains the
smallest possible peak load, five, at total support 320.

The [package verifier](../README.md#standalone-verification) independently
rechecks this code, the saved basis and exact distance; the full original
basis-search audit is in `certificates/logical_search_audit.json`.
