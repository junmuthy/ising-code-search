# `[[36,4,6]]` coset-2BGA reference code

This directory records the CSS code
`[[36,4,6]]coset2bga:2f346b16` from the machine-readable data accompanying
Victor V. Albert, [*Beyond transversality: structure of Clifford circuits for
CSS codes*](https://arxiv.org/abs/2608.05688).  The upstream record is in
[`ldpc_codes_depth1.json`](https://github.com/valbert4/two-fold-transversal/blob/main/data/ldpc_codes_depth1.json#L3918-L4017).

The source lists 16 independent checks of each CSS type, exact distance six,
and UUID `2f346b16-ddc0-4c7f-a774-c41ac4127c63`.  The presentation in
`code_data/presentation.json` preserves those checks and the source logical
basis.  It also records the lower-weight logical basis and symmetries derived
directly from the published stabilizers.

## Weight-seven cyclic logical basis

The four rows in each column below are pairwise disjoint.  Every row has weight
seven, and the pairing matrix is exactly
`logical_x @ logical_z.T = I_4 (mod 2)`.

| logical | `X` support | `Z` support |
|---|---|---|
| `0` | `{1,4,9,19,24,29,30}` | `{4,5,17,18,25,27,33}` |
| `1` | `{2,8,11,14,22,23,35}` | `{6,10,15,16,23,28,32}` |
| `2` | `{6,12,16,26,28,31,32}` | `{2,8,12,20,21,22,35}` |
| `3` | `{0,5,13,17,18,25,34}` | `{1,3,7,9,24,30,34}` |

The coordinate automorphism `p` maps the displayed `X` rows as

```text
X_0 -> X_1 -> X_2 -> X_3 -> X_0.
```

It has physical order eight but induced logical order four.  Thus the exact
statement is a logical `C4` action, not a sector-preserving physical
permutation of order four.

The coordinate permutation `q` is a ZX duality:

```text
q(C_X) = C_Z,
q(C_Z) = C_X,
q(X_i) = Z_i,
q(Z_i) = X_i.
```

Consequently `P_q H^tensor36` preserves the codespace and implements
`H^tensor4` in the displayed logical basis.  The Hadamard is permutation
assisted; bare `H^tensor36` is not the claimed physical gate.

## Minimum-support certificate

Exhaustive enumeration of all `15 * 2^16 = 983040` nontrivial pure-`X`
logical representatives finds 64 representatives of weight six.  Their
logical cosets span only a three-dimensional subspace.  Four independent
weight-six pure-`X` logicals are therefore impossible, even without the
disjointness constraint.  ZX duality gives the same statement for pure-`Z`
logicals.

The displayed weight-seven orbit proves that seven is the minimum common
support weight for four independent, pairwise-disjoint logicals cycled by
`p`.  If cyclicity is dropped, a disjoint basis with weights `(6,6,6,7)` also
exists, but unequal weights cannot form one coordinate-permutation orbit.

## Validation

From the repository root, run:

```bash
.venv/bin/python codes/n36_k4_d6_coset2bga/code_data/code.py
```

The validator reconstructs the CSS matrices and exhaustively checks their
ranks, exact `X/Z` distances, the weight-six obstruction, the symplectic
logical basis, disjointness, ZX duality, transversal logical Hadamard, and the
logical `C4` action.
