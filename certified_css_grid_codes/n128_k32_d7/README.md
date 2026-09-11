# Connected [[128,32,7]] equivariant control

This earlier connected control has a regular physical and logical
`C4 x C8` action, equal X/Z check spaces, and a canonical basis with all
X and Z logicals of weight eleven. Transversal `H^128` acts as an individual
Hadamard on every logical without any physical or logical permutation.
Checks have rank 48 per Pauli type: 32 rows of weight 18 and 16 of weight
14. Exact X and Z distances are both seven.

This is a general equivariant orthogonal construction, **not** a
nonabelian square-GALA hit. Coordinates are four sheets over `C4 x C8`,
flattened as `32*s+8*i+j`. Start from 32 pair checks between sheets zero
and one and 16 pair checks within sheet two between `(i,j)` and `(i+2,j)`.
The initial logicals are single sites on sheet three.

Apply two binary orthogonal transformations `S=I+V^T V` to both sectors,
where the rows of `V` are the 32 translates of a four-sheet seed. The
saved recipe is:

```text
[a, b, ordering]
[536870913, 1073741824, [3,0,2,1]]
[8388608,   8,          [3,1,2,0]]
```

Before reordering, the four coefficient vectors are `[a,a,b,b]`; each
integer encodes a 32-site support. This is reconstructed directly by the
[portable verifier](../README.md#standalone-verification).

The weight-seven support `[1,21,30,42,64,75,94]` is a nontrivial logical
in both sectors, and complete enumeration excludes smaller logicals.
The historical `candidate.json` retains the initial lower bound six;
use `metadata.json` and the final `certificates/distance_audit.json` for
the exact distance seven.
