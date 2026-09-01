# `[[64,8,8]]` simultaneous Ising-basis search

This search requires one and the same ordered logical basis to have:

1. a regular permutation `C4 x C2` action;
2. a permutation ZX Gram matrix under transversal H;
3. an exact-cover partition of all eight logicals into internally disjoint Z
   batches, each containing at least two logicals.

The inexpensive phase exactly enumerates all affine physical code
automorphisms of the twisted torus, all commuting logical order-four/order-two
pairs with a regular action algebra, and all 255 nonzero logical seed classes.
Only candidates with a permutation Gram matrix reach the support stage.

The support stage first checks every arbitrary exact-cover partition directly.
It then uses Z3's native XOR constraints to choose both the partition and
arbitrary Z-stabilizer dressings jointly.  It tries one through four batches,
prefers fewer batches within each candidate, requires every batch to contain at
least two logicals, and constrains every logical to occur exactly once.
Stabilizer dressing cannot change the ZX Gram matrix.  `sat` and `unsat`
answers are exact; a solver timeout is explicitly recorded as unresolved.

This is exhaustive within the affine BB automorphism family.  It is not, by
itself, an enumeration of every stabilizer-row-space-preserving physical
permutation outside that family.

## Result

`results/run_003_low_weight` contains the preferred certified result with three
exact-cover batches of sizes `3+2+3`.  All eight Z representatives and all
eight canonical X representatives have weight eight.  The same basis has a
regular `C_4 x C_2` logical action and transversal Hadamard plus the logical
permutation `(0 5)(1 4)(2 7)(3 6)`.

See `RESULTS.md` for the complete interpretation and
`results/run_003_low_weight/simultaneous_basis_weight64.npz` for the preferred
matrix artifact.  The earlier `run_001` and `run_002` results are retained
unchanged for provenance.
