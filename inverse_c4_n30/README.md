# Mixed-orbit `[[30,4,>=6]]` synthesis

This directory implements the intermediate search between the exact `n=24`
obstruction / `n=28` distance-five frontier and the cleaner `n=32` geometry.

The physical `C4` action has orbit decomposition

\[
30=7\cdot4+2.
\]

The first seven orbits are regular `C4` fibres.  The final two qubits form a
single `C2` orbit under the same order-four physical translation.  Four
pairwise-disjoint weight-six logical representatives occupy six regular
fibres; the seventh regular fibre and the short orbit are slack.

The short orbit cannot occur in the disjoint logical representatives, because
translation by two would make two different logical supports overlap.  It may,
however, participate in stabilizer checks and couple to the regular fibres.

Every candidate has rank 13 in each CSS check space and therefore `k=4`:

\[
30-13-13=4.
\]

The hard constraints are the same as in `inverse_c4_minimum`:

- exact CSS orthogonality;
- four disjoint logical `Z` supports forming one `C4` orbit;
- logical `X` supports obtained by a physical involution `P`;
- `P T P^{-1}=T^{\pm1}`;
- permutation logical ZX pairing;
- `row(H_Z)=row(H_XP)`, so the logical gate is `P H^{\otimes30}`;
- connected Tanner graph, qubit coverage, and generating-check weight at most
  12;
- exact distance certification by enumeration of all 8,192 stabilizers in all
  15 nonzero logical cosets in both Pauli sectors.

The first search targets module types `(4,4,4,1)` and `(4,4,3,2)`, the two
natural rank-13 continuations of the `n=28` distance-five frontier.  The fold
catalog contains both actions on the short orbit for every stored regular
fold.

All runs are checkpointed and refuse to overwrite an existing output
directory.
