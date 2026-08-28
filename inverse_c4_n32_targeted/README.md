# Targeted `n=32` folded-ZX continuation

This search starts from the saved `[[32,4,5]]` `(4,4,4,2)` frontier instead
of synthesizing arbitrary rank-14 modules with Z3.

The first stage enumerates all 232 involutions of the seven occupied thickness
fibres, with both the identity and reflection actions on the logical `C4`
coordinate.  The eighth spectator fibre is fixed.  For every saved seed it
tests

\[
H_XH_Z^T=0,\qquad H_Z=P(H_X),
\]

the logical pairing, distinctness of the two check spaces, and Tanner
connectivity.  Because `P` preserves the stored logical-support family and
physical weight, a folded seed retains the seed's exact `d_X=d_Z=5` distance.

This scan is a cheap prerequisite for folded local refinement.  It does not
claim to cover folds with thickness-dependent shifts of the logical `C4`
coordinate.  `scan_shifted_folds.py` supplies that next targeted stage for the
identity thickness permutation, exhaustively covering all shifts compatible
with an involutive fold and permutation logical pairing.

`folded_refinement.py` and `run_folded_refinement.py` then replace complete
translated stabilizer orbits inside the productive `(4,4,4,2)` module.  New
orbit generators are drawn from the exact linear nullspace imposed by the
retained checks, logical grid, and folded CSS bilinear form.  Each accepted
neighbor is checked for rank, folded orthogonality, Tanner connectivity, and
exact logical distance.
