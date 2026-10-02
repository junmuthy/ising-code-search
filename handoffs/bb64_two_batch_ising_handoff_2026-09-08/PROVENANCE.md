# Provenance and claim boundary

## Published code

The self-dual bivariate-bicycle construction and `[[64,8,8]]` parameters are
due to Zijian Liang and Yu-An Chen, “Self-dual bivariate bicycle codes with
transversal Clifford gates,” arXiv:2510.05211v2 (2026), especially Table 1,
Figure 1, and Appendix A.  See `CITATION.bib`.

This package reconstructs the code from

\[
f=1+x+y+y^{-1},\qquad g=f^\dagger
\]

on the 32-element twisted torus with `x^4 y^4=1` and `y^8=1`.
`reconstruct_code.py` compares the resulting matrices bit-for-bit with the
archive, while `verify_distance.py` independently obtains minimum nontrivial
logical weight eight with SciPy/HiGHS.

## Derived two-batch basis

The preferred all-weight-eight logical basis was derived by the local
GALA/Ising code-search project.  The reference one-/two-batch refinement was
run on 2026-09-01 with Z3, a 60-second limit per case, and random seed `0`.
It tested eight deduplicated affine logical-class candidates at representative
weight eight:

- all eight one-batch cases were `unsat`;
- two two-batch cases were `sat`;
- six two-batch cases were `unsat`;
- no case timed out.

Candidate `0` was selected because it preserves the preferred ordered logical
classes while changing only their stabilizer dressings.  Its Z cover is
`{0,3,4,7}` together with `{1,2,5,6}`.  The complete inputs, solver, outputs, and
summary are in `provenance/`.

The one-batch negative result is exact only within this saved eight-candidate
affine search boundary and at logical weight eight.  It does not rule out
non-affine presentations or heavier representatives.  The two-batch `sat`
result itself is an explicit witness and does not depend on that boundary.

The absolute paths embedded in `provenance/search_summary.json` record the
original run location and are historical metadata, not runtime requirements.
The included portable solver defaults to archive-relative inputs.

## Syndrome and circuit-fault material

`baseline_schedule.json` is the selected simultaneous depth-eight syndrome
schedule.  `verify_package.py` checks that each layer is a perfect matching on
the 64 data qubits and that all X- and Z-check edges occur exactly once.

The files under `fault_distance/` are retained compact evidence for the
separate claim that the three-round Clifford memory circuit has fault distance
six in the stated Stim noise model.  They are not needed to reconstruct or
verify the static `[[64,8,8]]` code or its two-batch basis.  Large temporary
meet-in-the-middle tables were generated during that analysis and are not
archive inputs.
