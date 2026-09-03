# `[[22,2,6]]` shortened Golay code

This directory studies the shortened quantum Golay code in its cyclic
MCR/generalized-bicycle presentation.  It is kept separate from all earlier
GALA searches and reference-code data.

The defining polynomials over `F_2[x]/(x^11-1)` are

\[
f=1+x,\qquad p=1,\qquad
q=x+x^3+x^4+x^5+x^9.
\]

With `a=pf` and `b=qf`,

\[
H_X=[\operatorname{circ}(a)\mid\operatorname{circ}(b)],\qquad
H_Z=[\operatorname{circ}(b)^T\mid\operatorname{circ}(a)^T].
\]

There are eleven displayed checks of each CSS type, one redundant relation of
each type, and every displayed check has weight eight.  The universal bicycle
fold `L_i <-> R_{-i}` maps `H_X` to `H_Z`.

The `code_data` package constructs and exhaustively validates the matrices,
distance, fold, and disjoint logical representatives.  The `schedule` package
exhaustively enumerates translation-invariant simultaneous `X/Z` extraction
schedules.  Circuit-level fault-distance work belongs under
`stim_fault_distance`.

`N22_K2_D6_ISING_REFERENCE.md` is the consolidated physical and logical
reference, including the exact schedule, TMR decomposition, measured
post-selection probability, and parallel-factory completion times.  The
reproducible ClifT implementation and its focused result report are under
`tmr_postselection`.  The Appendix-C-style teleportation fidelity estimator,
joint terminal decoder, and full stochastic RUS analysis are under
`teleportation`.
