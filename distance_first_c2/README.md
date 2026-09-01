# Distance-first `C₂` Ising-code search

This search reverses the preceding fixed-geometry workflow.  It first searches
for a `[[n,2,6]]` CSS stabilizer pair and does **not** prescribe logical
representatives, a physical `C₂` translation, or a ZX fold.  Those structures
will be recovered from distance-six survivors in later post-processing stages.

The first target is `n=16`, with

\[
\operatorname{rank}(H_X)=\operatorname{rank}(H_Z)=7.
\]

Each move replaces one direction of one stabilizer subspace by another vector
in the orthogonal complement of the opposite subspace.  CSS orthogonality and
`k=2` therefore hold throughout the search.  Distances are evaluated exactly
by enumerating the two logical quotient spaces.

For each state, all 127 nonzero stabilizers of each type are also enumerated.
A greedy binary-matroid basis calculation determines the best check-weight
profile available in that stabilizer space; the search does not confuse an
arbitrary dense generator basis with an intrinsically high-weight code.

## Validation

Before searching, run:

```bash
python -m distance_first_c2.validate_reference
```

This must recover `[[32,4,6]]` and maximum check weight eight from the saved
reference code.

## Pilot

Every run uses a new directory and refuses to overwrite existing results:

```bash
python -m distance_first_c2.run_search \
  --length 16 \
  --output-root results/distance-first-c2 \
  --run-name n16-pilot-v1
```

Completed restarts are written immediately, while `progress.json` and the
terminal checkpoint report the best exact distance and cumulative work.

## Exact systematic pass

Every balanced `[[n,2,d]]` CSS code is equivalent under a qubit permutation to
one with

\[
H_X=[I\mid A],\qquad B=[I\mid C],\qquad H_Z=[BA^T\mid B].
\]

The exact solver uses this complete parameterization and directly excludes
every logical operator of weights one through five:

```bash
python -m distance_first_c2.run_systematic_solver \
  --length 16 \
  --output-root results/distance-first-c2 \
  --run-name n16-systematic-exact-v1
```

A final `sat` result produces a `[[16,2,>=6]]` candidate.  A final `unsat`
result proves that no binary CSS code with those parameters exists.  `unknown`
is only a solver timeout.

The monolithic formulation timed out at the `d >= 5` stage.  Splitting its 14
`C` entries into all 70 canonical row-pattern sectors resolved every sector as
UNSAT.  See `RESULTS.md` and the saved fixed-`C` run for the complete `n=16`
computational no-go.
