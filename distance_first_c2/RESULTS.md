# Distance-first small `C₂` search results

## `n=16` pilot

The first distance-only pilot searched commuting rank-seven X and Z stabilizer
spaces.  It did not prescribe logical supports, a physical `C₂` action, or a ZX
fold.

Run:

`results/distance-first-c2/n16-distance-first-32x2000-p8-260831-v1`

The 32 restarts performed 471,689 exact candidate evaluations in 67.3 seconds.
The best endpoint distribution was:

| Exact endpoint distance | Restarts |
|---:|---:|
| `3` | 20 |
| `4` | 12 |

No `d=5` or `d=6` code was found.  This is a stochastic pilot, not a
nonexistence result.

The strongest saved `d=4` endpoint is `restart-0011.json`.  It has exact
`d_X=d_Z=4` and admits generator bases with weights

\[
(4,5,5,5,5,5,5)
\]

and

\[
(5,5,5,5,5,5,6).
\]

An independent enumeration over all 65,536 binary operators confirms its
parameters.  The pilot suggests that the next `n=16` pass needs a more global
move or an exact systematic-form solver; simply extending this greedy run is
unlikely to be the best use of computation.

## Exact systematic pass

Run:

`results/distance-first-c2/n16-systematic-exact-d6-260831-v1`

The solver used the complete systematic form

\[
H_X=[I\mid A],\qquad B=[I\mid C],\qquad H_Z=[BA^T\mid B],
\]

which represents every balanced `[[16,2,d]]` CSS code up to a qubit
permutation.  It directly constrained every physical operator at each weight.

The staged results were:

| Excluded operator weights | Solver result | Exact model distance | Solve time |
|---|---|---:|---:|
| `1` | `sat` | `2` | 0.002 s |
| `1–2` | `sat` | `3` | 0.004 s |
| `1–3` | `sat` | `4` | 0.024 s |
| `1–4` | `unknown` | — | 600.001 s |

Thus the run neither found nor ruled out a `[[16,2,5]]` or `[[16,2,6]]` CSS
code.  It localizes the first hard exact decision to eliminating all
weight-four logical operators.  The next exact pass should split on the 14
variables in `C` and add coordinate-permutation symmetry breaking rather than
merely extending the monolithic timeout.

## Complete fixed-`C` no-go at `n=16`

Run:

`results/distance-first-c2/n16-c-sectors-70-d6-260831-v1`

The 14 binary entries of the `7 x 2` matrix `C` give 16,384 raw matrices.
Quotienting row permutations and exchange of the two residual columns leaves
70 canonical sectors.  Their raw multiplicities sum exactly to

\[
2^{14}=16{,}384.
\]

All 70 sectors completed without a timeout or worker error.  Every sector was
exactly `unsat` after excluding logical operators of weights one through four:

| Result | Sectors |
|---|---:|
| `d >= 5` is `unsat` | 70 |
| `unknown` | 0 |
| worker error | 0 |

Consequently,

\[
\boxed{\text{no balanced binary CSS }[[16,2,d\geq5]]
\text{ code with }r_X=r_Z=7\text{ exists}.}
\]

Unequal-rank CSS `[[16,2,*]]` codes are outside this computation.  They cannot
possess the required permutation ZX duality because a permutation cannot map
stabilizer spaces of different dimensions.  Thus the desired ZX-dual
`[[16,2,6]]` Ising code is impossible even before requiring physical `C2`
symmetry, disjoint logical representatives, Tanner connectivity, or low-weight
checks.  The complete sector run took 67.6 seconds with four workers.
`validation.json` independently checks sector coverage, canonical counts, raw
multiplicity, exact statuses, and absence of missing or errored tasks.

## Complete `n=18` result

Primary run:

`results/distance-first-c2/n18-c-sectors-95-d6-260831-v1`

Direct-`d6` supplement:

`results/distance-first-c2/n18-c-sectors-unresolved12-direct-d6-260831-v1`

For `n=18`, ZX balance requires `r_X=r_Z=8`.  The `8 x 2` matrix `C` has
65,536 raw assignments and 95 canonical sectors.  The primary staged run found:

| Primary result | Sectors |
|---|---:|
| `d >= 5` is `unsat` | 76 |
| `d >= 5` is `sat`, then `d >= 6` is `unsat` | 7 |
| `d >= 5` is `unknown` at 60 s | 12 |

Increasing the 12 unknown sectors to 300 seconds left all 12 unresolved at
`d >= 5`.  Solving the stronger `d >= 6` condition directly then resolved all
12 as exact `unsat` in 84.5 seconds total.  Combined coverage is therefore:

| Combined `d >= 6` result | Sectors |
|---|---:|
| `unsat` | 95 |
| `unknown` | 0 |
| `sat` | 0 |

Consequently,

\[
\boxed{\text{no balanced binary CSS }[[18,2,d\geq6]]
\text{ code with }r_X=r_Z=8\text{ exists}.}
\]

At least seven canonical sectors do contain exact `[[18,2,5]]` CSS codes.
Thus `n=18` improves on the `n=16` distance-four ceiling, but cannot reach the
required distance six even before imposing physical ZX and `C2` permutations,
disjoint logical supports, Tanner connectivity, or a check-weight limit.
