# `[[56,8,6]]` `C4 x C2` automorphism-dual BB code

This directory records a rate-`1/7` CSS code for the eight sites of a `4 x 2`
logical grid.  It was found by the relaxed search that imposes no logical
support-disjointness constraint.  The displayed logical basis happens to be
disjoint, but that fact was neither searched for nor used in accepting the
code.

## Construction

Let `G = C4 x C7`, with elements `(x,y)` indexed as `7*x+y`.  For a group-ring
polynomial `c`, define the regular matrix

```text
A(c)[g,h] = coefficient of h-g in c.
```

Use

```text
a = 1 + y + x + x*y^2
b = 1 + y^-1 + x + x*y^-2

H_X = [ A(a)   A(b)   ]
H_Z = [ A(b)^T A(a)^T ]
```

Thus `n=56`, every displayed check has weight 8, both check ranks are 24, and
`k=56-24-24=8`.  The Tanner graph is connected.

Qubits are indexed by

```text
q = half*28 + 7*x + y,
half in C2, x in C4, y in C7.
```

## Physical and logical symmetries

The commuting physical code automorphisms are

```text
T4: (half,x,y) -> (half,x+1,y)
T2: (half,x,y) -> (1-half,x,-y).
```

They have physical and logical orders 4 and 2.  Starting from the Z logical
supported on `(half,x)=(0,0)` for every `y`, their `C4 x C2` orbit spans all
eight logical classes.  In logical site order `site=2*x+half`, these supports
are the eight seven-qubit fibres.

Hadamard duality is implemented by transversal `H` followed by

```text
tau: (half,x,y) -> (1-half,-x,-y).
```

The permutation obeys `row(H_X tau)=row(H_Z)` and the reverse relation.  In
the saved grid basis its logical site permutation is

```text
(x,half) -> (-x,half+1),
```

or `[1,0,7,6,5,4,3,2]` in the saved site order.  It normalizes the grid by
inverting `T4` and commuting with `T2`, so nearest-neighbor Ising bonds remain
trackable.

## Exact distance certificate

The repository's independent CVXPY/HIGHS logical MILP proves that no X or Z
logical has weight at most 5.  Unconstrained HIGHS minimizations find:

```text
Z witness: [0,1,7,33,34,41]
X witness: [2,9,10,29,35,36]
```

Both have weight 6, certifying `d_X=d_Z=6`.

## Syndrome-extraction schedule

The `schedule/` directory contains a simultaneous bare-ancilla schedule with
eight CNOT layers, which meets the weight-eight lower bound.  Every layer is a
56-gate perfect matching using every data qubit and all 56 dedicated syndrome
ancillas exactly once.  A complete round therefore uses 448 CNOTs.

The layers are translation invariant, collision-free, clean under X/Z
cross-ancilla back-action, and related by the nonidentity ZX fold plus time
reversal.  Exact guarded-bulk CNOT-fault analysis gives

```text
d_fault^X = d_fault^Z = 6.
```

See `schedule/README.md`, `schedule/optimal_depth8_schedule.json`, and
`schedule/circuit_fault_certificate.json` for the compact ordering, all 448
indexed CNOTs, and the certificate.

## Files and verification

- `code.json`: human-readable construction, logical data, relations, and
  distance results.
- `code.npz`: `matrix_x`, `matrix_z`, physical permutations, the logical Z
  orbit, and the logical Hadamard permutation.
- `verify.py`: rebuilds the code and reruns all exact checks.
- `verify_with_qldpc.py`: independently rebuilds the matrices and verifies
  stabilizer validity, ranks, canonical logical pairing, and both exact CSS
  distances using qLDPC's native full enumeration (no saved distance results,
  custom MILP, randomized bounds, or assumed X/Z distance equality).
- `construction.py`, `common.py`, `distance.py`: local supporting code.
- `schedule/`: the depth-optimal syndrome schedule, its structural verifier,
  and circuit-fault certificate.

From this directory, with the `qldpc`, `numpy`, `cvxpy`, and HIGHS Python
dependencies available, rerun the full verification with:

```bash
python verify.py --output /tmp/c4xc7-code-check.json \
  --matrix-output /tmp/c4xc7-code-check.npz
```

For the standalone qLDPC verification (tested with qLDPC 0.3.3):

```bash
../.venv/bin/python verify_with_qldpc.py
# Optionally save a fresh JSON report (existing files are never overwritten):
../.venv/bin/python verify_with_qldpc.py --output /tmp/c4xc7-qldpc-verification.json
```

This script constructs `qldpc.codes.CSSCode` directly from the polynomials
above. It calls `get_distance_exact(Pauli.X, cutoff=0)` and then the analogous
Z calculation, with no early exit, and checks `d_X=d_Z=6`. Full enumeration
can take several minutes; progress goes to stderr and the final JSON report
goes to stdout. This verifies **code distance**, not syndrome-circuit fault
distance; the report explicitly leaves `fault_distance` unset. It does not
run Stim or read the separate circuit certificates.
