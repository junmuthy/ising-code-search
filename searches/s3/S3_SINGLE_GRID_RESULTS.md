# Self-dual `S_3 x C_8 x C_4` single-grid search

## Outcome

The targeted search found three connected GALA codes with one clean,
ZX-closed `C_8 x C_4` logical grid.  The best representative has certified
parameters

\[
\boxed{[[384,200,7]]}.
\]

Its candidate ID is
`s3-l4-j1-self-dual-w16-36d018790fa7a28a`.

The two weight-four polynomial entries are

\[
\begin{aligned}
F_0 &= 1+x^6y^2+\tau_1x^4y+\tau_2y,\\
F_1 &= x^2+x^4y^3+\sigma x^4+\sigma x^5y^2.
\end{aligned}
\]

Set `G_0=F_0^dagger` and `G_1=F_1^dagger`.  The active check matrices are
then exactly equal:

\[
H_X=H_Z=[F_0,F_1,F_0^\dagger,F_1^\dagger].
\]

The code therefore has identity physical ZX duality: transversal Hadamard
requires no qubit permutation.

## The 32-qubit logical grid

One logical seed has physical support

```text
[32, 122, 164, 209, 266, 298, 329].
```

Its 32 translates under `C_8 x C_4` satisfy all of the following:

- every representative has weight 7;
- the representatives are pairwise disjoint;
- their combined rank modulo stabilizers is 32;
- physical `x` translation is exactly the cyclic eight-site grid shift;
- physical `y` translation is exactly the cyclic four-site grid shift;
- the same supports are valid X and Z logicals;
- their X-Z pairing matrix is exactly `I_32`.

The last two facts follow especially cleanly from self-duality and odd,
disjoint supports:

\[
\langle z_{a,b},z_{a',b'}\rangle
=\delta_{a,a'}\delta_{b,b'}.
\]

Thus `H^otimes384` implements 32 independent logical Hadamards on this grid,
not Hadamard followed by a swap into a spectator sector.

The grid uses 224 physical qubits and leaves 160 outside its support.  This
slack is what avoids the even-weight saturation obstruction encountered by
the two-grid weight-six ansatz.

## Distance certificate

The symmetry-reduced Tanner search exhaustively found no X- or Z-kernel
vector at weights 1 through 6.  The weight-seven grid seed supplies a
nontrivial logical witness, so the distance is exactly 7.  X and Z distances
are equal because `H_X=H_Z`.

## Ising and STAR conditions

| Requirement | Certified realization |
| --- | --- |
| 32 red or black sites | One canonical `C_8 x C_4` logical grid |
| Nearest-neighbour matchings | Exact physical `x` and `y` translations |
| Global logical Hadamard | Identity transversal `H^otimes384` |
| Inter-block CNOT | Pairwise transversal CNOT between identical CSS blocks |
| Parallel STAR rotations | 32 pairwise-disjoint weight-seven Z supports |
| TMR syndrome parity | Every column has even degree 4 |
| Doubly-even checks | Every check has weight 16 |
| Error-correction distance | Certified `d=7` |

Two identical code blocks, one for each checkerboard colour, require 768
physical data qubits.  This is 128 fewer than two 448-qubit packed BB blocks
when only one simulation copy is needed.

The principal caveat is that the stabilizer code encodes 200 logical qubits,
not exactly 32.  The desired grid is an invariant canonical subsystem under
Hadamard and translations, while 168 logical qubits remain spectators.  A
future gauge-fixing or lower-rate-protograph search would be needed to remove
those spectators.

## Search method and yield

The successful ansatz differs from the earlier block-swapping ZX fold.  It
uses `G_i=F_i^dagger`, filters for active orthogonality and a nonzero latent
row, and then requires the code to be SWEL before doing distance work.  The
SWEL filter removes codes whose all-ones vector is a stabilizer and whose
logical supports are consequently all even.

For seed `260828`, the reproducible run examined 89,550 raw candidates:

| Stage | Count |
| --- | ---: |
| Algebraic survivors | 2,107 |
| SWEL survivors | 268 |
| Certified `d >= 6` | 25 |
| Connected accepted grids | 3 |
| Disconnected grid rejected | 1 |

The three connected hits had distances 6, 7, and 6.  The distance-seven
representative above was the second hit.

## Reproduce

Run the complete search with:

```bash
.venv/bin/python -m searches.s3.run_s3_single_grid_search \
  --target-candidates 3 --raw-limit 150000 --seed 260828 \
  --max-nodes 5000000
```

Recompute only the best-candidate certificate with:

```bash
.venv/bin/python -m searches.s3.analyze_s3_single_grid_candidate
```

Machine-readable artifacts:

- [`results/s3-l4-self-dual-single-grid-seed260828.jsonl`](../../results/s3-l4-self-dual-single-grid-seed260828.jsonl)
- [`results/s3-l4-self-dual-single-grid-seed260828.summary.json`](../../results/s3-l4-self-dual-single-grid-seed260828.summary.json)
- [`results/s3-single-grid-best.json`](../../results/s3-single-grid-best.json)
