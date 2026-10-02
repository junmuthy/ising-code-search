# Four-copy abelian packed-code search

Search date: 2026-08-25

## Outcome

The first scalable abelian tier already produces an exact success:

\[
\boxed{[[896,128,7]]}.
\]

It is a self-dual BB/minimal-GALA code over `C_112 x C_4`, with
`x^112 = y^4 = 1` and

\[
\begin{aligned}
a
&=(1+x^{16})+x(1+x^{48})+x^{16}y(1+x^{32})\\
&=1+x+x^{16}+x^{16}y+x^{48}y+x^{49}.
\end{aligned}
\]

Taking `b=a^dagger` gives `H_X=H_Z=[A | A^T]`. Consequently physical `H`
on every data qubit exchanges the canonical logical `X` and `Z`
representatives exactly.

## Four disjoint logical grids

The thickness subgroup is `D=<x^16>`, isomorphic to `C_7`, and its orbit sum
is

\[
\omega_7=1+x^{16}+x^{32}+x^{48}+x^{64}+x^{80}+x^{96}.
\]

For physical half `h in {L,R}`, copy label `c in {0,1}`, and grid coordinate
`(a,b) in C_8 x C_4`, the canonical support is

\[
z_{h,c,a,b}=x^{c+2a}y^b\omega_7
\]

on half `h`. These 128 supports have weight 7, are pairwise disjoint, and
partition all 896 physical data qubits. They are a complete orthonormal
logical basis, not merely a selected subspace of a higher-dimensional code.

The verified logical translation action is:

- `x^2`: a `C_8` shift of `a` within each of the four grids;
- `y`: a `C_4` shift of `b` within each grid;
- `x`: the full `C_16` action that interleaves the two copies on each half.

The translations act synchronously on all packed copies. The four logical
sectors and their STAR supports are independent, but this construction does
not claim a separate physical translation switch for each sector.

## Certified properties

| Property | Result |
|---|---:|
| Physical qubits | `896` |
| Logical qubits | `128` |
| Exact distance | `7` |
| Check weight | `12` |
| Qubit degree per CSS type | `6` |
| Rank of `H_X=H_Z` | `384` |
| Logical grids | `4 x 32` logical qubits |
| Canonical logical weight | `7` |
| Cross-support overlap | `0` |
| Tanner graph | connected |
| Tanner four-cycles per CSS graph | `6,720` |
| Physical `x`, `x^2`, and `y` translations | verified automorphisms |
| Half-swap plus inversion | verified automorphism |
| Syndrome CNOT layers per CSS type | `12` (optimal) |

HiGHS independently minimized a representative nontrivial logical class on
each physical half. Both minima are 7. Translation symmetry relates all 64
classes within a half, while the weight-7 fibres give the matching upper bound,
so the global distance is exactly 7.

## Search statistics

The normalized connected weight-12 family was

\[
\begin{aligned}
a={}&(1+x^{16r_0})
 +x^{1+16p_1}(1+x^{16r_1})\\
 &+x^{16p_2}y(1+x^{16r_2}),
\end{aligned}
\]

with nonzero pair separations modulo 7. After quotienting support translations
and reflections, the full sweep contained 1,323 representatives:

- 1,293 passed every algebraic, logical-module, and connectivity check;
- 30 had accidental extra logical sectors and failed the exact-rank target;
- the accepted Tanner graphs had four-cycle counts `6,720`, `10,304`, or
  `17,472`;
- 282 candidates occupied the best `6,720` tier, including the certified seed.

The seed was retained because it already attains the logical-weight distance
ceiling and the best observed four-cycle tier.

## Explicit syndrome schedule

Each of the six monomials in `a` lifts to a permutation matrix on the left
physical half. The six inverse monomials in `a^dagger` do the same on the
right half. Scheduling one monomial per layer therefore yields 12 layers,
each with 448 simultaneous CNOTs and no shared check ancilla or data qubit.

The machine-readable certificate verifies that these layers cover every edge
of `H_X` exactly once. The same schedule applies to `H_Z` because `H_X=H_Z`.
Twelve layers is optimal in the one-ancilla-per-check model because a
weight-12 stabilizer requires 12 sequential ancilla interactions. Ancilla
preparation, measurement, hardware routing, hook-error ordering, and circuit
fault distance are not included in this static depth statement.

## Rate and architecture interpretation

The rate is `k/n = 128/896 = 1/7`. This saturates the elementary bound
`n >= k d` imposed by 128 disjoint logical representatives at distance 7.
Scaling from the `[[448,64,7]]` code therefore improves the number of
simulations packed into one stabilizer code, but cannot improve the asymptotic
encoding rate while retaining disjoint weight-7 supports.

For the bipartite Ising architecture, use two identical `[[896,128,7]]` code
blocks, one for each checkerboard color. Pairing corresponding grids between
the two code blocks gives four parallel 64-site simulations. The total is
1,792 physical data qubits, or 448 data qubits per simulation, before factory
ancillas.

## Reproducibility artifacts

- `searches/many_copy/run_many_copy_search.py`: cached parallel structural sweep.
- `searches/many_copy/certify_many_copy.py`: exact distance, automorphism, and schedule certificate.
- `results/many-copy/weight12-r2-m7.jsonl`: all 1,323 sweep records.
- `results/many-copy/natural-seed-certificate.json`: complete recommended-code certificate.
- `gala_search/many_copy.py`: construction and verification library.
- `tests/test_many_copy.py`: regression and exact-distance tests.
