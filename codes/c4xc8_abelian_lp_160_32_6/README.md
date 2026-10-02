# Connected `[[160,32,6]]` `C4 x C8` automorphism-fold code

## Result

This directory contains a new connected CSS code with the following certified
properties:

| Property | Value |
|---|---:|
| Parameters | `[[160,32,6]]` |
| `X`- and `Z`-check weight | exactly `9` |
| Physical/logical translation group | regular `C4 x C8` |
| Canonical CSS-preserving Tanner automorphism group | exactly `C4 x C8` (order `32`) |
| ZX-duality | involutive physical fold `tau` |
| Logical Hadamard | `H` on all `32` logicals, followed by a permutation |
| Tanner graph | connected |
| Translated minimum-logical orbit | `32` weight-six representatives of rank `32` per Pauli type |
| Disjoint representatives in that orbit | `8` independent `X` and `8` independent `Z` representatives |
| Disjoint symplectic pairs in that orbit | `4` |

The code improves the previous explicit `n=224` upper bound for this target to
`n=160`.  It is the smallest code found in this search, not a proof that no
code exists at `n<160`.

## Construction

Let

```text
G = C4 x C8 = <x,y | x^4 = y^8 = [x,y] = 1>.
```

Use the five-block lifted/balanced-product CSS construction of Hong,
[arXiv:2607.27644](https://arxiv.org/abs/2607.27644), with

```text
a = 1 + x^3 y^3 + x^3 y^6
b = 1 + x^2 y^4 + x^3
phi(x) = x^3 y^2
phi(y) = y
c = phi(a^dagger) = 1 + x^3 y^7 + x^3 y^4
d = phi(b^dagger) = 1 + x^2 + x^3 y^2.
```

Here `dagger` inverts every group element.  The automorphism `phi` is a
nonidentity involution.  With `A,B,C,D` the regular-representation matrices of
`a,b,c,d`, respectively,

```text
H_X = [ A  0  B  0  C^T ]
      [ 0  A  0  B  D^T ]

H_Z = [ C  D  0  0  A^T ]
      [ 0  0  C  D  B^T ].
```

Both matrices have shape `64 x 160`, rank `64`, and row weight `9`, so the
code encodes exactly `160-64-64=32` qubits.  All four trinomials are units in
the group algebra of the 2-group `C4 x C8`; this also yields the explicit
one-generator symplectic logical basis stored in `logical_bases.npz`.

## `C4 x C8` action and Hadamard

Simultaneous translation by `x` or `y` in every one of the five physical
blocks preserves both check spaces.  On the systematic logical basis these
are the regular shifts

```text
(r,s) -> (r+1,s),       order 4
(r,s) -> (r,s+1),       order 8.
```

They commute and act freely and transitively on all 32 logical labels.
`pynauty` independently finds that the full canonical CSS-type-preserving
Tanner automorphism group has order `32`, so it is exactly this translation
group.

Define `theta(g)=phi(g^{-1})` and exchange physical blocks 2 and 3 (zero-based
blocks 1 and 2).  The resulting involution

```text
tau = P_theta^(direct sum 5) Pi_23
```

maps `H_X` to `H_Z` and conversely.  Applying Hadamard to every physical qubit
and then `tau` acts on the logical basis as

```text
Z_(r,s) -> X_(r,-2r-s mod 8)
X_(r,s) -> Z_(r,-2r-s mod 8).
```

Thus it is logical `H` on every encoded qubit followed by a known involutive
permutation.  If X- and Z-check colors are allowed to exchange, the canonical
Tanner automorphism group has order `64`.

## Exact distance

Two independent methods certify both CSS distances:

1. A complete syndrome meet-in-the-middle enumeration excludes every
   nontrivial logical support of weights `1` through `5`.
2. Separate CVXPY/HiGHS MILPs are infeasible at weight at most `5` and return
   optimal weight-`6` logicals on both Pauli sides.

The stored MILP witnesses are:

```text
Z: [34, 47, 56, 105, 108, 125]
X: [84, 87, 91, 110, 119, 126]
```

The abelian-syzygy theorem in Appendix D.1 of the source construction also
gives the analytic upper bound `d<=6`, consistent with these witnesses.

## Disjoint logical subsets

The translated orbit of

```text
Z seed: [0, 8, 20, 64, 74, 77]
X seed: [0, 14, 16, 32, 41, 44]
```

contains 32 distinct minimum-weight logical representatives and has logical
rank 32 for each Pauli type.  Within each orbit, an exact maximum-clique audit
finds at most eight pairwise-disjoint representatives, and an explicit set of
eight of each type is stored in `logical_supports.json`.

Four logical qubits can simultaneously be chosen so that their four `X`
supports are pairwise disjoint, their four `Z` supports are pairwise disjoint,
and the `4 x 4` cross-pairing matrix is the identity.  This is an orbit-level
result; it does not claim a global optimum over arbitrary stabilizer dressings.

## Stabilizer schedule and circuit fault distance

The `schedule` directory contains a simultaneous bare-ancilla extraction with
provably optimal CNOT depth `12`.  Every layer contains exactly `96` CNOTs,
and a complete round uses `1,152` CNOTs and `128` dedicated check ancillas.
The layers retain `C4 x C8` translation symmetry, are collision free, obey the
ZX fold under time reversal, and pass exact cross-ancilla back-action tests.

In the isolated guarded-bulk CNOT fault model, exact pair
meet-in-the-middle calculations independently exclude all logical failures of
one through four faults in both memory bases.  Explicit six-fault witnesses
give

```text
5 <= d_fault^X,d_fault^Z <= 6.
```

Thus the requested circuit fault-distance lower bound of five is certified.
See `schedule/README.md` for the compact twelve-layer schedule, scope of the
fault model, and full reproduction commands.

## Files

- `code.py`: self-contained construction and artifact generator.
- `verify.py`: algebra, automorphism, logical-action, disjointness, and exact
  distance checks.
- `search.py`: deterministic randomized search that reproduces the candidate.
- `checks.npz`: dense binary `H_X` and `H_Z` arrays.
- `logical_bases.npz`: the weight-10 symplectic logical bases and translation/fold permutations.
- `presentation.json`: group-polynomial presentation and indexing convention.
- `distance_certificate.json`: independent exact-distance results.
- `automorphism_certificate.json`: translation, fold, and Tanner-group results.
- `logical_supports.json`: minimum-logical orbits and disjoint subsets.
- `schedule/`: optimal-depth schedule, Stim circuit builder, and circuit-fault
  certificates.

Run the complete verification in the `gala-code-search` environment with:

```bash
NUMBA_CACHE_DIR=/tmp/lp160-numba-cache ../../.venv/bin/python verify.py
```
