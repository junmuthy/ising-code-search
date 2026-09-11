# C4 x C8 search: first feasibility milestone

Date: 2026-09-11. The first bounded milestone is complete. **The requested parameter and logical-symmetry requirements are achievable.** We have an independently certified nonabelian `[[128,32,6]]` construction and a connected, broader-construction `[[128,32,7]]` control. Neither result is a claim of literature novelty or a circuit-level STAR implementation.

## Certified results

| Property | Nonabelian two-block construction | Connected equivariant control |
| --- | --- | --- |
| Parameters | `[[128,32,6]]` | `[[128,32,7]]` |
| Check-space ranks | `48 + 48` | `48 + 48` |
| Displayed checks per Pauli type | `64`, all weight `12` | `48`: `32` weight `18`, `16` weight `14` |
| Physical decomposition | Two `[[64,16,6]]` components | One `128`-qubit component |
| Logical grid | One regular `C4 x C8` action on all `32` logicals | One regular `C4 x C8` action on all `32` logicals |
| Physical translation orders | `4` and `16`; commute modulo central action | Exactly `4` and `8`; commute physically |
| Logical Hadamard | `P H^128`, followed logically by `(i,j) -> (3-i,2-j)` | `H^128` directly, no permutation needed |
| Saved logical representative weights | `Z:43`; `X:33–57`, not optimized | `X:11`, `Z:11` for every grid site |
| Construction scope | Nonabelian group-algebra, square commuting-matrix parent | General equivariant CSS, **not** the square GALA parent |

The logical Hadamard and translations are verified **in the same canonical basis**, with all 64 logical Pauli images checked modulo the appropriate stabilizer space. No disjoint-logical condition was imposed.

The nonabelian construction meets the stated hard requirements, but it is not an indecomposable 128-qubit block. Its physical translations exchange the two components, so the action on all 32 logicals is nevertheless one regular grid. The connected control also passes a row-space/RREF decomposition test, not merely connectivity of the displayed check graph.

### Main artifacts

- [Nonabelian certificate](certified/extension_128_32_6/audit.json), [candidate and logical basis](certified/extension_128_32_6/candidate.json), [binary checks](certified/extension_128_32_6/checks.npz).
- [Connected distance-seven certificate](certified/connected_128_32_7/audit.json), [candidate and logical basis](certified/connected_128_32_7/candidate.json), [binary checks](certified/connected_128_32_7/checks.npz).
- Each certificate directory also contains `logical_bases.npz` and `physical_permutations.json`. Array coordinates and all permutations are zero-based, with permutations mapping **old coordinate to new coordinate**.
- The earlier `connected_128_32` and `connected_128_32_refined` directories retain intermediate audits of the same connected candidate; use `connected_128_32_7` for the final audit.

## Nonabelian construction

Use the order-64 presentation

```
z^2 = 1, z central, [x,y] = z, x^4 = 1, y^8 = z.
```

Elements are `(i,j,t) = x^i y^j z^t`, ordered by `2*(8*i+j)+t`, with `0<=i<4`, `0<=j<8`, `0<=t<2`. Physical coordinates are `(block,g)` with two blocks of 64. Multiplication is implemented and exhaustively associativity-tested in `models.Extension`.

The successful example uses these group-algebra supports:

```
a: (0,0,0), (0,2,0), (1,5,1), (1,7,1),
   (2,0,1), (2,6,0), (3,1,1), (3,7,0)
b: (0,1,1), (0,7,1), (2,1,1), (2,3,1)
```

Equivalently, little-endian coefficient integers are `a=4613955421298753553`, `b=584115585032`. With actual binary left-regular lifts,

```
H_X = [L_a | L_b]
H_Z = [L_b^T | L_a^T].
```

The matrices commute as required. Right multiplication by `x,y,z` supplies the saved permutations. The induced central action is identity on both logical Pauli sectors. Thus the logical action factors through `G/<z> = C4 x C8`, despite the order-16 physical `y` generator.

The Hadamard fold exchanges the two physical blocks and applies the group automorphism `x -> x^3 z`, `y -> y^7 z`, `z -> z`. Its induced logical permutation in the saved basis is `(i,j) -> (3-i mod4, 2-j mod8)`.

The two components have 64 physical qubits and ranks 24 per Pauli type. The translations and Hadamard fold exchange them. The code is not just a claim based on the abstract group: the saved logical orbit has quotient rank 32, and both X and Z shift actions were explicitly checked.

## Connected control construction

This branch deliberately permits check orbits of size 16, rather than requiring all displayed checks to come from free 32-site group-ring rows. It is a useful GALA-inspired comparison, not a nonabelian GALA hit.

Start with four sheets over `A=C4 x C8`, flattened as `32*s+8*i+j`. Use the same X and Z checks:

- 32 pair checks between sheets 0 and 1 at each group coordinate;
- 16 pair checks within sheet 2 between `(i,j)` and `(i+2,j)`;
- sheet 3 initially supplies the 32 canonical logicals.

Then apply two translation-equivariant binary orthogonal transformations to both Pauli sectors. For a 32-row translated array `V` with `V V^T=0`, the transformation is

```
S = I + V^T V,       S S^T = I.
```

It preserves CSS commutation, equal X/Z checks, canonical pairing and physical translations. `fallback.py` reconstructs the final code exactly from this saved recipe:

```
[a, b, order]
[536870913, 1073741824, [3,0,2,1]]
[8388608,   8,          [3,1,2,0]]
```

Here entries before reordering are `[a,a,b,b]`, with each integer a 32-bit coefficient vector over `A`. The resulting code has `H_X=H_Z`, canonical `L_X=L_Z`, and `L_Z L_Z^T=I`. The grid basis consists of 32 translates of a weight-11 seed. Hadamard therefore acts as an individual H on every logical without any physical or logical permutation.

## Distance and independent audit

The search verifier uses complete pair/triple syndrome hashing, including logical signatures to distinguish stabilizers from logical errors. `certify.py` then independently rebuilds arrays and uses `galois.GF(2)` for rank, commutation, canonical pairing, row-space membership, translations, central quotient, and all Hadamard images.

The independent distance check enumerates every zero-syndrome support of weight at most five and tests stabilizer membership directly, without using the search verifier's logical signatures. For both reported codes, there are actually **no nonzero zero-syndrome supports of weight at most five** in either sector.

- Nonabelian code: weight-six Z witness `[0,1,92,93,110,111]`; weight-six X witness `[0,1,18,19,110,111]`. Thus both distances are exactly six.
- Connected code: exhaustive triples-versus-triples enumeration additionally excludes logical weight six. The verified support `[1,21,30,42,64,75,94]` is a weight-seven logical in either sector. Thus both distances are exactly seven.

Separate HiGHS MILPs returned `infeasible` for weight at most five on **both** Pauli sides of each code. The MILP fixes odd pairing with logical zero; this is complete because the independently verified regular logical grid acts transitively on the dual basis. For the connected code, another MILP supplied the weight-seven witness, checked afterward using exact binary arithmetic. Floating-point solver statuses are cross-checks; the exact enumerations and exact witness checks establish the distances. Timeouts or inaccurate infeasibility are never treated as proofs.

## What was searched

| Run | Scope | Outcome |
| --- | --- | --- |
| `gl_capacity` | `48` seed/fold configurations, `20` distinct linear spaces | `42` configurations have insufficient union-row capacity; `6` pass capacity but sparse commuting samples do not attain ranks `48,48` |
| `gl_pilot_w12` | `128` fixed matrices, basis-heavy initial sampler | `164` rank-correct candidates, all with zero quotient norm |
| `gl_mixed_w16` | `512` fixed matrices, full commutants and mixed kernel combinations | `727` rank-correct candidates, all with zero quotient norm |
| `extension_pilot_w12` | Four order-64 presentations, `128` fixed elements | `201` regular-grid candidates, all with explicit weight-two/four logicals |
| `extension_mixed_w16` | Four presentations, `512` fixed elements | `1641` regular-grid candidates, all below the distance threshold |
| `extension_products_w16` | Noncentral products and sums, `1024` fixed elements | `62949` evaluated candidate presentations; `6146` rank-correct; `1878` regular-grid; `93` pass distance threshold; `61` also pass the common-basis Hadamard test |
| `general_equivariant_w16` | `256` broader-control trials, weight-16 check ceiling | `20` within ceiling, all low-distance; others overweight |
| `general_equivariant_dense_control` | Same `256` recipes, check ceiling removed for a feasibility control | `175` threshold-passing presentations; lightest connected examples have maximum check weight `18` |

The successful extension run contained 33 accepted presentations with weight-12 checks, 18 with weight-14 checks, and 10 with weight-16 checks. All 61 have two 64-qubit components. These are **presentations, not 61 proven inequivalent or novel codes**. Equivalence classification has not been performed. The connected control's extra weight tier is a comparison only; it is not counted as a success of the weight-16 nonabelian search.

The decisive change was allowing noncentral factor products, including products of three augmentation binomials, rather than only central-factor seeds. The first reported nonabelian hit has `rank(L_a)=44`; its partner raises both check ranks to 48. The earlier common-factor family often obtained the desired logical quotient only with surviving short fibre operators.

The union-rank bounds close only their explicitly imposed displayed-row seed/fold spaces. The failed GL samples are not a no-go theorem for all faithful GL constructions. Likewise, no general obstruction at `n<=130` is asserted. The 20 distinct linear spaces were deduplicated by their exact kernel, not classified up to every physical equivalence.

## Reproduction

All new work is isolated here. The existing `/home/judah_unmuth/gala-code-search` checkout and its untracked artifacts were left unchanged. Its Python environment was reused, and its saved `[[160,32,6]]` control was reconstructed and compared with the exact `checks.npz` from Git revision `8481d1e`. That control also passed the full grid, common-basis Hadamard and exact distance checks.

From this directory, with the existing environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/gala-n128-numba.XXXXXX)"
/home/judah_unmuth/gala-code-search/.venv/bin/python -m pytest -o addopts='' test_search.py -q -p no:cacheprovider

/home/judah_unmuth/gala-code-search/.venv/bin/python certify.py \
  certified/extension_128_32_6/candidate.json --output /tmp/gala-extension-audit-new
/home/judah_unmuth/gala-code-search/.venv/bin/python certify.py \
  certified/connected_128_32_7/candidate.json --output /tmp/gala-connected-audit-new

/home/judah_unmuth/gala-code-search/.venv/bin/python search.py \
  --branch extension --extension-products --fixed 1024 --per-fixed 128 \
  --maxweight 16 --seconds 600 --seed 20260913 --output /tmp/gala-extension-run-new
```

Choose fresh output directories; the drivers refuse to overwrite existing journals/certificates. Seventeen tests cover binary algebra, all four group presentations, actual binary lifts/transposes, positive and negative logical controls, same-basis Hadamard recovery, brute-force distance comparisons, degeneracy, capacity equations, reconstruction of both reported results, and exact distance seven for the connected code.

Observed environment: Python 3.12, NumPy 2.5.2, SciPy 1.18.1, galois 0.4.11, CVXPY 1.9.2, HiGHS 1.15.1, Z3 5.1.0.0, pytest 9.1.1. Search checkout revision: `b1271109039529901b31f6de9a0b375209642489`.

Each run retains a candidate journal, RNG seed, parameter record, and source hashes. All accepted candidates include full matrices and logical/permutation certificates. The two bootstrap weight-12 runs used the initial basis-heavy sampler; current mixed-sampler code is not expected to reproduce those bootstrap counts. The saved candidate inputs remain exactly reconstructible. Later witness and certificate upgrades are retained separately rather than rewriting the search journal's original lower-bound claims.

## Next decision point

The recommended continuation is to prioritize the connected distance-seven control for check-weight reduction and STAR logical-operator optimization, while seeking an indecomposable nonabelian code with comparable check sparsity. A quick check of all products of up to three displayed stabilizer generators did not lower the connected code's maximum generator weight below 18: rows of weight at most 16 spanned only rank 16.

Not yet done: equivalence/novelty classification, optimized logical representatives or injection batches, a transversal-S audit, CNOT routing optimization, syndrome-extraction circuits, decoder benchmarks, or circuit fault-distance certification. The current certificates are static code and logical-action certificates, not STAR performance claims. No search process is left running.
