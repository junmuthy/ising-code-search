# `C2 x C7` BB search for `[[28,4,6]]`

## Target geometry

For the minimal two-block BB construction over `C2 x C7`, the block length is

\[
n=2|C_2 x C_7|=28.
\]

The `C7` subgroup supplies weight-seven fibre logicals and the quotient `C2`
supplies the logical translation.  Each physical BB half contains one logical
`C2` cycle, giving four logical qubits total:

\[
(L_0,L_1),\qquad(R_0,R_1).
\]

The four supports are pairwise disjoint, have weight seven, and have identity
ZX pairing.  Translation by the `C2` generator swaps `L_0<->L_1` and
`R_0<->R_1`.

## Natural shortened bicycle-chain code

Reducing the known `C4 x C7` bicycle-chain polynomial to `C2 x C7` gives

\[
a=1+y^3+xy^2+xy^4,\qquad b=a^\dagger.
\]

qLDPC certifies this code as `[[28,4,5]]`, with weight-eight checks.  It has
`H_X=H_Z`, exact transversal Hadamard, a connected Tanner graph, and the four
complete disjoint fibre logicals—but not distance six.

## Exhaustive weight-eight search

For a polynomial to annihilate the `C7` fibre sum, it must contain an even
number of monomials in each `C2` coset.  A connected four-term polynomial must
therefore have two terms in each coset.  Up to an overall translation, every
such polynomial has the form

\[
a=1+y^r+xy^s+xy^t.
\]

All 126 normalized supports were exactly enumerated.  Of these, 120 had the
complete connected `[[28,4]]` logical structure:

| Exact distance | Count |
| --- | ---: |
| `2` | 48 |
| `4` | 48 |
| `5` | 24 |
| `>=6` | 0 |

## Exhaustive weight-twelve search

For six-term polynomials, connected fibre-compatible supports have a `(2,4)`
split between the two `C2` cosets, up to exchanging the cosets.  All 210
normalized supports were exactly enumerated.  Every candidate had the desired
connected `[[28,4]]` structure:

| Exact distance | Count |
| --- | ---: |
| `2` | 18 |
| `4` | 144 |
| `5` | 48 |
| `>=6` | 0 |

## Conclusion and scope

There is no `[[28,4,6]]` code in the complete self-dual paired-polynomial
`C2 x C7` BB family with stabilizer weight at most 12.  The best distance is
five at both weights eight and twelve.

This does not rule out every general BB code of length 28.  In particular, it
does not classify independent polynomial pairs `a != b^dagger` whose ZX
duality uses the generic BB fold rather than `H_X=H_Z`.  Such a broader search
would preserve a fold-transversal Hadamard but could exchange the two physical
halves at the logical level.

## Artifacts

- `results/c2xc7-bb/paired-weight8-exhaustive-v1/summary.json`
- `results/c2xc7-bb/paired-weight8-exhaustive-v1/candidates.jsonl`
- `results/c2xc7-bb/paired-weight12-exhaustive-v1/summary.json`
- `results/c2xc7-bb/paired-weight12-exhaustive-v1/candidates.jsonl`
