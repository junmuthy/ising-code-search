# Connected [[64,16,6]] with lower total check support

This is the distance-six alternative identified for resource-state studies:
candidate `d9173bb8cd878461c62a0dd5b7f9cb5f494b401aaf662cf40e45eb883551ebcf`.
It retains a regular logical `C4 x C4` grid and individual logical Hadamards
up to permutation in one canonical CSS basis. Logical supports may overlap.
It is the later sparse-search code, not the original weight-14 component
extracted from a nonabelian 128-qubit parent.

Two presentations of the same connected code are preserved:

| Property | Default independent presentation | Translation-closed alternative |
| --- | --- | --- |
| Checks per Pauli type | `4` of weight `8`, `20` of weight `12` | `32` of weight `12` |
| Check rank per Pauli type | `24` | `24` |
| Total check support, both types | `544` | `768` |
| Maximum check weight | `12` | `12` |
| Translations permute displayed checks | Generally only modulo row space | Yes |
| Data location | This directory | `alternatives/translation_closed/` |

The check-weight enumeration certificate records exactly four weight-eight
stabilizers and no other nonzero stabilizers below weight twelve in each
sector. Their rank is four; weight-twelve rows raise it to 24. Thus maximum
check weight twelve and total independent support 544 are optimal for this
code's CSS generators. This is an archived fixed-code certificate, not a
bound on other codes or a circuit-level resource estimate.

Saved logical representatives have weights `X: 9,13,19` (counts `4,8,4`)
and `Z: 9` (all 16). These representatives are not support-optimized. In
particular, the all-weight-ten logical basis belongs to the separate
[`[[64,16,8]]`](../n64_k16_d8/README.md) code.

## Construction and gates

Use two sheets over `C8 x C4`, with physical index `32*s+4*i+j`.
Little-endian coefficient integers are `a=1319523`, `b=67596`. Form the
binary left-regular lifts and set `H_X=[L_a | L_b]`, `H_Z=P(H_X)`, where

```text
P(0,i,j) = (1, 3*i mod 8, -j mod 4)
P(1,i,j) = (0, 3*i+4 mod 8, -j mod 4).
```

This is an abelian fold-constrained construction, not the conventional
square commuting-matrix parent. The physical translations commute and
have orders eight and four; the corresponding logical orders are four
and four. The fold has physical order four, with `P^2=Tx^4` acting trivially
on logicals. In logical grid coordinates modulo four,

```text
Tx: (i,j) -> (i+1,j)
Ty: (i,j) -> (i,j+1)
P H^64: individual Hadamards, then (i,j) -> (-i,2-j).
```

## Distance and backup contents

Both Pauli distances are exactly six. Explicit logical witnesses are
`X: [0,2,12,14,48,50]` and `Z: [0,2,32,34,36,38]`; exhaustive syndrome
matching excludes nontrivial supports of weight at most five.

Both presentation folders contain the original candidate, NPZ checks and
logical bases, distance audit, normalized permutations, support lists,
metadata and provenance. `certificates/check_weight_enumeration.json`
and `certificates/check_presentation_report.json` preserve the original
optimization results and basis changes. Original source paths in archived
reports are provenance only.

The [package verification command](../README.md#standalone-verification)
checks the default presentation and recomputes its exact distance. The
regression suite also verifies the translation-closed presentation, equality
of both check spaces and logical bases, check-orbit closure, and the fold
square relation. It does not rerun the full stabilizer-weight enumeration.
