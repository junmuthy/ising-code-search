# Joint-ZX revisit of the earlier `S_3` searches

## Question

The original search scored each translated 32-logical grid `V_i` using only
its diagonal ZX pairing

\[
B_{ii}=V_i(PV_i)^T.
\]

The successful weight-12 `L=12` search showed that `B_ii=0` is not fatal:
several grids can have a full-rank combined matrix through off-diagonal blocks
`B_ij`.  This report applies that corrected criterion to every relevant saved
weight-16 `L=4` and `L=8` candidate.

## Method

For every candidate, the revisit:

1. exhaustively enumerated every graph-supported weight-six kernel orbit up to
   `C_8 x C_4` translation;
2. retained its 32 translated supports and logical rank modulo stabilizers;
3. tested all mutually disjoint grid combinations allowed by the available
   orbits;
4. computed both their combined logical rank and full block ZX pairing;
5. separately searched graph-supported weight-seven grids in every saved
   candidate previously known only to have `d >= 7`.

The exact-weight enumeration is complete here because the original distance
screens already proved that no smaller zero-syndrome support exists.  Thus an
exact-weight solution cannot be hidden behind a disconnected smaller kernel
component.

## Named candidates

| Candidate | Complete grid enumeration | Best selected sector | Logical rank | ZX rank | Result |
| --- | --- | --- | ---: | ---: | --- |
| `[[384,194,6]]` `s3-l4-j1-w16-e1bb058d91825488` | exactly one weight-six orbit | one grid | `32/32` | `0/32` | not rescued |
| `[[768,388,6]]` `s3-l8-j2-w16-b77391bd2f2fd82b` | exactly two weight-six orbits | two disjoint grids | `60/64` | `56/64` | not rescued |

The compact candidate's only grid has rank-zero self-pairing under both its
known structured ZX folds.  There is no second graph-supported weight-six
orbit to serve as a clean partner.

The extended near miss really is exhaustive rather than a greedy accident:
its two previously reported grids are the only graph-supported weight-six
translation orbits.  They are disjoint, but four logical relations reduce the
combined logical rank to 60 and the ZX form has rank 56.

## Full saved batches

| Family | Saved exact-`d=6` codes | Enumerated weight-six grid orbits | Self-pairing ranks | Full-rank joint sectors |
| --- | ---: | ---: | --- | ---: |
| `L=4,J=1,n=384` | `20` | `20` | `0` for 17; `16` for 3 | `0` |
| `L=8,J=2,n=768` | `14` | `16` | `0` for all 16 | `0` |

In the compact batch, no candidate had two mutually disjoint grid orbits at
all.  In the extended batch, five candidates had a disjoint pair, but none was
logically independent with rank 64.  The best candidate was the named
`[[768,388,6]]` near miss at logical rank 60 and ZX rank 56.  No extended
candidate contained three graph-supported weight-six orbits.

The previously unresolved higher-distance subset was also exhausted:

| Family | Saved candidates with no kernel through weight six | Graph-supported weight-seven grid orbits | Completed without cutoff |
| --- | ---: | ---: | ---: |
| `L=4` | `9` | `0` | `9/9` |
| `L=8` | `23` | `0` | `23/23` |

This does not assert that those codes have no arbitrary weight-seven logical;
it asserts the relevant stronger-structure result that they have no
graph-supported weight-seven seed whose translations give disjoint Ising
logicals.

## Saturation obstruction

The corrected off-diagonal criterion does not remove the earlier even-weight
saturation proof.  Two weight-six grids in 384 qubits, or four in 768 qubits,
would partition every physical qubit.  The folded `X` supports would also form
a partition, so every row of the complete pairing would have parity

\[
6\bmod 2=0.
\]

The all-ones vector would therefore remain in the pairing kernel.  Hence the
saturated useful rate `1/6` is impossible for these even-weight grid supports,
regardless of whether the nonzero pairing is diagonal or off diagonal.

The new `[[1152,580,6]]` result avoids this obstruction because its four grids
occupy only 768 of 1152 physical qubits.  That slack permits the off-diagonal
pairing to be a full 128-by-128 permutation matrix.

## Conclusion

The discovery changes the correct search rule but does not rescue the saved
`L=4` or `L=8` candidates.  Their earlier negative conclusion now has a
stronger basis:

- the relevant translation-grid orbits have been exhaustively enumerated;
- all disjoint combinations have been tested jointly;
- the unresolved weight-seven candidates contain no suitable translated grid.

Future searches should always score sets of disjoint grids by the full block
pairing matrix.  They should also retain physical-support slack instead of
targeting saturated even-weight packing.

## Reproduction

Reproduce the two named candidates with:

```bash
.venv/bin/python revisit_s3_joint_pairing.py --family l4-w16
.venv/bin/python revisit_s3_joint_pairing.py --family l8-w16
```

Reproduce the complete saved batches with:

```bash
.venv/bin/python revisit_s3_saved_batch.py --family l4-w16
.venv/bin/python revisit_s3_saved_batch.py --family l8-w16
.venv/bin/python revisit_s3_saved_batch.py --family l4-w16 --logical-weight 7
.venv/bin/python revisit_s3_saved_batch.py --family l8-w16 --logical-weight 7
```

Machine-readable artifacts:

- [`results/s3-compact-joint-pairing.json`](results/s3-compact-joint-pairing.json)
- [`results/s3-extended-joint-pairing.json`](results/s3-extended-joint-pairing.json)
- [`results/s3-l4-w16-seed240828-joint-pairing.jsonl`](results/s3-l4-w16-seed240828-joint-pairing.jsonl)
- [`results/s3-l8-w16-seed240827-joint-pairing.jsonl`](results/s3-l8-w16-seed240827-joint-pairing.jsonl)
- [`results/s3-l4-w16-seed240828-weight7-joint-pairing.jsonl`](results/s3-l4-w16-seed240828-weight7-joint-pairing.jsonl)
- [`results/s3-l8-w16-seed240827-weight7-joint-pairing.jsonl`](results/s3-l8-w16-seed240827-weight7-joint-pairing.jsonl)
