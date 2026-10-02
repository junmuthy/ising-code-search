# One-block `C56 x C4` exact-self-dual search

This package searches one-block CSS codes with target parameters
`[[224,32,d]]` and `d <= 7`.  Physical qubits are indexed by
`C56 x C4`; the order-seven subgroup `<x^8>` supplies thickness, and its 32
cosets are the disjoint weight-seven logical fibres translated by `C8 x C4`.

## Construction

Factoring the thickness algebra gives one trivial and two reciprocal
three-dimensional sectors:

```text
t^7 + 1 = (t + 1)(t^3 + t + 1)(t^3 + t^2 + 1).
```

The nontrivial sectors are copies of `GF(8)[C8 x C4]`.  For an ideal `I` in
this quotient algebra, the binary stabilizer space is the first reciprocal
sector over `I` plus the dagger of the other sector over `Ann(I)`.  This has
rank 96, is self-orthogonal, omits the trivial fibre sector, and therefore
gives exactly 32 logical qubits with `H_X = H_Z`.

## First-pass scope

- all 495 monomial ideals of `GF(8)[U,V]/(U^8,V^4)`;
- all normalized quotient binomials;
- all normalized zero-augmentation trinomials;
- physical orbit generators of weight at most 12;
- exact distance through seven.

Translations and nonzero `GF(8)` scaling reduce the sparse inputs to 17
binomials and 930 trinomials.  Duplicate ideals are analyzed once.

Weights one through five are excluded by exact subset enumeration.  Remaining
codes use an exact vectorized `3+3` meet-in-the-middle test for weight six.
If no weight-six logical exists, the known fibre gives the matching
weight-seven upper bound and certifies `d=7`.

## Running

```bash
../../.venv/bin/python searches/one_block_c56xc4/run_search.py \
  --output results/one-block-c56xc4/NAME \
  --checkpoint-every 10
```

The runner refuses to overwrite an existing output directory and writes every
candidate before advancing.  `progress.json` and `summary.json` provide
checkpoint and final counts.
