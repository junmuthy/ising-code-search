# Floating-`k` self-dual `C4 x C4` result

## Scope

The search exhausts all 14,892 nonzero polynomials of weight at most six in
`GF(2)[C4 x C4]`.  Each polynomial defines

```text
H_X = H_Z = [A | A^T]
```

on 32 qubits, with check weight at most 12.  The dimension is allowed to
float: every code with `k >= 4` is retained for the logical-geometry test.

The physical translation by `(1,0)` divides the 32 qubits into eight `C4`
fibres.  A distance-compatible odd logical seed selects one qubit from seven
of those fibres.  Its four translates are then disjoint and, because
`H_X=H_Z`, have identity ZX pairing.

For each code, the seven-of-eight-fibre zero-syndrome problem is solved
exactly by a three-plus-four meet-in-the-middle search.

## Exact counts

```text
k       polynomial count
0                  4,944
8                  7,488
12                   320
14                   960
16                   912
18                    64
20                   176
24                    28
```

There are 9,948 codes with `k >= 4`; 9,248 have connected Tanner graphs and
700 are disconnected.

None of the 9,948 codes contains a disjoint weight-seven logical `C4` orbit.
The failure occurs at the exact zero-syndrome condition, before stabilizer
nontriviality or distance needs to be tested.

Therefore the floating-dimension relaxation does not rescue the self-dual
`C4 x C4` BB family under check weight 12.  This is an exact negative for the
enumerated polynomial family, not a sampled result.

Artifacts:

- `results/c4xc4-floating/exhaustive-w12-structural-v1/`
- `results/c4xc4-floating/exhaustive-w12-all-components-v1/`
- `results/c4xc4-floating/exhaustive-w12-all-k-structural-v1/`
