# Nonabelian [[128,32,6]]

This is the earlier nonabelian group-algebra result, with a regular
`C4 x C8` action on all 32 logical qubits and common-basis Hadamard. Its
physical code splits into two `[[64,16,6]]` components; it is not an
indecomposable 128-qubit block. There are 64 displayed weight-12 checks per
Pauli type, of rank 48. Saved logical weights are unoptimized:
`X: 33–57`, `Z: 43`.

Use the order-64 group

```text
z^2 = 1, z central, [x,y] = z, x^4 = 1, y^8 = z.
```

Elements `(i,j,t)=x^i y^j z^t` have index `2*(8*i+j)+t`, and physical
coordinates `(sheet,g)` have index `64*sheet+g`. Little-endian coefficient
integers `a=4613955421298753553`, `b=584115585032` give

```text
H_X = [L_a | L_b]
H_Z = [L_b^T | L_a^T].
```

Right multiplication by `x,y,z` supplies the saved physical permutations.
Their orders are four, sixteen and two, respectively. The central action
is trivial on logical Paulis, giving the regular logical quotient
`G/<z> = C4 x C8`; the physical generators do not commute.

The Hadamard fold swaps the two sheets and applies
`x -> x^3 z`, `y -> y^7 z`, `z -> z`. It is an involution and induces
individual logical Hadamards with permutation
`(i,j) -> (3-i mod 4, 2-j mod 8)`.

Exact distance-six witnesses are `X: [0,1,18,19,110,111]` and
`Z: [0,1,92,93,110,111]`. The [portable verifier](../README.md#standalone-verification)
reconstructs the checks, certifies both actions and recomputes both
distances. The connected 128-qubit comparison is archived separately as
[`[[128,32,7]]`](../n128_k32_d7/README.md).
