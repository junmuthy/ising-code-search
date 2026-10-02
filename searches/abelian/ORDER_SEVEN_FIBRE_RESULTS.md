# Certified order-seven-fibre abelian codes

## Result

Both requested codes exist in the paired-polynomial, self-dual two-block
family:

| Physical group | Parameters | Stabilizer weight | Logical organization |
|---|---:|---:|---|
| `C28 x C4` | `[[224,32,7]]` | `12` | two disjoint `C4 x C4` grids |
| `C7 x C4` | `[[56,8,7]]` | `12` | two disjoint `C4` chains |

Both codes have `H_X=H_Z`, so physical transversal Hadamard exchanges each
listed logical `Z` with the logical `X` on exactly the same support.  Their
logical fibre supports are pairwise disjoint and partition all physical
qubits.

## Common construction

Let `q=4` for `C28 x C4` and `q=1` for `C7 x C4`.  Work in

\[
R=\mathbb F_2[C_{7q}\times C_4]
\]

with generators `x` and `y`.  The thickness subgroup is

\[
D=\langle x^q\rangle\cong C_7,
\qquad
\omega_D=\sum_{t=0}^{6}x^{qt}.
\]

Every generator polynomial is a sum of pairs separated inside `D`:

\[
x^u y^v(1+x^{qr}).
\]

Consequently `a omega_D=0` and `a^dagger omega_D=0`.  If `A` is the binary
regular-representation matrix of `a`, the checks are

\[
H_X=H_Z=[A\mid A^T].
\]

On each physical half, the vectors

\[
x^i y^j\omega_D
\]

are weight-seven logical candidates.  They are disjoint, have odd
self-pairing, and are translated regularly by the quotient group.  The rank
tests below establish that they are a complete logical basis, rather than
merely kernel vectors.

## `[[224,32,7]]` over `C28 x C4`

The certified polynomial is

\[
\begin{aligned}
a
&=1+x+x^8y+x^{12}+x^{20}y+x^{25} \\
&=(1+x^{12})+x^{25}(1+x^4)+x^8y(1+x^{12}).
\end{aligned}
\]

Its exact properties are:

| Property | Value |
|---|---:|
| `rank(H_X)=rank(H_Z)` | `96` |
| Encoded qubits | `32` |
| Stabilizer weight | `12` |
| Qubit degree per CSS type | `6` |
| Logical fibre weight | `7` |
| Disjoint logicals per half | `16` |
| Logical translation group per half | `C4 x C4` |
| Certified distance | `7` |

The two halves therefore encode two independent 4-by-4 logical grids.  The 32
logical supports collectively partition all 224 physical qubits.

## `[[56,8,7]]` over `C7 x C4`

The certified polynomial is

\[
\begin{aligned}
a
&=1+y+x+xy+x^2+x^4 \\
&=(1+x)+y(1+x)+x^2(1+x^2).
\end{aligned}
\]

Its exact properties are:

| Property | Value |
|---|---:|
| `rank(H_X)=rank(H_Z)` | `24` |
| Encoded qubits | `8` |
| Stabilizer weight | `12` |
| Qubit degree per CSS type | `6` |
| Logical fibre weight | `7` |
| Disjoint logicals per half | `4` |
| Logical translation group per half | `C4` |
| Certified distance | `7` |

This has exactly the same two-chain logical organization as the
`[[56,8,6]]` bicycle-chain code, but increases the exact distance from six to
seven at the cost of raising stabilizer weight from eight to twelve.

## Search scope and counts

The structural sweep quotiented physical translations and independent axis
reflections and retained only connected codes with exact self-duality,
complete orthonormal fibre logicals, target dimension, even syndrome parity,
and the required logical translations.

| Target | Inequivalent candidates | Structural hits |
|---|---:|---:|
| `[[224,32,7]]`, six terms | `1,323` | `1,293` |
| `[[56,8,7]]`, four and six terms | `450` | `438` |

For `C7 x C4`, all 21 structurally valid four-term candidates were exactly
certified.  Their distance distribution was:

| Distance | Codes |
|---:|---:|
| `2` | `9` |
| `4` | `3` |
| `6` | `9` |
| `7` | `0` |

Thus the four-term paired family cannot improve the known bicycle-chain
distance.  The first distance-seven result appeared after 29 exact
certifications in the broader six-term family.

## Exact-distance certificate

Because the fibre rows form a complete orthonormal logical basis, every
nontrivial zero-syndrome vector anticommutes with at least one fibre row.  The
physical translations make all constrained logical classes within a half
equivalent.  It is therefore sufficient to solve one binary minimum-weight
problem from each physical half:

```text
min |e|  subject to  H e = 0  and  L_i e = 1.
```

HiGHS returned an optimal weight of seven for both representatives of each
code.  The known weight-seven fibres supply the matching upper bound, so the
distance is exactly seven.

## Files

The implementation is in `gala_search/abelian_fibre_codes.py`.  The exhaustive
structural runner is `searches/abelian/run_abelian_fibre_search.py`, and exact certification is
performed by `searches/abelian/certify_abelian_fibre.py`.

Machine-readable results are under
`results/fibre-codes/q4-and-row-d7-v1/`.  Every runner refuses to overwrite an
existing output.
