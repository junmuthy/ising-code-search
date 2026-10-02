# Weight-eight search results

Search date: 2026-08-24

## Scope

The search exhaustively considered the normalized family

\[
a(x,y)=1+y^r+xy^s+xy^{s+t}
\]

for `m = 7, 11, 13`, quotiented by torus translations and independent
reflections of the `C_4` and `C_m` axes.  There were 261 symmetry
representatives.  Of these, 247 passed every algebraic requirement, including
`rank(H) = 4m - 4`, `k = 8`, and completeness of the eight canonical column
logicals.  The remaining 14 representatives had additional logical sectors.

Distance screening used 200 BP+OSD trials on the combined sweep and 2,000
trials on `m = 11, 13`.  Any candidate left above the best known distance was
then handled by binary minimum-weight ILPs.  Threshold ILPs supplied rigorous
logical witnesses for rejection; full eight-ILP runs supplied exact distances.

## Best distance by lift order

| `m` | Parameters | Best certified distance | Status |
|---:|---:|---:|---|
| 7 | `[[56,8,6]]` | 6 | Known baseline reproduced |
| 11 | `[[88,8,8]]` | 8 | Maximum within the searched family |
| 13 | `[[104,8,10]]` | 10 | Maximum within the searched family |

For each `m`, every other accepted symmetry representative has a verified
logical witness at or below the listed maximum.  Thus these are maxima of this
enumerated family, rather than merely the largest randomized bounds observed.

## Certified `[[104,8,10]]` representatives

| Candidate | Polynomial `a(x,y)` |
|---|---|
| `w8-m13-r4-s2-t6` | `1 + y^4 + x y^2 + x y^8` |
| `w8-m13-r4-s1-t6` | `1 + y^4 + x y^1 + x y^7` |
| `w8-m13-r2-s5-t3` | `1 + y^2 + x y^5 + x y^8` |
| `w8-m13-r2-s1-t3` | `1 + y^2 + x y^1 + x y^4` |
| `w8-m13-r1-s4-t5` | `1 + y + x y^4 + x y^9` |
| `w8-m13-r1-s2-t5` | `1 + y + x y^2 + x y^7` |

The first representative is a convenient primary candidate:

\[
Q_{4,13}\left(1+y^4+xy^2+xy^8\right)=[[104,8,10]].
\]

It has identical X/Z check matrices, check weight 8, qubit degree 4 per CSS
type, even syndrome parity, and the intended two disjoint logical `C_4` chains.

## Reproducibility artifacts

- `results/weight8.jsonl`: combined 200-trial sweep.
- `results/weight8-m11-2000.jsonl`: stronger `m=11` screen.
- `results/weight8-m13-2000.jsonl`: stronger `m=13` screen.
- `results/certifications.jsonl`: ILP witnesses and exact certificates.
- `results/weight8.csv`: merged tabular results.
- `results/summary.md`: generated ranked report.

These results establish static code distance only.  Circuit fault distance,
syndrome schedule depth, transversal-gate phases, and STAR postselection
performance remain separate validation stages.
