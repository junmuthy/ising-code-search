# Initial batched `C4 x C2` search results

## Status

The search pipeline is implemented through the compact `L=2`, baseline
`L=4`, and expanded `L=6` branches.  Distance-six codes now occur in the
compact natural-`S3` branch, and regular ZX-paired logical grids also occur,
but no candidate found so far has both properties.  Every distance rejection
contains an explicit nontrivial logical of weight at most five; there were no
distance-solver timeouts in the completed pilots.

## Deterministic quotient control

Reducing the saved `S3 x C8 x C4` `[[384,200,7]]` polynomial modulo `x^4=1`
and `y^2=1` gives a connected exact-ZX code with

\[
[[96,52,*]],
\]

uniform binary check weight 16, and column degree four.  It has full-rank
regular `C4 x C2` logical modules, demonstrating that the previous rank-zero
mechanism is absent.  It nevertheless has a nontrivial logical of weight two,
so it fails the distance gate.

The authoritative corrected run is
`results/batched-c4xc2-search/quotient-seed-260831-v2`.  The earlier `v1`
directory is retained rather than overwritten; only its summary metadata
misreported the irrelevant default profile as nominal weight eight.

## Complete abelian `L=4,J=1` scan through weight 12

For each profile, one monomial of `F_0` is normalized to the identity.  The
listed finite families are otherwise exhaustive; no symmetry quotient is
needed for the negative conclusion.

| F-entry weights | Check weight | Candidates | Structural survivors | `d < 6` | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(2,2)` | 8 | 196 | 96 | 96 | 0 |
| `(3,3)` | 12 | 1,176 | 1,104 | 1,104 | 0 |
| `(2,4)` | 12 | 490 | 472 | 472 | 0 |
| `(4,2)` | 12 | 980 | 944 | 944 | 0 |
| Total |  | 2,842 | 2,616 | 2,616 | 0 |

This rules out the strict identity-fold, two-entry abelian family through
check weight 12.

## Nonabelian pilots

These are reproducible random pilots, not exhaustive searches.

| Top lift | `n` | F-entry weights | Nominal weight | Raw candidates | Structural survivors | `d < 6` |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| natural `S3` | 96 | `(2,2)` | 8 | 250 | 56 | 56 |
| faithful `GL(2,2)` | 64 | `(2,2)` | 8 | 250 | 29 | 29 |
| natural `S3` | 96 | `(3,3)` | 12 | 1,000 | 278 | 278 |
| faithful `GL(2,2)` | 64 | `(3,3)` | 12 | 1,000 | 116 | 116 |
| natural `S3` | 96 | `(2,4)` | 12 | 1,000 | 235 | 235 |
| faithful `GL(2,2)` | 64 | `(2,4)` | 12 | 1,000 | 94 | 94 |
| natural `S3`, fold 0 | 96 | `(4,4)` | 16 | 1,000 | 169 | 169 |
| natural `S3`, fold 1 | 96 | `(4,4)` | 16 | 1,000 | 151 | 151 |
| faithful `GL(2,2)`, fold 0 | 64 | `(4,4)` | 16 | 1,000 | 22 | 22 |
| faithful `GL(2,2)`, fold 1 | 64 | `(4,4)` | 16 | 1,000 | 24 | 24 |

The faithful lift can turn one group monomial into a binary row contribution
larger than one.  In its weight-12 pilot, 209 raw candidates violated the
actual binary check-weight ceiling of 16, sometimes alongside another
rejection.  This is why the implementation records both nominal polynomial
weight and measured binary row weight.

The two `fold 1` runs use the nonidentity strict GALA fold: transversal
Hadamard plus a fixed reflection of the protograph blocks exchanges the exact
binary X- and Z-check spaces.  Hence their negative result does not depend on
requiring `H_X = H_Z`.  The identity and nonidentity folds have comparable
structural yields, and every survivor in both branches has an explicit
logical operator of weight at most five.

For the faithful lift at nominal weight 16, 706 of 1,000 candidates in each
fold violate the measured binary weight-16 ceiling (possibly together with
another structural rejection).  This leaves only 22 and 24 structural
survivors.  The natural lift stays within the nominal ceiling and supplies
many more valid CSS presentations, but the distance bottleneck remains.

## `L=6,J=1` pilots and the binomial obstruction

| Top lift | `n` | F-entry weights | Nominal weight | Structural survivors | Witness weight |
| --- | ---: | --- | ---: | ---: | ---: |
| abelian | 48 | `(2,2,2)` | 12 | 763 | 2 for all 763 |
| faithful `GL(2,2)` | 96 | `(2,2,2)` | 12 | 81 | 2 for all 81 |
| natural `S3` | 144 | `(2,2,2)` | 12 | 199 | 2 for all 199 |
| abelian | 48 | `(2,2,4)` | 16 | 981 | 2 for all 981 |
| faithful `GL(2,2)` | 96 | `(2,2,4)` | 16 | 17 | 2 for all 17 |
| natural `S3` | 144 | `(2,2,4)` | 16 | 172 | 2 for all 172 |

All 2,213 structural survivors have duplicate check-matrix columns that give
a nontrivial weight-two logical.  For a permutation lift, a two-term entry is
a sum of two permutation matrices, and its transpose has the same column set.
The strict fold places those columns in paired physical blocks.  Avoiding a
binomial in all three nonempty entries requires check weight at least 18, so
the `L=6,J=1` permutation-lift branch is structurally incompatible with the
weight-16 ceiling.  The faithful lift is not a pure permutation lift, but its
measured binary weights are larger and its two pilots show the same failure.

## Compact `L=2,J=1` pilots

The compact branch has `n=32` for the faithful lift and `n=48` for the natural
lift.  The faithful samples remained negative.  The natural lift produced:

| Polynomial weight | Check weight | Structural survivors | `d < 6` | `d >= 6` |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 89 | 89 | 0 |
| 6 | 12 | 122 | 115 | 7 |
| 8 | 16 | 101 | 90 | 11 |

All 18 distance-qualified codes have `n=48`; 17 have `k=8` and one has
`k=12`.  Exhaustive enumeration of every nonzero logical class for `k <= 12`
proves that none contains a cyclic class whose `C4 x C2` orbit has rank eight.
Thus the negative grid result is not a random-seed miss.

Screening all 312 structural survivors across the three natural-`S3` profiles
found exactly two codes with a full-rank, ZX-paired regular grid.  Both occur
at polynomial weight eight and have distance three.  The stronger saved seed
is

\[
[[48,16,3]],
\]

with measured X/Z check weights 12 and 16 and raw four-batch disjoint logical
representatives of weight nine.  In the ordered `S3` basis

\[
(h_0,\ldots,h_5)=(e,(12),(01),(012),(021),(02)),
\]

its normalized polynomial is

\[
a=(h_0+h_3)(1+x^2)+x^3(h_1+h_2)+x^3y(h_0+h_5).
\]

The authoritative record, including 32 regular logical modules and explicit
translated supports, is in
`results/batched-c4xc2-search/s3-natural-l2-all-structural-logical-action-260831-v1`.

## Exact local search around the grid seed

The exact one-term neighborhood contains 280 candidates.  It has 98
structural survivors, 96 distance failures, and two `d >= 6` codes; neither
distance survivor has a full-rank grid orbit.

The exact two-term neighborhood contains 16,380 candidates:

| Outcome | Count |
| --- | ---: |
| Structural rejection | 13,407 |
| Explicit logical below weight six | 2,802 |
| `d >= 6`, but no full-rank grid orbit | 171 |
| Regular ZX grid and `d >= 6` | 0 |

The empty intersection is therefore exact for this normalized radius-two
neighborhood, not a random-search statement.

## Next ansatz

The evidence says to stop enlarging the same strict identity-fold polynomial.
The next compact search should retain `L=2,n=48` but broaden the GALA ZX fold:
search independent `F` and `G` polynomials and use a half-swapping physical
fold, rather than imposing `G=F^T`.  A tractable first subfamily makes `F` and
`G` individually transpose-equivalent (for example, inverse-closed supports),
so transversal Hadamard followed by the half swap exchanges X and Z.  Active
orthogonality then filters `FG^T+GF^T=0`.  This doubles the useful generator
freedom while retaining strict GALA ZX duality, `n=48`, check weight at most
16, and the logical-action-first acceptance gate learned here.

## Independent `F/G` half-swap pilots

The proposed follow-up has now been implemented and run.  It fixes the
natural `S3` lift and `L=2,J=1,n=48`, while searching two independent
transpose-invariant supports.  Writing their binary lifts as `A` and `B`
gives

\[
H_X=[A\mid B],\qquad H_Z=[B\mid A].
\]

Thus transversal Hadamard followed by the nonidentity permutation that swaps
the two 24-qubit halves exchanges the exact X and Z presentations.  The GALA
ZX requirement is not weakened.  CSS orthogonality is instead the active
nonabelian constraint

\[
AB+BA=0.
\]

Nine reproducible 1,000-candidate pilots cover the balanced profiles and all
even asymmetric profiles selected at total weights 12 and 16:

| `(wt(F),wt(G))` | Nominal check weight | Structural survivors | `d < 6` | `d >= 6` | Regular grid and `d >= 6` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(4,4)` | 8 | 40 | 40 | 0 | 0 |
| `(6,6)` | 12 | 45 | 43 | 2 | 0 |
| `(4,8)` | 12 | 57 | 52 | 5 | 0 |
| `(8,4)` | 12 | 66 | 63 | 3 | 0 |
| `(8,8)` | 16 | 38 | 34 | 4 | 0 |
| `(6,10)` | 16 | 40 | 37 | 3 | 0 |
| `(10,6)` | 16 | 71 | 62 | 9 | 0 |
| `(4,12)` | 16 | 50 | 47 | 3 | 0 |
| `(12,4)` | 16 | 74 | 70 | 4 | 0 |
| **Total** |  | **481** | **448** | **33** | **0** |

All 33 distance survivors have `k <= 12`: 28 are `k=8`, four are `k=10`,
and one is `k=12`.  Every nonzero logical class was therefore enumerated
exactly.  Their maximum cyclic `C4 x C2` orbit ranks are

| Maximum orbit rank | Codes |
| ---: | ---: |
| 2 | 3 |
| 4 | 25 |
| 5 | 4 |
| 6 | 1 |
| 8 | 0 |

The induced logical-action diagnostic explains the empty intersection.  For
each code, the eight physical translations span an operator algebra of the
same dimension, only 2 through 6.  Acting on any single logical class can
therefore generate at most that many independent translates.  In 13 codes
the logical action explicitly collapses the second translation according to

\[
T_y=I\quad\text{or}\quad T_y=T_x^2.
\]

In the other 20, `T_y` is distinct from every power of `T_x`, but the eight
group actions are still linearly dependent on the logical quotient.  This is
a stronger diagnosis than merely failing to sample a useful representative:
no eight-site cyclic logical grid exists in any of the 33 codes.

The authoritative logical-action records are in
`results/batched-c4xc2-search/s3-natural-l2-halfswap-d6-action-analysis-260831-v1`.
No batching MILP was needed because the regular-grid prerequisite failed
first.

## Updated next gate

Future candidates should compute the dimension of the translation image
algebra before distance certification.  A dimension below eight is an exact,
cheap rejection for the desired cyclic grid.  The next generator ansatz also
has to loosen individual transpose invariance—while retaining a strict GALA
ZX permutation—because adding independent inverse-closed `F/G` freedom found
distance-six codes but repeatedly collapsed their logical translation
module.

## X-reflected half-swap pilots

The updated gate and broader fold have now been implemented.  Let `alpha` be
the bottom-group reflection

\[
\alpha:x\mapsto x^{-1},\qquad y\mapsto y.
\]

The generator constraint is

\[
F^\dagger=\alpha(F),\qquad G^\dagger=\alpha(G),
\]

rather than `F=F^dagger` and `G=G^dagger`.  Thus a monomial at `x` no longer
forces a partner at `x^{-1}`.  At the binary level, transversal Hadamard,
x reflection within each 24-qubit half, and the half swap exchange the exact
X/Z row spaces.  Applying the same x reflection to the check labels gives
literal matrix equality, so this remains strict GALA ZX duality rather than a
logical-only test.

The runner computes the algebra spanned by the eight induced logical
translations immediately after the structural gates.  Dimension below eight
is an exact rejection; only dimension-eight candidates proceed to distance.

| `(wt(F),wt(G))` | Nominal check weight | Structural survivors | Translation algebra `<8` | Translation algebra `=8` | `d < 6` | `d >= 6` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(4,4)` | 8 | 149 | 138 | 11 | 11 | 0 |
| `(6,6)` | 12 | 164 | 155 | 9 | 9 | 0 |
| `(4,8)` | 12 | 123 | 113 | 10 | 10 | 0 |
| `(8,4)` | 12 | 188 | 176 | 12 | 12 | 0 |
| `(8,8)` | 16 | 141 | 132 | 9 | 9 | 0 |
| `(6,10)` | 16 | 113 | 103 | 10 | 10 | 0 |
| `(10,6)` | 16 | 151 | 138 | 13 | 13 | 0 |
| **Total** |  | **1,029** | **955** | **74** | **74** | **0** |

The 74 dimension-eight candidates prove that the broader fold removes the
previous translation-action obstruction.  Their logical dimensions are 16
for 67 codes, 18 for four, 20 for one, and 24 for two.  However, the exact
distance screens find 24 weight-two logicals and 50 weight-three logicals.
None reaches weight four, much less the target six.

This cleanly separates the two failure modes:

1. the inverse-closed half-swap family produced distance-six codes whose
   logical translation algebra had dimension below eight;
2. the x-reflected family produces the full translation algebra, but its
   corresponding high-rate codes have distance only two or three.

All seven checkpointed runs are stored under
`results/batched-c4xc2-search/s3-natural-l2-xreflect-halfswap-*-pilot-1000-260831-v1`.
No logical-module or batching optimization was run after distance, because no
candidate passed the distance gate.

The next productive search should not simply enlarge these random pilots.  It
should use the dimension-eight algebra condition during generator synthesis
and optimize against the observed weight-two/three logical witnesses, so that
distance and logical translation structure are coupled rather than selected
independently.

## Witness-guided beam search

The proposed coupled search has now been run.  It starts from all 74
dimension-eight candidates above and explores complete twisted-transpose
orbit swaps in either `F` or `G`.  Each move preserves the polynomial weight
and the exact x-reflected half-swap ZX fold.  Candidates are filtered by CSS
structure and translation-algebra dimension before their complete logical
spectrum through weight four is counted.  The lexicographic beam score first
maximizes the smallest observed logical weight, then minimizes the numbers of
weight-two, weight-three, and weight-four logicals.

With beam width 20 and three generations, the search evaluated 6,269 unique
candidates:

| Outcome | Count |
| --- | ---: |
| Structural rejection | 2,760 |
| Translation algebra dimension below eight | 2,017 |
| Full algebra, but a logical through weight four | 1,492 |
| Clear through weight four | 0 |

The initial beam already contained candidates with no weight-two logicals.
Generation one removed the remaining weight-four population from the best
score, but generations two and three could not reduce the 16 weight-three
logicals.  The best presentation is therefore

\[
[[48,16,3]],
\]

with nominal check weight 16 and measured X/Z row weights 14 or 16.  In the
ordered natural-`S3` basis

\[
(h_0,\ldots,h_5)=(e,(12),(01),(012),(021),(02)),
\]

one representative has

\[
\begin{aligned}
F &= h_0+h_1+x(h_1+h_2)+x^3y(h_3+h_4),\\
G &= (h_3+h_4)\left[x(1+y)+x^2(1+y)+x^3\right].
\end{aligned}
\]

Its exact Z-logical counts at weights one through four are `(0,0,16,0)`;
the strict ZX fold gives the corresponding X obstruction.  This is a local
negative result, not a global no-go theorem: it shows that single-orbit swaps
from all 74 random seeds reach a common distance-three plateau.  A subsequent
search should use coordinated multi-orbit moves or change the generator
family, rather than merely extending this one-orbit beam to more generations.

The complete append-only record, per-generation beams, and the reconstructible
best candidate are stored in
`results/batched-c4xc2-search/s3-natural-l2-xreflect-witness-beam20-g3-260831-v1`.

## Coordinated two-orbit beam and the natural-`S3` fiber obstruction

The next run started from all 20 members of the final one-orbit beam and
sampled 1,000 exact two-orbit replacements per parent in each of two
generations.  A final candidate is evaluated directly, so neither of its two
one-orbit intermediates has to survive the beam.  The first generation had
19,694 deduplicated proposals and the second had 19,544.  Across both
generations and their seeds, 34,957 unique presentations were evaluated:

| Outcome | Count |
| --- | ---: |
| Structural rejection | 17,646 |
| Translation algebra dimension below eight | 14,847 |
| Full algebra, but a logical through weight four | 2,464 |
| Clear through weight four | 0 |

Every algebra-qualified survivor is a `[[48,16,*]]` code.  Among the 2,288
without a weight-two logical, the smallest number of weight-three logicals is
16; 2,247 codes attain that minimum.  Both generations retain the same best
score `(d=3, N_2=0, N_3=16, N_4=0)` as the starting beam.

The shared witnesses expose a representation-theoretic obstruction.  Write
the natural three-dimensional permutation basis of `S3` as
`e_0,e_1,e_2`.  For physical half `alpha` and bottom-group coordinate
`g in C4 x C2`, define the weight-three fiber

\[
u_{\alpha,g}
=
e_{0,\alpha,g}+e_{1,\alpha,g}+e_{2,\alpha,g}.
\]

The vector `e_0+e_1+e_2` spans the trivial invariant subrepresentation of the
natural `S3` permutation module.  In the binary indexing used by the saved
matrices, `u_{L,e}` has support `(0,8,16)`.  Its 16 translates—eight bottom
coordinates on each of the two physical halves—are mutually disjoint.

An exact diagnostic over all 2,464 algebra-qualified codes found:

| Fiber property | Codes satisfying it |
| --- | ---: |
| All 16 fibers are nontrivial Z logicals | 2,464 |
| All 16 fibers are nontrivial X logicals | 2,464 |
| Z-fiber logical-class rank is 16 | 2,464 |
| X-fiber logical-class rank is 16 | 2,464 |

Thus this ansatz produces precisely the clean logical geometry sought for the
Ising application—two independent, disjoint eight-site grids—but the physical
thickness of each logical fiber is only three.  No generator mutation can
raise the code distance while those fiber classes remain present.

This is stronger than another local-search failure.  The observed distance
ceiling is tied to the trivial subrepresentation inside the natural `S3`
permutation lift.  Further radius-two or radius-four mutation searches in the
same component are not justified.  The next compact ansatz should remove that
invariant three-point fiber, most naturally by revisiting the faithful
two-dimensional `GL(2,2)` representation with the broadened x-reflected fold
and the translation-algebra-first gate learned here.

The authoritative run is
`results/batched-c4xc2-search/s3-natural-l2-xreflect-witness-double-beam20-g2-n1000-260901-v1`.
Its `evaluations.jsonl` is append-only, and `fiber-obstruction.json` records
the exact all-fiber diagnostic.

## Faithful `GL(2,2)` x-reflected `L=2` search

The natural-`S3` obstruction comes from its fixed three-dimensional
permutation fiber.  Replacing that lift by the faithful two-dimensional
representation of

\[
GL(2,2)\cong S_3
\]

removes every nonzero globally fixed top vector and reduces the block length
to

\[
n=2\cdot2\cdot |C_4\times C_2|=32.
\]

The represented transpose is not group inversion in this basis.  The new
model therefore constructs the actual binary transpose and uses transversal
Hadamard followed by x reflection inside both 16-qubit halves and their
exchange.  Applying x reflection to the check rows gives exact matrix
equality with the opposite CSS sector.  This is the broad permutation form
of GALA ZX duality, not the stronger restriction `H_X=H_Z`.

Nine independent-`F/G` pilots covered balanced and asymmetric profiles from
nominal check weight eight through sixteen.  Among 9,000 candidates, 43
passed CSS orthogonality, connectivity, the actual weight ceiling, and
`k >= 8`.  All 43 failed the necessary regular-grid gate: the dimensions of
the algebra spanned by the eight induced translations were

| Translation-algebra dimension | Survivors |
| ---: | ---: |
| 2 | 7 |
| 3 | 10 |
| 4 | 20 |
| 5 | 6 |
| 8 | 0 |

Thus none of these codes can contain a cyclic eight-class logical orbit,
irrespective of how its physical logical representatives are dressed.

### Exact-commuting centralizer ansatz

In the faithful lift, a group-algebra element is a `2 x 2` matrix over the
commutative bottom ring

\[
R=\mathbb F_2[C_4\times C_2].
\]

The targeted follow-up samples

\[
G=p(x,y)I+q(x,y)F.
\]

Because `p` and `q` are central, `[F,G]=0` identically, so active CSS
orthogonality is constructed rather than selected randomly.  The reflected
transpose fixes every bottom monomial, preserving the same exact ZX fold.
Ten 1,000-candidate profiles scanned `wt(F)=4,6`, even `wt(p)=2,4,6,8`, and
`wt(q)=1,2`, always rejecting actual lifted check weight above sixteen.

This produced 939 structural survivors—more than a twenty-fold yield
improvement over independent sampling—but still no regular grid:

| Translation-algebra dimension | Survivors |
| ---: | ---: |
| 2 | 165 |
| 3 | 140 |
| 4 | 518 |
| 5 | 37 |
| 6 | 29 |
| 7 | 50 |
| 8 | 0 |

All 50 dimension-seven cases occur for the norm polynomial

\[
p=\omega_{C_4\times C_2}=\sum_{a=0}^{3}\sum_{b=0}^{1}x^a y^b.
\]

An exact calculation on every case finds the same missing relation,

\[
\sum_{a=0}^{3}\sum_{b=0}^{1}T_x^aT_y^b=0.
\]

Their apparent proximity to dimension eight is therefore forced: they are
modules over the seven-dimensional norm quotient and cannot contain the
regular logical grid.

Finally, three boundary profiles set `p=0`, hence `G=qF`.  Among 3,000
candidates, 598 passed the structural gates and 44 recovered translation
algebra dimension eight.  Their dimensions were `k=16` in 41 cases and
`k=20` in three cases.  Every one of the 44 has an explicit weight-two
logical joining the proportional physical halves, so all have `d=2`.

The faithful `L=2,n=32` centralizer branch therefore exhibits a sharp
tradeoff:

\[
\begin{array}{c|c|c}
p\ne0 & \dim\mathcal A_{\rm trans}<8 & \text{no regular grid}\\
p=0 & \dim\mathcal A_{\rm trans}=8\ \text{is attainable} & d=2
\end{array}
\]

This is a targeted negative result for the scanned x-reflected centralizer
ansatz, not a no-go theorem for all faithful nonabelian GALA codes.  It does
show that simply enlarging these random pilots is poorly motivated.  A next
family should change the protograph or ZX pairing so a full regular bottom
module is present without making the two data halves polynomially
proportional.

The authoritative append-only records are under
`results/batched-c4xc2-search/s3-linear-l2-xreflect-halfswap-*-260901-v1`,
`results/batched-c4xc2-search/s3-linear-l2-commutant-*-pilot-1000-260901-v1`,
and
`results/batched-c4xc2-search/s3-linear-l2-commutant-*-boundary-1000-260901-v1`.

## Faithful protograph probes at `n=64` and `n=96`

The next probes use `r=L/2` faithful blocks per physical half and construct

\[
G_i=p_i(x,y)I+q(x,y)F_{i+a}.
\]

Here `q` is one central bottom-group monomial and `a` is a nonzero cyclic
protograph shift.  The shifted nonabelian products and the central `p_i`
products cancel pairwise in the CSS commutator.  Every `F_i` and `G_i` is
invariant under the faithful reflected transpose, so transversal Hadamard,
x reflection within every 16-qubit block, reversal of the protograph cycle,
and exchange of the two halves give an exact GALA ZX fold.

Four 1,000-candidate profiles were run at each length under the actual lifted
check-weight ceiling 16:

| `n` | Candidates | Structural rejection | Translation algebra `<8` | Full algebra but `d<6` | `d>=6` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4,000 | 714 | 11 | 3,275 | 0 |
| 96 | 4,000 | 948 | 0 | 3,052 | 0 |

The larger protographs remove the `n=32` grid-module obstruction almost
completely.  At `n=64`, 3,275 candidates had translation-algebra dimension
eight; at `n=96`, all 3,052 structural survivors did.  The remaining failure
is distance:

| `n` | Weight-2 logical | Weight-3 logical | Weight-4 logical |
| ---: | ---: | ---: | ---: |
| 64 | 1,798 | 843 | 634 |
| 96 | 3,017 | 35 | 0 |

The logical dimensions explain the severity: accepted `n=64` structures had
`k=32,34,36,40`, while every accepted `n=96` structure had `k=64`.  This
`J=1` centralizer protograph packs too many unconstrained logical sectors.
Moving the same ansatz to `n=192` would therefore be unlikely to fix distance;
the next nonabelian family would need additional active rows or a different
orthogonality construction.

The eight checkpointed pilots are stored under
`results/batched-c4xc2-search/s3-linear-twisted-proto-n*-pilot-1000-260901-v1`.

## Published self-dual `[[64,8,8]]` BB solution

A web and literature search identified the self-dual bivariate-bicycle code
of Liang and Chen, [arXiv:2510.05211v2](https://arxiv.org/abs/2510.05211).
It is defined by

\[
f=1+x+y+y^{-1},\qquad g=f^\dagger
\]

on the twisted torus

\[
x^4y^4=1,\qquad y^8=1.
\]

Full reference: Zijian Liang and Yu-An Chen, “Self-dual bivariate bicycle
codes with transversal Clifford gates,” arXiv:2510.05211v2 (2026), especially
Table 1, Figure 1, and Appendix A.  Table 1 reports the `[[64,8,8]]`
parameters and polynomial; Appendix A derives the explicit logical
representatives used in the local acceptance test.

The paper reports `[[64,8,8]]` and gives explicit weight-eight logical
patterns.  The local reconstruction independently verifies:

| Condition | Result |
| --- | --- |
| Parameters | `[[64,8,8]]` |
| Stabilizer weights | uniform weight 8 |
| ZX duality | `H_X=H_Z`; exact transversal Hadamard |
| Inter-block CNOT | transversal CSS CNOT |
| Phase gate | transversal `S` from doubly-even checks |
| Logical translation group | regular `C4 x C2` on all eight logicals |
| Logical support weight | eight for every grid representative |
| STAR batching | four raw disjoint batches of two logicals |

The exact distance was also cross-checked locally: the MILP proves there is
no logical through weight seven, while the published `alpha` representative
has weight eight.

The logical order-four generator is physical y translation.  It has physical
order eight but logical order four.  A commuting physical involution is

\[
(x^a y^b,\mathrm{half})
\longmapsto
(x^{1-a}y^b,\mathrm{opposite\ half}),
\]

namely x reflection, one x translation, and exchange of the two BB halves.
The orbit of the paper's weight-eight `alpha` logical under these two
permutations has logical-class rank eight and ZX-pairing rank eight.

With orbit ordering

\[
(1,T_2,T_4,T_4T_2,T_4^2,T_4^2T_2,T_4^3,T_4^3T_2),
\]

the four pairs

\[
\{0,1\},\quad\{2,3\},\quad\{4,5\},\quad\{6,7\}
\]

have disjoint physical supports within each pair.  Thus two STAR rotations
can be injected in parallel and all eight sites are covered in four batches,
with no stabilizer dressing optimization.

The reconstructed checks, logical orbit, and the two physical permutations
are saved in
`results/batched-c4xc2-search/published-selfdual-bb64-grid-presentations-260901-v2/raw-disjoint-grid.npz`.
The exact affine search record is in the adjacent `summary.json`.  This code
meets the relaxed 4x4 half-grid target already, so an `n=192` search was not
started automatically.

There is one logical-basis caveat.  Formal GALA ZX duality is exact and the
saved disjoint grid has ZX-pairing rank eight, but its pairing matrix has
three ones in every row rather than being a permutation matrix.  Therefore
transversal Hadamard is a traceable invertible multi-logical Clifford on this
particular grid basis, not eight independent Hadamards up to relabeling.  The
paper separately supplies a basis in which Hadamard acts by logical pairs,
but that basis has not yet been shown to retain the regular grid and raw
disjoint batches simultaneously.  If the Ising compiler requires sitewise
Hadamard rather than GALA ZX duality in its general sense, this simultaneous-
basis condition remains the final acceptance test.
