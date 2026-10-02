# Minimum `C2` search results

## Exact `n = 12` obstruction

The `[[12,2,6]]` target is impossible under the required disjoint-support and
permutation ZX-duality conditions. The two logical-Z supports must both have
weight six and partition all twelve physical qubits. For every physical
permutation `P`, the logical pairing

\[
M_{ij}=z_i\mathbin{\cdot}P(z_j)
\]

then has zero row sums, because `P` fixes the physical all-ones vector and each
`z_i` has even weight. Thus `(1,1)` lies in the kernel of `M`, contradicting
the required rank-two permutation pairing. A machine-readable certificate is
saved at the root of every run.

## Fold catalogs

The search enumerates every involutive fold commuting with the chosen physical
`C2` translation, filters for permutation logical pairing, and quotients by
relabelings preserving the two logical supports and slack-orbit types.

| Geometry | Canonical folds | Raw valid folds |
| --- | ---: | ---: |
| `n14-w6-s2-0-s1-2` | `0` | `0` |
| `n14-w6-s2-1-s1-0` | `20` | `3744` |
| `n14-w7-s2-0-s1-0` | `40` | `6512` |
| `n16-w6-s2-0-s1-4` | `0` | `0` |
| `n16-w6-s2-1-s1-2` | `40` | `7488` |
| `n16-w6-s2-2-s1-0` | `40` | `14976` |
| `n16-w7-s2-0-s1-2` | `80` | `13024` |
| `n16-w7-s2-1-s1-0` | `80` | `13024` |
| `n16-w8-s2-0-s1-0` | `0` | `0` |

The zero-fold entries are themselves exact geometry-level pairing
obstructions. In particular, an even-weight logical partition cannot support
a nonsingular permutation pairing unless the fold can mix active and slack
`C2` transposition orbits.

## Initial weight-eight pilots

The first runs use X-basis generator weight at most eight. No distance-six hit
has appeared yet, but these are pilots rather than completed negative searches.

| Run | Tasks | Exact `UNSAT` | Unresolved | Models | Best distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n14-w8-pilot20-260830-v1` | `20` | `17` | `3` | `50` | `3` |
| `n14-w7-w8-pilot20-260830-v1` | `20` | `16` | `4` | `74` | `3` |
| `n16-w7-s2-1-w8-pilot20-260830-v1` | `20` | `17` | `3` | `83` | `3` |

The non-free `C2` module sectors usually close exactly after only a few CEGIS
rounds. The unresolved tasks are the free-module-rich sectors after thousands
of exact low-weight logical exclusions. Their cut frontiers are saved as JSONL
and can be replayed into longer runs without modifying these pilot artifacts.

An eager formulation encoding all errors through weight five at once was also
tested on one `n = 14` sector. It was exact but slower for Z3 than the
incremental formulation and timed out after 60 seconds, so production runs
retain the checkpointed CEGIS search.

## Expanded `n = 16`, weight-eight sweep

The next sweep broadened the fold and logical-support geometries while keeping
the maximum displayed check weight at eight. The most useful aggregate results
are:

| Geometry and fold range | Tasks | Exact `UNSAT` | Unresolved | Models | Best distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n16-w7-s2-1-s1-0`, folds `5`–`9` | `20` | `15` | `5` | `73` | `3` |
| `n16-w7-s2-0-s1-2`, folds `0`–`4` | `20` | `17` | `3` | `92` | `3` |
| `n16-w6-s2-1-s1-2`, folds `0`–`4` | `20` | `17` | `3` | `50` | `2` |
| `n16-w6-s2-2-s1-0`, all folds `0`–`39` | `160` | `144` | `16` | `307` | `4` |

The final row covers every canonical fold and every rank-seven `C2` module
type for the two-logical-weight-six, two-slack-transposition geometry. Its 307
models had the exact distance distribution

\[
275\text{ at }d=2,\qquad
26\text{ at }d=3,\qquad
6\text{ at }d=4.
\]

No distance-five or distance-six model appeared. This is not an exact no-go
for the entire geometry: sixteen free-module sectors remain solver-limited.
The four unresolved free sectors which reached distance four—folds `0`, `6`,
`31`, and `33`—were each resumed from roughly 740–780 saved cuts with solver
timeouts increasing through 60 seconds. None produced another model or closed
exactly.

One saved distance-four frontier was independently reconstructed with qLDPC,
which returned

\[
\boxed{[[16,2,4]]}.
\]

It has rank seven per CSS type, distinct X/Z spaces exchanged by the stored
fold, physical `C2` symmetry, two disjoint weight-six logical supports in both
bases, permutation logical ZX pairing, a connected Tanner graph, and
weight-eight displayed checks. It fails the target only on distance. The
independent validation is saved as
`results/inverse-c2-minimum/n16-w6-s2-2-s1-0-d4-independent-validation-260830-v1.json`.

## Local refinement of all distance-four frontiers

The six distance-four models were then used as seeds for local searches which
retain the fold, physical `C2`, disjoint weight-six logical supports, rank-seven
CSS spaces, connected Tanner graph, and weight-eight presentation. Two move
classes were tested:

| Move class | Starts | Iterations | Raw proposals | Exact connected neighbors | Best distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Replace one `C2` orbit | `48` | `96,000` | `35,622,100` | `9,137` | `4` |
| Replace two `C2` orbits, allowing `d=3` bridges | `48` | `48,000` | `43,300,373` | `36,960` | `4` |

The one-orbit search improved the number of weight-four logical representatives
from as many as 32 per sector to 16 per sector, but found no distance-five or
distance-six code. The two-orbit search explicitly crossed the local barrier:
it accepted 840 downhill moves, exactly evaluated 4,256 distance-three and
13,791 distance-four neighbors, and again found no distance-five or
distance-six code. It also found the same 16-per-sector weight-four plateau in
several folds.

These runs materially strengthen the negative evidence at `n = 16`, but they
are not an impossibility proof. The direct solver still has unresolved sectors,
and orbit replacement samples rather than exhausts the global space. The
checkpointed artifacts are saved under:

- `results/inverse-c2-minimum/n16-d4-local-refinement-48x2000-w8-260830-v1/`;
- `results/inverse-c2-minimum/n16-d4-two-orbit-48x1000-w8-260830-v1/`.

## Descending the saved `[[32,4,6]]` code

The most direct inheritance route was checked exactly. Quotient the saved
code by `T^2`, where `T` is its physical `C4` translation. This identifies
the sixteen data pairs

\[
\{i,i+16\},\qquad 0\leq i<16,
\]

and similarly quotients the X- and Z-check orbits. The fold descends, the two
intended logical supports remain disjoint weight-seven supports exchanged by
the quotient `C2`, and the quotient becomes exactly self-dual. However, its
parameters are

\[
\boxed{[[16,4,2]]}.
\]

The minimum logical is the spectator pair `\{7,15\}`. The quotient has rank
six per CSS type, so a `[[16,2,*]]` descendant would require one further
independent X check and its folded Z partner. Exhausting every such extension
that preserves the inherited fold, physical `C2`, and intended logical pair
leaves exactly one distinct gauge fixing. It has

\[
\boxed{[[16,2,3]]},
\]

with a minimum logical supported on `\{0,3,4\}`. Independent qLDPC
reconstruction returned `(16,4,2)` and `(16,2,3)`, respectively. Thus the
known `[[32,4,6]]` code supplies useful quotient structure, but it cannot be
halved into the desired `[[16,2,6]]` code by quotienting and compatible gauge
fixing.

The reproducible derivation is in `derive_from_n32.py`; its matrices and exact
enumeration are saved under
`results/inverse-c2-minimum/n32-c4-to-c2-quotient-260830-v1/`.

## Exact-analogue `n = 16` follow-up

The closest literal analogue of the saved `[[32,4,6]]` logical geometry uses
two disjoint weight-seven supports and one spectator `C2` pair. All 80
canonical folds were scanned in the free rank-seven module sector
`(2,2,2,1)`. The direct CEGIS search examined 406 models: 42 folds closed
exactly and 38 remained solver-limited, with direct best distance three.

Local refinement found six distance-four endpoints, but three complementary
searches did not cross the next boundary:

| Start set and move class | Iterations | Raw proposals | Best distance |
| --- | ---: | ---: | ---: |
| All 33 distance-three seeds, one orbit | `132,000` | `45,775,492` | `4` |
| Promoted distance-four seeds, one orbit | `160,000` | about `60.4` million | `4` |
| Promoted distance-four seeds, two orbits | `60,000` | about `48.9` million | `4` |

Every promoted distance-four row space admits a rank-seven displayed basis of
weight at most six, so the failure was not caused by enforcing the weight-eight
presentation ceiling. The current conclusion at `n = 16` remains strong
negative evidence, not an impossibility proof.

## `n = 18` transition

The `n = 18` search keeps the same target—`k=2`, physical `C2`, two disjoint
logical supports, a commuting involutive permutation ZX fold, connected Tanner
graph, and displayed check weight at most eight—but raises the stabilizer rank
to eight per CSS sector. Three support geometries were tested in the free
module sector `(2,2,2,2)`.

| Logical geometry | Direct folds | Direct models | Direct distance counts | Best distance |
| --- | ---: | ---: | --- | ---: |
| Weight seven plus two spare `C2` pairs | `0`–`9` | `108` | `32` at `d=2`, `68` at `d=3`, `8` at `d=4` | `4` |
| Weight six plus three spare `C2` pairs | `0`–`4` | `23` | `18` at `d=2`, `4` at `d=3`, `1` at `d=4` | `4` |
| Two weight-nine supports filling all qubits | `0`–`4` | `131` | `37` at `d=2`, `84` at `d=3`, `10` at `d=5` | `5` |

The fully packed odd-support geometry is a qualitative improvement. Four fold
tasks saved connected candidates independently reconstructed by the search as

\[
\boxed{[[18,2,5]]}.
\]

Their two weight-nine logical supports partition the 18 physical qubits, the
physical `C2` swaps them, the stored ZX fold induces a nonsingular logical
permutation pairing, and the representative presentation has eight X checks
and eight Z checks, all of weight eight. Independent qLDPC validation confirms
`n=18`, `k=2`, `d_X=d_Z=5`, both rank-eight check spaces, both disjoint logical
bases, and every stored symmetry relation. Thus `n = 18` supports the full
static Ising geometry through distance five.

Targeted refinement of the four distance-five seeds used both one-orbit and
two-orbit moves. Across `112,000` iterations and `46,591,582` replacement
attempts it evaluated `17,785` more distance-five neighbors, but found no
distance-six code. This is again a sampled negative result rather than a
no-go theorem. The key artifacts are:

- `results/inverse-c2-minimum/n18-w9-s2-0-folds00-04-m2222-w8-260831-v1/`;
- `results/inverse-c2-minimum/n18-w9-s2-0-d5-one-orbit-4x5000-w8-260831-v1/`;
- `results/inverse-c2-minimum/n18-w9-s2-0-d5-two-orbit-2x4000-w8-260831-v1/`;
- `results/inverse-c2-minimum/n18-w9-s2-0-d5-qldpc-validation-260831-v1.json`.

## Stopped `n = 20` search

The `n = 20` extension uses rank-nine X and Z check spaces and retains the
same physical `C2`, disjoint-logical, permutation-ZX, connectivity, and
weight-eight constraints. The leading geometry has two disjoint weight-nine
logical supports and one spare physical `C2` pair. In the free-rich module
sector `(2,2,2,2,1)`, complete pilots over folds `0`–`9` examined 223 models:

\[
74\text{ at }d=2,\qquad
122\text{ at }d=3,\qquad
12\text{ at }d=4,\qquad
15\text{ at }d=5.
\]

Local one- and two-orbit refinement around three saved distance-five seeds
then used `69,000` iterations and `29,844,500` replacement attempts. It
evaluated `7,846` additional distance-five neighbors without reaching
distance six.

Independent qLDPC reconstruction confirms the representative candidate as

\[
\boxed{[[20,2,5]]},\qquad d_X=d_Z=5.
\]

It has two disjoint weight-nine logicals in both bases, the physical `C2`
swaps each logical pair, the involutive fold commutes with that translation
and maps the X checks to the Z checks, and the logical ZX pairing is the
identity. Its displayed presentation has nine X and nine Z checks of weights

\[
(8,8,8,8,8,8,8,8,6)
\]

in each sector.

Two alternative first-five-fold pilots were weaker: the weight-seven geometry
examined 140 models and stopped at distance four; the weight-eight geometry
examined 75 models and also stopped at distance four. A stratified follow-up
of the weight-nine fold catalog was stopped at the user's request during fold
10 after nine models, with best distance four. No search process remains
active, and the partial checkpoint was preserved rather than overwritten or
deleted.

Key artifacts are:

- `results/inverse-c2-minimum/n20-w9-s2-1-folds00-04-m22221-w8-260831-v1/`;
- `results/inverse-c2-minimum/n20-w9-s2-1-folds05-09-m22221-w8-260831-v1/`;
- `results/inverse-c2-minimum/n20-w9-s2-1-d5-one-orbit-3x5000-w8-260831-v1/`;
- `results/inverse-c2-minimum/n20-w9-s2-1-d5-two-orbit-2x4000-w8-260831-v1/`;
- `results/inverse-c2-minimum/n20-w9-s2-1-d5-qldpc-validation-260831-v1.json`.

## Scope

The current static phase verifies the code space, disjoint logical geometry,
physical `C2` action, GALA-definition permutation ZX duality, and exact X/Z
distance. Even-parity measured presentations, syndrome schedules, and circuit
fault distance are intentionally deferred until a static distance-six survivor
exists.

All run data is written to new directories below
`results/inverse-c2-minimum/`; no previous search artifacts are modified.
