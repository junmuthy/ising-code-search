# BB64 simultaneous syndrome schedules

This directory constructs and screens bare-ancilla syndrome schedules for the
all-weight-eight simultaneous basis of the Liang--Chen `[[64,8,8]]` BB code.
The presentation has 32 displayed `X` rows and 32 displayed `Z` rows, each of
weight eight and rank 28.

## Exact depth-eight schedule family

Every data qubit meets four `X` checks and four `Z` checks. A collision-free
simultaneous bare-ancilla schedule therefore has the lower bound

\[
d_{\mathrm{CNOT}}\geq 4+4=8.
\]

The saved schedules attain this bound. Each of their eight layers is a perfect
matching containing 64 CNOTs and using all 64 data qubits and all 64 dedicated
syndrome ancillas. Consequently the entangling layers have no idle qubits.

`enumerate_schedules.py` uses the full 32-element twisted-torus translation
action. The 256 Tanner edges of one check type form eight free edge orbits. It
enumerates all `8! = 40,320` orbit colorings, identifies time reversal, and
requires:

- collision freedom for simultaneous `X` and `Z` extraction;
- the `Z` order to be the reverse of the `X` order under the identity ZX fold;
- even cross-ancilla backaction parity for every `X`/`Z` check pair;
- exactly one use of every Tanner edge.

The exhaustive identity-fold run found:

| Stage | Count |
| --- | ---: |
| Raw orbit colorings | 40,320 |
| Time-reversal representatives | 20,160 |
| Collision-free colorings | 4,608 |
| Clean depth-eight schedules | 32 |

The stable baseline is `baseline_schedule.json`, schedule ID
`67e327dc7cb35335`.

## ASCII atom animation

`animate_schedule.py` turns the canonical schedule into eight labeled ASCII
interaction frames, bracketed by preparation and measurement frames. It fixes
the 32 `X` and 32 `Z` syndrome atoms in two displayed planes, routes the two
32-atom data grids with the exact rigid group translation derived from each
layer, and marks all 64 simultaneous CNOT bonds. Run it interactively with

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.animate_schedule \
  --animate --fps 1 --cycles 2
```

To inspect every frame in Emacs, open `schedule_animation.txt`. It can be
regenerated without animation using

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.animate_schedule \
  --output \
  bb64_simultaneous_basis_search/schedule_fault_search/schedule_animation.txt
```

This is a combinatorial rigid-translation view of the CNOT matchings. It does
not claim a continuous collision-free tweezer trajectory; such a trajectory
would require hardware geometry, separation, speed, and blockade constraints
that are not contained in the stabilizer schedule.

### Spatial supercell view

`animate_spatial_schedule.py` provides the more geometric neutral-atom view.
It places the two fixed syndrome atoms at the center of every site and the two
mobile data atoms on their contact sides:

```text
L00--X00 Z00--R01
```

Here both `--` links are active CNOT bonds. In a position frame the same atoms
are separated by spaces, indicating that transport occurs with the bonds off.
Data identities scroll around the 4-by-8 array and the `L` and `R` grids change
sides as the rigid translations change. Run it with

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.animate_spatial_schedule \
  --animate --fps 2 --cycles 2
```

The complete sequence is also saved as `spatial_schedule_animation.txt`. This
view proves that each contact configuration consists of two rigid 32-atom group
translations. It still does not specify the continuous paths used to exchange
sides or implement the twisted periodic wrap.

### Strict planar optical-tweezer view

`animate_planar_schedule.py` removes the twisted chart from the drawing. It
uses the direct-product coordinates

\[
(u,v)=(x+y\bmod 8,x\bmod 4)
\in C_8\times C_4
\]

and draws an ordinary 8-by-4 rectangle. Each fixed `XZuv` token represents the
two syndrome atoms at one planar site; `Luv` and `Ruv` are mobile data atoms.
For example,

```text
 L00--XZ00-- R10
```

is one four-atom contact cell with two active CNOT bonds. Boundary data atoms
that cannot follow the bulk Euclidean displacement are marked with `*` and
assigned to exterior bypass lanes. Side exchanges send `L` through an upper
corridor and `R` through a lower corridor. Run this view with

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.animate_planar_schedule \
  --animate --fps 2 --cycles 2
```

The complete static sequence is `planar_schedule_animation.txt`. The planar
view specifies every contact endpoint, bulk displacement, boundary-bypass set,
and side-exchange corridor. It is a concrete routing topology, but it is not a
calibrated or formally collision-certified continuous trajectory; that final
step requires the physical tweezer pitch, clearance, speed, and blockade
radius.

## Reproduction

```bash
.venv/bin/python -B -m \
  bb64_simultaneous_basis_search.schedule_fault_search.enumerate_schedules \
  --output-dir \
  bb64_simultaneous_basis_search/schedule_fault_search/results/identity-fold-run-001
```

Generated search logs and complete schedules live under the ignored `results/`
tree. The stable baseline is kept outside that tree so downstream validation is
reproducible from tracked inputs.
