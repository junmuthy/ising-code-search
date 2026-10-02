# Minimum `C2` folded-CSS search

This directory searches for a `[[n,2,6]]` CSS code with `n <= 16` and the
same static Ising geometry as the saved `[[32,4,6]]` code:

- two pairwise-disjoint logical supports;
- a physical involution exchanging the logical qubits;
- a permutation ZX duality in the sense of Definition 7 of the GALA paper;
- exact static X and Z distance at least six.

The quantum Singleton bound gives `n >= 12`. Equal X/Z ranks under ZX duality
force `n` even, so only `n = 12, 14, 16` need be considered.

The `n = 12` case has an exact pairing obstruction: the two even-weight-six
logical supports fill all physical qubits, forcing the all-ones logical vector
into the kernel of every permutation-induced ZX pairing. The computational
search therefore begins at `n = 14`.

The static phase deliberately does not assume a BB or GALA polynomial ansatz.
It synthesizes any translation-invariant X-check row space, defines the
Z-check space through an involutive physical fold, and uses exact CEGIS cuts
to exclude every logical operator below weight six. Low-weight, even-parity
measured presentations and circuit fault distance are later acceptance stages,
not assumptions about the static row space.

Run a checkpointed `n = 14` pilot with:

```bash
.venv/bin/python -m searches.inverse_c2_minimum.run_search \
  --output-root results/inverse-c2-minimum \
  --run-name n14-w8-pilot \
  --length 14 --maximum-check-weight 8 --maximum-tasks 20 \
  --checkpoint-every-models 5 --stop-on-hit
```

Every task writes its transcript, distance cuts, best model, and summary
atomically. A hard task can be continued in an isolated new run using its
saved cut file with `--initial-cuts`, together with one geometry, fold index,
and module type.

`derive_from_n32.py` separately constructs the exact `T^2` quotient of the
saved `[[32,4,6]]` code and exhausts all compatible folded rank-seven gauge
fixings. This provides a reproducible test of whether the smaller target can
be inherited from the known code rather than synthesized independently.

`local_refinement.py` and `run_local_refinement.py` refine saved distance-four
frontiers by replacing one or two complete `C2` check orbits. The runner keeps
the weight ceiling on the explicit orbit presentation, validates every
connected neighbor at exact distance, writes one atomic result per start, and
prints a checkpoint whenever a worker finishes. A two-orbit bridge run is:

```bash
.venv/bin/python -m searches.inverse_c2_minimum.run_local_refinement \
  --input results/inverse-c2-minimum/n16-w6-s2-2-s1-0-folds00-04-w8-260830-v1 \
  --input results/inverse-c2-minimum/n16-w6-s2-2-s1-0-folds05-09-w8-260830-v1 \
  --input results/inverse-c2-minimum/n16-w6-s2-2-s1-0-folds30-39-w8-260830-v1 \
  --output results/inverse-c2-minimum/n16-two-orbit-refinement \
  --repetitions-per-seed 8 --iterations 1000 \
  --attempts-per-replacement 2000 --replacement-width 2 \
  --downhill-distance-floor 3 --downhill-probability 0.05 \
  --maximum-check-weight 8
```
