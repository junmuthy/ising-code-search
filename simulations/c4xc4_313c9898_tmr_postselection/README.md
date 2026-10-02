# 313c9898: one- and four-resource TMR post-selection

This package analyzes the frozen `[[64,16,6]]` code in the adjacent
`codes/c4xc4_64_16_6_313c9898` directory. It does not modify that reproducibility
bundle, its canonical logicals, or any older experiment. Only logical `0`
(`N=1`) and logicals `{0,2,4,6}` (`N=4`, zero-based indices) are injected.
All sixteen logicals are retained in the encoder and ideal-output checks.
Eight- and sixteen-resource experiments are intentionally excluded.

The completed first-milestone findings are in [RESULTS.md](RESULTS.md), with
the full numerical table in [reports/production/RESULTS.md](reports/production/RESULTS.md).

## Protocol and exact checks

Each sorted weight-six logical Z support is partitioned into three pairs.
The partial-X-syndrome ranks are exactly 2 and 8 for one and four logicals,
respectively. The kernels contain only the choices of none/all three pieces
of each logical. An explicit enumeration of all 8 or 4096 rotation-expansion
terms independently checks the output on the complete sixteen-logical space.

We use `Rz(theta) = exp(-i theta Z/2)`. For `M=3`, choose
`phi = -2 atan(cuberoot(tan(theta/2)))` for the small positive angles in this
study. The physical syntax `R_Z(phi/pi)` implements this convention. The
ideal acceptance is `[cos(phi/2)^6 + sin(phi/2)^6]^N`. M=1 is also implemented
as a control, but the primary sweep uses M=3.

Full preparation: initialize physical plus states; measure Z checks;
apply the syndrome-dependent virtual X correction; apply one parallel TMR
batch; extract one full X/Z syndrome round. The initial Z outcomes are not
postselected. Initial redundant-check relations are recorded diagnostically
but also are not rejection conditions in this historical-comparison baseline.

`tmr-x` rejects nonzero terminal X outcomes. `strict-xz` rejects nonzero
terminal X or Z outcomes. Both are evaluated on identical sampled shots.
No correction/recovery of retained nonzero Z branches is modeled.

## Saved schedules and resource counts

The preferred presentation measures **24 X and 24 Z checks, all weight 12**.
Its full syndrome round uses **576 CNOTs at CNOT depth 12**. The redundant
parent measures 32 weight-12 checks per sector and uses 768 CNOTs at the
same full-round depth. The code's alternative basis with 16 weight-8 and
8 weight-12 checks per sector is saved separately and is **not** used in
these simulations.

The following counts are for the complete resource-preparation attempt,
not just one syndrome round:

| Presentation | Ancillas | Total qubits | CNOTs, N=1 | CNOTs, N=4 | CNOT depth |
|---|---:|---:|---:|---:|---:|
| `preferred` | 48 | 112 | 870 | 888 | 26 |
| `original` redundant parent | 64 | 128 | 1158 | 1176 | 26 |

The full round alone has depth 12. Full preparation contains 12 initialization
CNOT layers, 2 TMR CNOT layers, and 12 terminal CNOT layers, plus one physical
rotation layer. Reset/readout durations and hardware routing are excluded.
TMR uses 3 or 12 physical rotations. No early rejection saves physical work
in this single-terminal-projection baseline.

Resource costs use `N * acceptance` candidate resources per attempt; all N
resources are discarded on rejection. Allocated-qubit CNOT-layer cost is an
explicit timing proxy, not calibrated hardware wall time. No fidelity claim
is attached to an accepted candidate. The saved memory fault-distance
certificate does not transfer to this resource-preparation circuit.

## Noise convention

The existing `codes/bicycle_chain_l2_paper/tmr_postselection` circuit primitives
are reused unchanged, also as in the BB56 cross-check. At probability p:

- D1(p) after physical data/ancilla preparations and physical rotations;
- D2(p) after each physical CNOT;
- measurement-bit flips with probability p;
- D1(p) on live idle qubits in each CNOT layer.

Initialization live qubits are data and Z ancillas; TMR live qubits are data;
the terminal round has data and both ancilla sets live. There is no separate
idle noise for reset, readout, or the single rotation layer, nor loss,
coherent calibration noise, feedback latency, or noisy teleportation. This
is the historical acceptance-comparison convention, not the memory
`full_wait` convention. Virtual X-frame feedback has no physical gate cost.
The zero-angle control retains its TMR gates and noise locations.

## Reproduce

From the GALA repository root, use the existing ClifT environment:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m unittest discover -s simulations/c4xc4_313c9898_tmr_postselection/tests -v

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m simulations.c4xc4_313c9898_tmr_postselection.validate \
  --output /tmp/313-output-validation.json

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m simulations.c4xc4_313c9898_tmr_postselection.run \
  --output simulations/c4xc4_313c9898_tmr_postselection/results/reproduction \
  --shots 200000 --threads 4 --max-seconds 300
```

The saved production dataset instead uses `--shots 1000000
--checkpoint-shots 100000 --threads 4 --seed 3139898`; all other scientific
defaults are unchanged. Its manifest records every setting. The validated
environment is Python `3.12.3`, NumPy `2.5.2`, Stim `1.16.0`, and the existing
ClifT development build `0.10.0rc2.dev1+gdbc721208`. The ClifT build is not
assumed to be downloadable from a public package index.

Generate a report from completed results with:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m simulations.c4xc4_313c9898_tmr_postselection.summarize \
  --input simulations/c4xc4_313c9898_tmr_postselection/results/reproduction \
  --output /tmp/313-reproduction-report
```

Append `--resume` to continue after the wall-time budget. Resuming requires
identical source hashes, frozen inputs, scientific configuration, simulator
versions, shot target, chunk size, thread count, and seeds. Each chunk uses
a distinct deterministic seed; pilots and production use separate seeds.
The runner never stops at a survivor target. Confidence intervals are
pointwise 95% Wilson binomial intervals for completed fixed-shot runs;
interrupted runs are explicitly marked incomplete.

The runner checks the frozen input hashes, audits schedule collisions and
cross-ancilla propagation, records source/environment hashes and emitted
circuits, and checkpoints after every shot chunk. `GALA313_CODE_ROOT` can
override the location of the unchanged code bundle. No qLDPC runtime
dependency or network access is needed, but the older GALA circuit helpers
and ClifT environment are dependencies.

The exact shared `model.py`, `partitions.py`, and `circuit.py` helpers and
their package initializers are included under
`../bicycle_chain_l2_paper/tmr_postselection/` (and its parent), so a fresh
checkout does not require switching to the paper-analysis branch. Their
original Git revision and byte hashes are in `DEPENDENCY_PROVENANCE.json`.
These are the shared helper subset, not the complete paper-code experiment.
Saved production JSON records and report tables are retained; generated
circuit dumps and pilot/checkpoint outputs can be regenerated.

## Literature and interpretation

- Ismail et al., *Fast and Parallel High-Rate STAR Architecture for Megaquop
  Quantum Simulation*, Appendix C, https://arxiv.org/abs/2606.25011.
- Toshio et al., *Practical Quantum Advantage on Partially Fault-Tolerant
  Quantum Computer*, https://doi.org/10.1103/PhysRevX.15.021057.
- Local `codes/bicycle_chain_l4_paper/tmr_postselection/PAPER_CROSSCHECK.md` explains
  the X-only versus strict-XZ acceptance convention issue.

These results estimate preparation acceptance, not decoded conditional
fidelity, a new fault distance, or full RUS/teleportation performance.
