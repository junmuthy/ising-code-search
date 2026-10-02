# Reproducible [[64,16,6]] code: 313c9898

This is the connected CSS code with all 16 canonical logical X and all 16
canonical logical Z representatives of weight six, regular logical
`C4 x C4` symmetry, and individual logical Hadamards up to a qubit
permutation. Logical supports may overlap.

The preferred schedule `30f5d7baeecc8a92` uses 24 weight-12 checks of each
Pauli type, 48 dedicated ancillas (112 total qubits), **576 CNOTs and CNOT
depth 12 per round**. Both memory bases have **`5 <= d_fault <= 6`** under
the saved three-round `full_wait` Pauli-noise experiment.

The code also has an optimal independent check basis with 16 weight-eight
and eight weight-twelve checks per type, totaling 448 incidences. That
basis is saved separately; it is NOT the measured basis of the certified
depth-12 circuit. The tested 448-CNOT depth-20 orderings all failed at four
faults or fewer.

## Reproduce

The package has no runtime dependency on the qLDPC checkout, its original
absolute paths, the surrounding GALA branch, or network access. It vendors
the exact algebra, circuit, replay, and fault-search routines it needs.

Using the existing GALA environment, from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/gala-313-numba ../.venv/bin/python reproduce.py
```

This checks every compact-manifest hash, regenerates the group-algebra checks and
schedule, rechecks exact static distance and logical symmetries, rebuilds
both circuits, validates both saved lower certificates, and physically
replays the six-fault upper witnesses. It does **not** rerun the expensive
lower-bound fault search unless requested.

For a fresh exhaustive one-through-four-fault exclusion in both bases:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/gala-313-numba ../.venv/bin/python reproduce.py --exact-four --output /tmp/gala-313-fresh
```

The output directory must not already exist. The compact pair tables use
about 0.62 GB each, sequentially, plus process overhead. A typical run on
the original machine takes under a few minutes. For the independent,
original pair-table engine, add `--engine reference`; allow several
minutes and roughly 3 GB per pair table plus additional working memory.

To set up another machine, use Python 3.12 and install the pinned packages
in `requirements.txt` into a virtual environment. Replace `../.venv/bin/python`
with its interpreter. No package installation is performed by the script.

Run regression tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/gala-313-numba ../.venv/bin/python -m pytest -o addopts='' -p no:cacheprovider -q tests
```

## Construction

Physical coordinates are `q = 32*s + 4*i + j`, with `s` in `{0,1}`, `i`
modulo eight and `j` modulo four. The two coefficient bitmasks are
`a = 34865219`, `b = 1611923744`. The translation-closed X checks are
`H_X = [L_a | L_b]`, over `F2[C8 x C4]`, and `H_Z = P(H_X)`, where

```text
P(0,i,j) = (1,3*i,-j)
P(1,i,j) = (0,3*i+4,-j).
```

The physical fold has order four. Physical translations have orders eight
and four, while their induced logical translations both have order four.
The logical Hadamard relabeling is `(i,j) -> (2-i,3-j)` modulo four.

`construction.json` stores the twelve seed-edge times of the valid
translation-closed parent and the retained full-rank check indices. The
reproduction script regenerates every CNOT layer from these parameters,
not merely by copying the final circuit.

## Contents

- `preferred/`: standalone preferred code, binary matrices, logical basis,
  schedule, X/Z circuits, and six-fault witnesses.
- `minimum_weight_checks/`: the alternative 448-incidence check basis.
- `original/`: the valid redundant depth-12 parent schedule and checks.
- `certificates/`: preferred-circuit certificates from both exact engines.
- `reproduce.py`, `lib/`, `tests/`: portable reproduction and verification.
- `archive/source/`: byte-preserved original search and analysis source snapshots.
- Other `archive/` directories: optional local history, including unsuccessful
  schedules and intermediate audits; excluded from the compact Git bundle.
- `PROVENANCE.json`: source hashes, vendoring/extraction map, versions,
  and original repository revisions.
- `MANIFEST.reproduction.json`: relative-path integrity inventory for the
  compact Git bundle; used by default by `reproduce.py`.
- `MANIFEST.json`: historical full-archive inventory retained locally, not
  required or included in the compact Git bundle. Its hashes describe the
  original package before this packaging-only update.
- `REPRODUCTION_VALIDATION.json`: results of testing the assembled package.

Historical reports under `archive/` and copied handoff reports retain their
original path strings for provenance. They are not runtime dependencies;
use the top-level reproduction command instead of running archived scripts
with their old path assumptions. Vendored function bodies are copied
unchanged; their import headers are made local and recorded in provenance.

The distance claim is specific to three noisy memory rounds, including
CNOT, preparation, measurement, and idle faults. It does not establish
exact fault distance six, longer-memory performance, decoder error rates,
resource-state preparation, or noisy Hadamard routing. Reset and readout
durations are not included in CNOT depth.

Saving this directory does not switch Git branches or create a commit.
