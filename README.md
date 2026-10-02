# GALA quantum-code search and STAR simulations

Research on quantum error-correcting codes with logical symmetries for
GALA/Ising operations, including code searches, exact certification,
syndrome schedules, circuit-fault analysis, and transversal multi-rotation
(TMR) simulations.

## Find your work

| Directory                            | Contents                                                                                      |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| [Searches](searches/README.md)       | Code, logical-basis, automorphism, and schedule search campaigns; their scripts and reports.  |
| [Simulations](simulations/README.md) | Circuit, recovery, postselection experiments, and comparisons.                                |
| [Reference codes](codes/README.md)   | Code constructions, matrices, logical bases, schedules, certificates, and portable verifiers. |
| [Handoffs](handoffs/README.md)       | Self-contained colleague packages and local ZIP exports.                                      |
| [Notebooks](notebooks/README.md)     | Executable research walkthroughs.                                                             |
| [Documentation](docs/README.md)      | Research background, detailed results index, and repository migration notes.                  |
| [`gala_search/`](gala_search/)       | Shared Python search and algebra library.                                                     |
| [`results/`](results/)               | Existing shared run outputs and caches; many are local and ignored.                           |
| [`tests/`](tests/)                   | Shared regression and integration tests.                                                      |

Start with the [detailed research guide and results index](docs/RESEARCH_GUIDE.md)
for the scientific findings and certification boundaries. Useful reference
artifacts include the [certified CSS grid codes](codes/certified_css_grid_codes/README.md),
the [reproducible 313c9898 code](codes/c4xc4_64_16_6_313c9898/README.md),
and the [automorphism-dual BB56 code](codes/c4xc7_automorphism_dual_56_8_6/README.md).

## Environment and commands

Run repository commands from this directory. The existing environment uses the
editable qLDPC checkout listed in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export NUMBA_CACHE_DIR="$PWD/results/numba-cache"
export MPLCONFIGDIR="$PWD/results/matplotlib-cache"
```

A search and its exact-distance verifier now run as modules:

```bash
.venv/bin/python -m searches.paired_polynomial.run_search --help
.venv/bin/python -m searches.paired_polynomial.certify_distance --m 7 --r 3 --s 2 --t 2
.venv/bin/python -m searches.batched_c4xc2_search.run_search --help
```

Portable bundles retain their standalone commands; follow their own READMEs
from inside the bundle directory. Simulation dependencies vary by project.
Some implementations are on development branches; the category indexes identify
which directories contain only local artifacts or partial source on this branch.

## Checks

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m pytest codes/c4xc4_64_16_6_313c9898/tests
```

Project-local test suites run separately because some standalone projects use
the same unqualified module names. See the [validation record](docs/reorganization/VALIDATION.md)
for the baseline, additional suites, and existing dependency limitations.

## Adding work

Put each new project under its purpose category. Keep its README, code,
certificates, and project-specific artifacts together. Its README should state
the objective, status, command to run, dependencies, inputs, and results location.
Keep general reusable algebra in `gala_search/`. Code-centered studies may keep
nested simulations; link those from the simulations index.

Existing shared outputs remain under `results/`; existing project-local outputs
stay beside their projects. Preserve historical records and use a new run name
for new results. Commit compact scientific artifacts deliberately. Generated
caches, `texfrag/`, notebook checkpoints, and bulk outputs remain ignored.

For old paths and work on other branches, consult the
[reorganization notes and path map](docs/reorganization/README.md).
