# Two-batch `[[64,8,8]]` BB-code handoff

This archive is a self-contained, machine-checkable handoff of the preferred
Liang--Chen self-dual bivariate-bicycle code basis with two internally
disjoint sets of logical supports.  It contains the check matrices, all eight
canonical logical X/Z pairs, the two `4+4` batches, physical and logical
symmetries, the syndrome schedule, the exact search inputs, and standalone
reconstruction and verification scripts.

The preferred Z batches are

\[
\{0,3,4,7\}\sqcup\{1,2,5,6\}.
\]

The X batches are the reverse ordering of those same two sets.  Every logical
representative has weight eight, and the four supports within each batch are
pairwise disjoint.  Supports in different batches may overlap.

## Quick verification

Python 3.10 or newer is recommended.  From the extracted directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_package.py
.venv/bin/python reconstruct_code.py
.venv/bin/python verify_distance.py
```

`verify_package.py` checks file integrity, reconstructs the checks, verifies
the CSS dimensions, logical basis, two exact covers, transversal-H action,
regular `C_4 x C_2` logical action, physical code automorphisms, and the full
depth-eight syndrome schedule.  `verify_distance.py` uses SciPy/HiGHS to
minimize over all nontrivial zero-syndrome Pauli supports and returns optimum
weight eight.  Since `H_X=H_Z`, this establishes both CSS distances.

## Reproduce the two-batch search

The exact search inputs and portable search source are under `provenance/`.
After installing the requirements, run:

```bash
.venv/bin/python provenance/two_batch_search.py \
  --output-dir reproduced_search \
  --seconds-per-case 60
```

The reference run tested all eight saved affine logical-class candidates in
the one- and two-batch formulations.  It returned two satisfiable two-batch
cases and fourteen unsatisfiable cases, with no timeouts.  Candidate `0` is
the preferred basis saved as `bb64_two_batch_basis.npz`.  Runtimes and NPZ
container hashes can vary by platform; compare the arrays with
`bb64_two_batch_basis.npz`.

Regenerate the full human-readable specification with:

```bash
.venv/bin/python provenance/generate_representation_document.py \
  --output reproduced_representation.md
```

## Main files

- `BB64_TWO_BATCH_REPRESENTATION.md` gives the complete qubit-level
  presentation, logical supports, permutations, and all 512 directed CNOTs.
- `bb64_two_batch_basis.npz` is the authoritative compact matrix artifact.
- `code_metadata.json` and `matrices_csv/` provide portable text forms.
- `DATA_DICTIONARY.md`, `PROVENANCE.md`, and `MANIFEST.md` describe the data,
  claim boundary, and package contents.
- `fault_distance/` contains retained compact evidence for the separate
  three-round Clifford memory-circuit fault-distance result.

The code construction and published parameters are due to Zijian Liang and
Yu-An Chen, arXiv:2510.05211v2.  The two-batch logical-basis refinement is a
derived result of the local GALA/Ising search.  See `CITATION.bib` and
`PROVENANCE.md`.
