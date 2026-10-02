# Package manifest

| Path | Purpose |
|---|---|
| `README.md` | Entry point and reproduction commands. |
| `BB64_TWO_BATCH_REPRESENTATION.md` | Complete physical presentation, logical supports, automorphisms, and syndrome circuit. |
| `bb64_two_batch_basis.npz` | Authoritative matrix and logical-basis artifact. |
| `code_metadata.json` | Machine-readable construction, supports, batches, actions, and conventions. |
| `matrices_csv/` | Plain-text export of every array in the main NPZ artifact. |
| `DATA_DICTIONARY.md` | Array shapes and semantics. |
| `PROVENANCE.md` | Attribution, search history, reproduction boundary, and caveats. |
| `CITATION.bib` | BibTeX citation for the published code. |
| `requirements.txt` | Python dependencies for verification and search reproduction. |
| `verify_package.py` | Standalone integrity and algebraic verifier. |
| `verify_distance.py` | Standalone SciPy/HiGHS exact-distance optimization. |
| `reconstruct_code.py` | Reconstructs `H_X=H_Z` from the polynomial and twisted torus. |
| `export_package_data.py` | Regenerates `code_metadata.json` and `matrices_csv/`. |
| `baseline_schedule.json` | Canonical depth-eight, 512-CNOT syndrome schedule. |
| `provenance/two_batch_search.py` | Portable Z3 one-/two-batch search. |
| `provenance/search_helpers.py` | GF(2) helper routines used by the portable scripts. |
| `provenance/search_input_basis.npz` | Input basis used by the two-batch refinement. |
| `provenance/class_candidates.json` | Eight affine logical-class candidates searched. |
| `provenance/search_results.json` | Complete 16-case solver results from the reference run. |
| `provenance/search_summary.json` | Compact reference-run summary. |
| `provenance/source_two_batch_basis.npz` | Checksum-identical retained copy of the preferred output artifact. |
| `provenance/TWO_BATCH_SEARCH_RESULTS.md` | Human-readable interpretation of the solver run. |
| `provenance/generate_representation_document.py` | Regenerates the complete representation document. |
| `fault_distance/` | Supplementary compact circuit-fault certificates and their interpretation. |
| `SHA256SUMS` | SHA-256 digest of every other packaged file. |
