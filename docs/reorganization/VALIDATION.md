# Reorganization validation

Validation uses the existing repository `.venv` and editable qLDPC installation.
The baseline was recorded at `a68edde` before moving files.

## Working tree

- Shared suite: **205 passed** before and after the moves.
- Eight project-local suites (**89 tests**) pass before and after; they run in separate pytest
  processes to preserve the standalone projects' module-loading conventions.
- Three project-local suites have the same pre-existing collection failures:
  `one_block_c56xc4` lacks `algebra`; the surface-code demo requires `tsim`;
  the 313c9898 simulation requires `clifft`. No new failures were introduced.
- All 39 relocated loose entry points and five representative project entry
  points run. 43 support `--help`; `analyze_s3_candidate_folds` runs its audit
  directly. Its regenerated outputs were restored to the original saved bytes.
- Artifact hashes confirm all 6,649 inventoried files remain present, with no
  changes to untracked/ignored files, saved data, or frozen bundle contents.

Project-local suites checked:

```text
codes/c28xc4_component
searches/c28xc4_direct_d6
searches/one_block_c56xc4
searches/inverse_c4_minimum
searches/inverse_c4_n30
searches/inverse_c4_n32_targeted
codes/n32_k4_d6_reference_code
searches/gl2_c4xc4_n128_preflight
codes/c4xc4_64_16_6_313c9898/tests
simulations/surface_d3_rz_tsim/tests
simulations/c4xc4_313c9898_tmr_postselection/tests
```

## Clean checkout

The final committed layout is checked in a separate worktree using the same
Python environment. Clean-checkout results and any missing local inputs are
recorded after that check.
