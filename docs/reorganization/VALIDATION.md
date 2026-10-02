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
- Notebook setup succeeds with the kernel started in `notebooks/`; saved cell
  outputs are unchanged. All checked local Markdown links retain valid targets
  where they were valid before the move, and new navigation links resolve.

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

The committed layout at `0f96edf` was checked in a separate worktree using the
same Python environment, without copying ignored data from the working checkout.

- Shared suite: **203 passed, 2 failed** because saved, ignored search inputs
  are absent. Both failures were reproduced against a clean export of the
  original `a68edde` baseline.
- Project-local suites: **87 passed, 2 failed**, plus the same three collection
  failures listed above. The two additional failures read the same ignored
  folded-seed dataset. Those tests now locate data relative to their checkout;
  their previous hard-coded path pointed into the primary working checkout.
- All **43 CLI help checks passed** without local data.
- All **three handoff package verifiers passed** from their new directories.
- The frozen 313c9898 bundle's **67 tests passed** in both working and clean
  checkouts; the shared suite also verifies the certified CSS collection.

The missing local inputs are:

| Tests | Input relative to the repository root |
| --- | --- |
| Saved n28-to-n32 extension | `results/c4-invariant-lagrangian/orbit-replacement-all75x500-v1/best-candidates.jsonl` |
| Saved C2 local-refinement seed | `results/inverse-c2-minimum/n16-w6-s2-2-s1-0-folds00-04-w8-260830-v1/tasks/task-0000-*-best.json` |
| Both project-local folded-seed tests | `searches/inverse_c4_n32_targeted/results/shifted-identity-thickness-all100-260828-v1/folded-seeds.jsonl` |

These datasets remain intact in the primary working checkout. They were not
newly committed or regenerated to make the clean checkout pass. Tests requiring
them should run in a checkout with those saved inputs available. The notebook
likewise uses its pre-existing saved certificate from the shared `results/` tree.

## Git and preservation

- Git recognizes **833 renames** across the handoff and main layout commits;
  there are no deleted original tracked files.
- Every original file retains its tracked or local-only status. The broad
  existing ignore patterns already cover the new directory depths.
- All seven original local branch tips and existing linked worktrees are
  unchanged. The pre-migration Git bundle verifies successfully.
- Staged whitespace checks pass. The main working tree has no outstanding
  tracked changes after committing; pre-existing untracked artifacts remain.

Detailed inventories, logs, and recovery copies are local under `/tmp`; the
[migration notes](README.md) identify the backup. No commits were pushed.
