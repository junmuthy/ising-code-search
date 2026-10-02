# Repository reorganization

The `organize-repository` branch groups research by purpose. The baseline is
commit `a68edde`; the original `master` and development branch tips are retained.
The handoff move is isolated in commit `1d3fb4f` with 114 unchanged renames.
The main layout commit `0f96edf` includes the coupled import, data-path, command, notebook,
and documentation changes.

## Path mapping

[`path-map.json`](path-map.json) is the complete old-to-new map. Directory entries
apply to every descendant; exact file entries cover the loose scripts, reports,
notebook, and ZIPs. Examples:

| Previous path | Current path |
| --- | --- |
| `run_search.py` | `searches/paired_polynomial/run_search.py` |
| `bb64_simultaneous_basis_search/` | `searches/bb64_simultaneous_basis_search/` |
| `c4xc7_automorphism_dual_56_8_6/` | `codes/c4xc7_automorphism_dual_56_8_6/` |
| `c4xc4_313c9898_tmr_postselection/` | `simulations/c4xc4_313c9898_tmr_postselection/` |
| `ising_conditions.ipynb` | `notebooks/ising_conditions.ipynb` |

The detailed contents of the previous root README are retained in
[`RESEARCH_GUIDE.md`](../RESEARCH_GUIDE.md), with updated links and commands.
The root README is now the navigation and execution guide. Shared algorithms
remain in `gala_search/`; shared historical outputs remain in `results/`.

## Preservation and Git

Before moving files, all 6,649 project files outside the environment, Git
metadata, and regenerable cache directories were copied to
`/tmp/gala-reorganization-p12w8d0k/files/`. That backup includes ignored and
untracked research artifacts. Its `inventory.json` records sizes, SHA-256 hashes,
and tracking status; `repository.bundle` contains every original branch.
The backup is local and temporary; retain it elsewhere if long-term recovery is
needed. The existing `.venv` and shared result directories were not relocated.

Tracked moves used `git mv`; local-only directories and ZIPs moved alongside
tracked projects without being added to Git. Existing ignore rules apply at the
new depths. Frozen portable bundles, certificates, manifests, archived source,
provenance, and saved data retain their original bytes. Historical paths in those
records describe the original runs and can be translated with the map.

Review the migration with:

```bash
git diff --stat -M master...organize-repository
git diff --name-status -M master...organize-repository
git log --follow -- codes/c4xc7_automorphism_dual_56_8_6/README.md
```

Git infers renames from content; filenames and scientific content were preserved
to make those relationships clear. Untracked and ignored files remain local and
are not protected by commits. The migration does not publish or push changes.

## Integrating development branches

No existing branch or linked worktree is rebased, reset, or modified. Use a
separate integration branch/worktree when combining this layout with an existing
research branch. Merge the layout once, then reconcile the source and artifacts
before updating that research branch.

| Existing branch | Moved projects to reconcile |
| --- | --- |
| `bb64-hybrid-nonclifford` | `simulations/bb64_hybrid_nonclifford`, BB64 recovery/TMR, and `codes/n22_k2_d6_shortened_golay` |
| `bb64-tmr-postselection` | BB64 recovery/TMR under `simulations/`, and the n22 reference under `codes/` |
| `bicycle-chain-l2-analysis` | `codes/bicycle_chain_l2_paper`, BB64 TMR, and the n22 reference |
| `bicycle-chain-l4-postselection` | Both bicycle-chain references under `codes/`, BB64 experiments under `simulations/`, and n22 |
| `tmr-c4xc7-automorphism-dual` | `codes/c4xc7_automorphism_dual_56_8_6`, comparisons under `simulations/`, and branch-only `tmr_postselection/` |
| `tmr-c4xc8-abelian-lp` | `codes/c4xc8_abelian_lp_160_32_6`, comparisons under `simulations/`, and branch-only `tmr_postselection/` |

Git can infer many directory renames during a merge, but branch-only files and
projects that had no tracked implementation on the baseline require explicit
mapping. `branch_only_moves` maps root `tmr_postselection/` to
`simulations/tmr_postselection/`; nested `tmr_postselection/` directories stay
inside their reference projects. Reconcile the two TMR branches on their own
integration branches before deciding how to combine their implementations.

For every newly integrated source file, update absolute imports to the new
namespace, root-relative data paths, and any `__file__` parent-depth assumptions.
Bring over the intended ignore rules and dependencies, retain its local artifacts,
and rerun that branch's tests. Do not rely on merge rename detection to rewrite
Python imports or command strings. The category READMEs identify partial or
branch-only implementations in this checkout.

## Validation

See [`VALIDATION.md`](VALIDATION.md) for baseline comparisons, artifact checks,
command checks, and clean-checkout limits. Full research campaigns are not rerun
as part of a directory migration.
