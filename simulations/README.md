# Simulations

Dedicated circuit, noise, recovery, and postselection experiments live here.
Scripts, reports, plots, and project-local outputs stay together.

| Project | Purpose and status |
| --- | --- |
| [`c4xc4_313c9898_tmr_postselection/`](c4xc4_313c9898_tmr_postselection/README.md) | One- and four-resource TMR experiments using the frozen 313c9898 bundle. Requires `clifft`. |
| [`surface_d3_rz_tsim/`](surface_d3_rz_tsim/README.md) | Noiseless surface-code TMR demonstration. Requires `tsim`. |
| [`postselection_comparison/`](postselection_comparison/BB56_BB64_ACCEPTANCE_VS_N.md) | BB56/BB64 acceptance comparison, CSV, plot, and plotting script. |
| [`bb64_hybrid_nonclifford/`](bb64_hybrid_nonclifford/README.md) | Saved decoder output; implementation on `bb64-hybrid-nonclifford`. |
| [`bb64_tmr_postselection/`](bb64_tmr_postselection/README.md) | Saved manifests and outputs; implementation on `bb64-tmr-postselection` and follow-up branches. |
| [`bb64_syndrome_recovery/`](bb64_syndrome_recovery/README.md) | Local artifacts; implementation on the BB64 development branches. |

## Experiments kept with reference codes

- [Bicycle chain L2](../codes/bicycle_chain_l2_paper/README.md): nested TMR implementation.
- [Bicycle chain L4](../codes/bicycle_chain_l4_paper/README.md): nested TMR artifacts and branch information.
- [Shortened Golay n22](../codes/n22_k2_d6_shortened_golay/README.md): nested TMR/teleportation artifacts and branch information.

From the repository root, the 313c9898 entry point is:

```bash
.venv/bin/python -m simulations.c4xc4_313c9898_tmr_postselection.run --help
```

Install each experiment's documented dependencies in its intended environment.
The repository's current `.venv` lacks `clifft` and `tsim`; those test collection
failures existed before the reorganization. Existing local data is preserved,
but ignored outputs are not available in a clean checkout.

Development branches retain their old paths until integrated. Use the
[branch-integration instructions](../docs/reorganization/README.md) when bringing
those implementations into this layout.
