# Searches

Each campaign keeps its scripts and scientific reports together. The linked
report describes its achieved results, exclusions, and unresolved cases. Existing
outputs remain at their previous relative location under the repository-level
`results/` tree or inside the campaign.

| Campaign | Purpose | Results or status |
| --- | --- | --- |
| [`paired_polynomial/`](paired_polynomial/) | Original paired-polynomial searches and exact certification | [`RESULTS.md`](paired_polynomial/RESULTS.md) |
| [`packed/`](packed/) | Packed 2D Ising-lattice searches | [`PACKED_RESULTS.md`](packed/PACKED_RESULTS.md) |
| [`many_copy/`](many_copy/) | Scalable packed-code searches | [`MANY_COPY_RESULTS.md`](many_copy/MANY_COPY_RESULTS.md) |
| [`s3/`](s3/) | Nonabelian S3 searches, folds, pairing, and resize studies | [S3_RESULTS.md](s3/S3_RESULTS.md) |
| [`abelian/`](abelian/) | Abelian 32-qubit and order-seven-fibre searches | [`ORDER_SEVEN_FIBRE_RESULTS.md`](abelian/ORDER_SEVEN_FIBRE_RESULTS.md) |
| [`half_grid/`](half_grid/) | Seed-first and batched half-checkerboard searches | [`HALF_GRID_SEARCH.md`](half_grid/HALF_GRID_SEARCH.md) |
| [`single_row/`](single_row/) | Single-row searches and abelian/floating-dimension studies | [SINGLE_ROW_32_SEARCH.md](single_row/SINGLE_ROW_32_SEARCH.md) |
| [`batched_c4xc2_search/`](batched_c4xc2_search/) | Relaxed-batching Ising-grid searches | [`RESULTS.md`](batched_c4xc2_search/RESULTS.md) |
| [`bb64_simultaneous_basis_search/`](bb64_simultaneous_basis_search/) | BB64 logical bases, batchings, and fault-aware schedules | [`README.md`](bb64_simultaneous_basis_search/README.md) |
| [`bb288_c4xc8_automorphisms/`](bb288_c4xc8_automorphisms/) | Published BB288 code automorphism screen | [`RESULTS.md`](bb288_c4xc8_automorphisms/RESULTS.md) |
| [`bb288_k40_c4xc8_automorphisms/`](bb288_k40_c4xc8_automorphisms/) | Published k=40 BB288 code automorphism screen | [`README.md`](bb288_k40_c4xc8_automorphisms/README.md) |
| [`c28_one_block/`](c28_one_block/) | One-block C28 search | [`RESULTS.md`](c28_one_block/RESULTS.md) |
| [`c28xc4_direct_d6/`](c28xc4_direct_d6/) | Direct C28 x C4 distance-six search | [`RESULTS.md`](c28xc4_direct_d6/RESULTS.md) |
| [`c2xc7_bb/`](c2xc7_bb/) | Self-dual C2 x C7 BB search | [`RESULTS.md`](c2xc7_bb/RESULTS.md) |
| [`c4_invariant_lagrangian/`](c4_invariant_lagrangian/) | Direct C4-invariant 28-qubit search | [`RESULTS.md`](c4_invariant_lagrangian/RESULTS.md) |
| [`c4_invariant_n32/`](c4_invariant_n32/) | Direct C4-invariant 32-qubit search | [`RESULTS.md`](c4_invariant_n32/RESULTS.md) |
| [`c4xc4_floating/`](c4xc4_floating/) | Floating-dimension C4 x C4 search | [`RESULTS.md`](c4xc4_floating/RESULTS.md) |
| [`c4xc5_search/`](c4xc5_search/) | C4 x C5 single-row search | [`RESULTS.md`](c4xc5_search/RESULTS.md) |
| [`d4_single_row/`](d4_single_row/) | Faithful D4 single-row search | [`RESULTS.md`](d4_single_row/RESULTS.md) |
| [`distance_first_c2/`](distance_first_c2/) | Distance-first C2 code search | [`RESULTS.md`](distance_first_c2/RESULTS.md) |
| [`gl2_c4xc4_n128_preflight/`](gl2_c4xc4_n128_preflight/) | 128-qubit logical-grid preflight | [`RESULTS.md`](gl2_c4xc4_n128_preflight/RESULTS.md) |
| [`inverse_c2_minimum/`](inverse_c2_minimum/) | Minimum-size C2 folded-CSS synthesis | [`RESULTS.md`](inverse_c2_minimum/RESULTS.md) |
| [`inverse_c4_minimum/`](inverse_c4_minimum/) | Minimum-size C4 folded-CSS synthesis | [`RESULTS.md`](inverse_c4_minimum/RESULTS.md) |
| [`inverse_c4_n30/`](inverse_c4_n30/) | Mixed-orbit 30-qubit synthesis | [`PARTIAL_RESULTS.md`](inverse_c4_n30/PARTIAL_RESULTS.md) |
| [`inverse_c4_n32_targeted/`](inverse_c4_n32_targeted/) | Targeted 32-qubit folded-ZX continuation | [`RESULTS.md`](inverse_c4_n32_targeted/RESULTS.md) |
| [`one_block_c56xc4/`](one_block_c56xc4/) | One-block C56 x C4 search | [`RESULTS.md`](one_block_c56xc4/RESULTS.md) |
| [`reverse_geometry/`](reverse_geometry/) | Geometry-first natural-S3 search | [`RESULTS.md`](reverse_geometry/RESULTS.md) |

Run package commands from the repository root, for example:

```bash
.venv/bin/python -m searches.paired_polynomial.run_search --help
.venv/bin/python -m searches.s3.run_s3_ising_search --help
.venv/bin/python -m searches.c4xc5_search.run_search --help
```

Standalone scripts retain their project-specific instructions. Reusable algebra
remains in [`gala_search/`](../gala_search/). The BB64 and n32 schedule studies
share fault-analysis helpers across this category and [`codes/`](../codes/README.md).
