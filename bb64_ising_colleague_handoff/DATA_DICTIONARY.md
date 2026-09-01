# Data dictionary

The primary file is `bb64_complete.npz`.  All binary matrices use GF(2).

| Key | Shape | Meaning |
| --- | ---: | --- |
| `matrix_x`, `matrix_z` | `32 x 64` | Displayed CSS check matrices; they are equal. |
| `matrix_*_independent` | `28 x 64` | Independent row bases for the checks. |
| `independent_*_row_indices` | `28` | Rows selected from the displayed checks. |
| `stabilizer_symplectic` | `64 x 128` | All displayed stabilizers in `[X|Z]` convention. |
| `stabilizer_symplectic_independent` | `56 x 128` | Independent stabilizer basis. |
| `raw_grid_z` | `8 x 64` | Weight-eight regular `C4 x C2` Z grid. |
| `raw_same_support_x` | `8 x 64` | Same supports regarded as X operators. |
| `raw_dual_x` | `8 x 64` | Canonical X basis dual to `raw_grid_z`. |
| `raw_zx_pairing` | `8 x 8` | Pairing of raw Z with same-support X representatives. |
| `raw_disjoint_batches` | `4 x 2` | Four internally disjoint logical batches. |
| `hperm_change_from_raw_z` | `8 x 8` | GF(2) change from raw Z grid to Hadamard basis. |
| `hperm_z`, `hperm_x` | `8 x 64` | Canonical basis with Hadamard-plus-permutation action. |
| `hperm_zx_pairing` | `8 x 8` | Identity canonical pairing. |
| `logical_h_action` | `8 x 8` | Permutation matrix `(0 1)(2 3)(4 5)(6 7)`. |
| `logical_h_permutation` | `8` | Array form of that permutation. |
| `grid_*_physical_permutation` | `64` | Physical `C4` and `C2` generators. |
| `grid_*_physical_matrix` | `64 x 64` | Matrix forms of those permutations. |
| `raw_grid_*_logical_action` | `8 x 8` | Translation actions in the raw grid basis. |
| `hperm_grid_*_[zx]_action` | `8 x 8` | Translation actions in the Hadamard basis. |
| `bottom_group_elements` | `32 x 2` | `(x,y)` label of each qubit within a BB half. |
| `physical_hadamard_permutation` | `64` | Identity: no physical permutation is needed. |

Human-readable CSV copies of the main matrices are in `matrices_csv/`.
Support lists, weights, conventions, and provenance are in
`code_metadata.json`.
