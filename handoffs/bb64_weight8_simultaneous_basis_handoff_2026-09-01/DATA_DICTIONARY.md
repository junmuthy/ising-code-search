# Data dictionary

The main archive is `bb64_weight8_basis.npz`.  All binary matrices use NumPy
`uint8` entries and arithmetic over `GF(2)`.

| Key | Shape | Meaning |
|---|---:|---|
| `matrix_x` | `32 x 64` | Displayed X-check matrix `H_X`. |
| `matrix_z` | `32 x 64` | Displayed Z-check matrix `H_Z=H_X`. |
| `matrix_x_independent` | `28 x 64` | Independent row basis for `H_X`. |
| `matrix_z_independent` | `28 x 64` | Independent row basis for `H_Z`. |
| `stabilizer_symplectic_independent` | `56 x 128` | Independent X then Z stabilizers in `(x|z)` form. |
| `logical_z` | `8 x 64` | Ordered, all-weight-eight Z logical representatives. |
| `logical_x` | `8 x 64` | Canonical, all-weight-eight X logical representatives. |
| `logical_z_symplectic` | `8 x 128` | Z logicals in `(x|z)` form. |
| `logical_x_symplectic` | `8 x 128` | X logicals in `(x|z)` form. |
| `zx_pairing` | `8 x 8` | Same-support Z--Z Gram matrix; also the Hadamard permutation matrix. |
| `hadamard_permutation` | `8` | Map `p=[5,4,7,6,1,0,3,2]`. |
| `logical_grid_coordinates` | `8 x 2` | Row `i` is `(a,b)` with `i=2a+b`. |
| `grid_x_physical_permutation` | `64` | Physical permutation inducing `(a,b)->(a+1,b)`. |
| `grid_y_physical_permutation` | `64` | Physical permutation inducing `(a,b)->(a,b+1)`. |
| `grid_x_physical_matrix` | `64 x 64` | Matrix form of the physical X-direction permutation. |
| `grid_y_physical_matrix` | `64 x 64` | Matrix form of the physical Y-direction permutation. |
| `grid_x_z_logical_action` | `8 x 8` | Induced X-direction action on Z logical classes. |
| `grid_y_z_logical_action` | `8 x 8` | Induced Y-direction action on Z logical classes. |
| `grid_x_x_logical_action` | `8 x 8` | Induced X-direction action on X logical classes. |
| `grid_y_x_logical_action` | `8 x 8` | Induced Y-direction action on X logical classes. |
| `disjoint_batch_of_z_logical` | `8` | Z batch label for each logical index. |
| `disjoint_batch_of_x_logical` | `8` | X batch label for each logical index. |
| `z_batch_0`, `z_batch_1`, `z_batch_2` | variable | Z exact-cover index sets. |
| `x_batch_0`, `x_batch_1`, `x_batch_2` | variable | X exact-cover index sets. |

The files in `matrices_csv/` contain comma-separated exports of the principal
matrices and vectors for software that does not read NumPy archives.
