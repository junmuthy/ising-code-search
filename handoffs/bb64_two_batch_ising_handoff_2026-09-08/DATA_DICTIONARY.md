# Data dictionary

The authoritative archive is `bb64_two_batch_basis.npz`.  Its binary arrays
use NumPy `uint8` values with arithmetic over `GF(2)`; permutation and batch
vectors use `int64`.  Each one-dimensional CSV export is stored as one row.

| Key | Shape | Meaning |
|---|---:|---|
| `matrix_x` | `32 x 64` | Displayed X-check matrix `H_X`. |
| `matrix_z` | `32 x 64` | Displayed Z-check matrix `H_Z=H_X`. |
| `logical_z` | `8 x 64` | Ordered, weight-eight Z logical representatives. |
| `logical_x` | `8 x 64` | Canonical, weight-eight X logical representatives. |
| `zx_pairing` | `8 x 8` | Same-support Z/Z Gram matrix and Hadamard permutation matrix. |
| `hadamard_permutation` | `8` | Map `p=[5,4,7,6,1,0,3,2]`. |
| `grid_x_physical_permutation` | `64` | Physical permutation inducing the logical order-four translation. |
| `grid_y_physical_permutation` | `64` | Physical permutation inducing the logical order-two translation. |
| `grid_x_z_logical_action` | `8 x 8` | Induced X-direction action on Z logical classes. |
| `grid_y_z_logical_action` | `8 x 8` | Induced Y-direction action on Z logical classes. |
| `grid_x_x_logical_action` | `8 x 8` | Induced X-direction action on X logical classes. |
| `grid_y_x_logical_action` | `8 x 8` | Induced Y-direction action on X logical classes. |
| `disjoint_batch_of_z_logical` | `8` | Z batch label for each logical index. |
| `disjoint_batch_of_x_logical` | `8` | X batch label for each logical index. |

The permutation convention is `output[permutation[i]] = input[i]`.  Physical
qubits `0..31` form the L half and `32..63` form the R half.  Within a half,
the twisted-torus coordinate `(x,y)` has index `8*x+y`, where `0<=x<4` and
`0<=y<8`.

`code_metadata.json` repeats the logical supports and principal maps as JSON
lists for consumers that do not use NumPy.  Every NPZ key also has a same-name
file in `matrices_csv/`.
