# Certified CSS codes with regular logical grids

Portable backup of the 64- and 128-qubit results from the September 11,
2026 searches. All three codes have canonical CSS logical bases, a regular
logical grid, and transversal Hadamard followed by a physical permutation
that applies Hadamard to each logical up to a logical permutation. Logical
supports may overlap. The Hadamard and translation actions use the **same**
saved canonical basis.

| Code | Logical grid | Displayed check weights | Saved logical weights | Physical components |
| --- | --- | --- | --- | --- |
| [`[[64,16,8]]`](n64_k16_d8/README.md) | `C4 x C4` | `12` | All `X/Z: 10` | `64` |
| [`[[128,32,6]]`](n128_k32_d6/README.md) | `C4 x C8` | `12` | `X: 33–57`, `Z: 43` (unoptimized) | `64 + 64` |
| [`[[128,32,7]]`](n128_k32_d7/README.md) | `C4 x C8` | `14, 18` | All `X/Z: 11` | `128` |

The distance-six 128-qubit result is the nonabelian group-algebra code;
the connected distance-seven result is an equivariant orthogonal control,
not a nonabelian square-GALA construction. Both earlier results are retained
to distinguish them explicitly. The 64-qubit distance-eight code is not a
component of that distance-six parent.

## Data and conventions

Each code directory contains:

- `candidate.json`: the original candidate, including check rows, logical
  rows, construction recipe, physical translations, and Hadamard action.
- `checks.npz`: binary arrays `hx`, `hz`, with physical qubits as columns.
- `logical_bases.npz`: binary arrays `x`, `z`, one row per logical qubit.
- `supports.json`: the same four arrays as explicit support lists.
- `physical_permutations.json`: normalized names `px`, `py`, and `fold`,
  plus a logically trivial `central` action when present.
- `metadata.json`: final parameters, construction identifier, component
  sizes, and exact-distance upper witnesses.
- `certificates/`: original exact-distance audit and, for the 64-qubit
  code, the logical-support search and audit artifacts.

All coordinates are zero-based. In integer rows, bit `q` means physical
qubit `q`; permutation entry `p[q]` is its **destination**. Logical grid
coordinate `(i,j)` is flattened as `shape[1]*i+j`. In both Pauli sectors,
`px` sends logical `(i,j)` to `(i+1,j)` and `py` sends it to `(i,j+1)`,
modulo stabilizers. Physical orders need not equal logical orders.
`candidate.json` gives the Hadamard logical permutation in
`hadamard.logical_permutation` and the physical fold in `certificate.fold`.

The original JSON and NPZ files were copied byte-for-byte. Some candidate
JSON fields retain a historical distance lower bound of six: the **final**
distance is recorded in `metadata.json` and `certificates/distance_audit.json`.

## Standalone verification

From the repository root, in a Python environment with NumPy:

```bash
python -m certified_css_grid_codes.verify --distance
python -m pytest -q tests/test_certified_css_grid_codes.py
```

The verifier does not import qLDPC, the old search scripts, or external
solvers. It checks checksums, rebuilds each construction, compares the JSON,
NPZ and support-list representations, and checks binary ranks, CSS
commutation, canonical logical pairing, both translation actions, both
Hadamard images, component sizes and explicit distance upper witnesses.
With `--distance` it also exhaustively excludes all smaller nontrivial
pure-X and pure-Z supports, establishing the exact CSS distances. Without
that flag it verifies the algebra and upper witnesses only. Do not use
Python's assertion-disabling `-O` mode.

The lower-bound check uses exact syndrome matching. Every support is split
uniquely into its smallest `floor(weight/2)` sites and its remaining sites;
matching syndromes enumerate all zero-syndrome supports. Each match is
tested for stabilizer membership, so degeneracy is handled explicitly.
For a CSS code this suffices for the full Pauli distance: any nontrivial
mixed logical has a nontrivial pure-X or pure-Z component of no larger
weight. This verifier does not rerun the full logical-basis optimization.

Optional fresh machine-readable report:

```bash
python -m certified_css_grid_codes.verify --distance --output /tmp/grid-code-audit.json
```

Example data access:

```python
import numpy as np
from certified_css_grid_codes.verify import ROOT

with np.load(ROOT / "n64_k16_d8" / "logical_bases.npz", allow_pickle=False) as data:
    lx, lz = data["x"], data["z"]
assert np.array_equal(lx @ lz.T % 2, np.eye(16, dtype=np.uint8))
assert (lx.sum(axis=1) == 10).all()
```

## Provenance and certification boundary

`SOURCES.json` records the original absolute paths and SHA-256 hashes of
copied artifacts; these paths are provenance, not runtime dependencies.
`MANIFEST.json` covers every package file except itself. The source
snapshots and original reports under `provenance/` preserve research
context. They are archival material, not a reconstructed runnable search
workspace; historical relative links and audit source hashes may refer to
earlier script revisions. Use the portable verifier above for this bundle.

This backup deliberately omits large raw search journals, caches, solver
environments and unrelated simulations. It makes no claim of literature
novelty, circuit-level fault tolerance, disjoint logicals, or global
optimality beyond the explicitly scoped archived certificates.
