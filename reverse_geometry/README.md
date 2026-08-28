# Natural-`S3` reverse-geometry search

This isolated search targets a minimal two-block, `n=192` represented lift
over

```text
S3(natural degree 3) x C8 x C4.
```

It reverses the earlier generator-first workflow:

1. construct a weight-seven translated logical grid;
2. require disjoint supports within the two 16-site checkerboard batches;
3. require an exact permutation logical ZX pairing;
4. compute the complete fold-tied polynomial annihilator;
5. pass only rank-capable spaces to the sparse CSS-generator stage.

The initial fold fixes one natural-`S3` coordinate and exchanges the other
two.  A singleton on the fixed sheet supplies the odd logical pairing, while
identical supports on exchanged sheet pairs cancel over `GF(2)`.

Run without overwriting earlier results:

```bash
.venv/bin/python -m reverse_geometry.run_n192 \
  --run-name natural-s3-w7-seed260826
```

Results are saved under `results/reverse-geometry/<run-name>/`.

Run the bounded fold-tied, weight-at-most-12 generator pilot with:

```bash
.venv/bin/python -m reverse_geometry.run_n192_generators \
  --source results/reverse-geometry/natural-s3-w7-seed260826/geometry-and-annihilator.json \
  --run-name natural-s3-w12-fold-tied-pilot \
  --minimum-terms 6 --maximum-terms 6 \
  --require-no-zero-columns \
  --require-nonzero-top-augmentation
```

Apply the complete weight-one-through-four logical screen to structural hits:

```bash
.venv/bin/python -m reverse_geometry.certify_n192 \
  --source results/reverse-geometry/natural-s3-w12-fold-tied-exact6-nozero-pilot/fold-tied-generator-pilot.json \
  --run-name natural-s3-w12-fold-tied-exact6-certification
```

If the invariant natural-`S3` triplet becomes a weight-three logical, replace
the cancellation template by general batched seeds whose top-parity
projections are nonzero in both physical halves:

```bash
.venv/bin/python -m reverse_geometry.run_n192_projected_geometry \
  --run-name natural-s3-projected-geometry-seed260828
```

Rank the projected witnesses' complete fold-tied annihilator spaces with:

```bash
.venv/bin/python -m reverse_geometry.run_n192_annihilators \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-annihilators
```

Use three-plus-three meet-in-the-middle synthesis when the strengthened sparse
MILP times out:

```bash
.venv/bin/python -m reverse_geometry.run_n192_mitm \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-sparse-mitm \
  --weights 4 5 6
```

Enumerate all six natural-`S3` check-coordinate permutations while keeping
the same physical data fold with:

```bash
.venv/bin/python -m reverse_geometry.run_n192_check_folds \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-check-folds
```

Drop the displayed-row relation entirely and test ZX duality only on the
resulting check row spaces with:

```bash
.venv/bin/python -m reverse_geometry.run_n192_rowspace \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-rowspace-pilot
```

Project out the exactly inconsistent random-`F` space and enumerate only
logical-compatible sparse `F` supports with:

```bash
.venv/bin/python -m reverse_geometry.run_n192_rowspace_guided \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-rowspace-guided
```

Screen all saved geometries for their minimum compatible `F` with nonzero
top augmentation, then spend the rest of the formal weight-12 budget on `G`:

```bash
.venv/bin/python -m reverse_geometry.run_n192_rowspace_minimum \
  --source results/reverse-geometry/natural-s3-projected-geometry-seed260828/projected-geometry.json \
  --run-name natural-s3-projected-rowspace-minimum-all100 \
  --witnesses 100 --f-trials 1 \
  --seconds-per-f-solve 3 --seconds-per-g-solve 3
```

The guided runner accepts `--indices` for a comma-separated subset of saved
witnesses.  This is useful for longer certification passes without rewriting
or filtering the source geometry catalog.
