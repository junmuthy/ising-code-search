# Many-copy packed-code search

This directory contains notes for the scalable
packed BB/GALA family over `C_(56r) x C_4`. The Python implementation lives in
`gala_search/many_copy.py`; machine-readable outputs and caches live under
`results/many-copy/`.

The first target uses `r=2`, so one `[[896,128,d]]` stabilizer code contains
four independent `C_8 x C_4` logical grids. The preferred search tier uses six
polynomial monomials, check weight 12, and qubit degree 6.

Run the structural sweep with:

```bash
.venv/bin/python -m searches.many_copy.run_many_copy_search --copies-per-half 2 --m 7
```

Certify the recommended seed with:

```bash
.venv/bin/python -m searches.many_copy.certify_many_copy
```

The completed search is reported in [`../MANY_COPY_RESULTS.md`](MANY_COPY_RESULTS.md).
The certificate includes an explicit collision-free, depth-optimal 12-layer
check-interaction schedule for each CSS type.
