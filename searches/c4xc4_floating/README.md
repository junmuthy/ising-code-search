# Floating-`k` self-dual `C4 x C4` search

This isolated branch exhausts every self-dual two-block BB polynomial
`a in GF(2)[C4 x C4]` of weight at most six:

```text
H_X = H_Z = [A | A^T].
```

The code length is 32 and the check weight is at most 12.  Unlike the earlier
`[[32,4,*]]` search, this branch permits every `k >= 4` and asks only for one
protected orbit of four pairwise-disjoint weight-seven logicals under the
first `C4` translation.

Run without overwriting an earlier result:

```bash
NUMBA_CACHE_DIR=/tmp/gala-numba-cache \
  .venv/bin/python -m searches.c4xc4_floating.run_search \
  --run-name exhaustive-w12-all-k-v2
```

The completed result is reported in [`searches/paired_polynomial/RESULTS.md`](RESULTS.md).
