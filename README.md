# Paired-polynomial GALA search

This standalone project searches the self-dual minimal-GALA family

\[
Q_{4,m}(a)=\operatorname{CSS}([A\mid A^T],[A\mid A^T])
\]

over `GF(2)[C_4 × C_m]`.  Its first search family is

\[
a(x,y)=1+y^r+xy^s+xy^{s+t}.
\]

The project uses the editable qLDPC checkout at
`/home/judah_unmuth/Documents/multistaq/qLDPC`, but keeps scripts, caches, and
results outside that repository.

The completed chain findings are summarized in [`RESULTS.md`](RESULTS.md).  The
packed 2D Ising-lattice search is summarized in
[`PACKED_RESULTS.md`](PACKED_RESULTS.md).

An executable walkthrough of the recommended packed code and its Ising/STAR
acceptance tests is in [`ising_conditions.ipynb`](ising_conditions.ipynb).

Execute and save all notebook outputs with:

```bash
.venv/bin/jupyter execute --inplace --timeout=300 ising_conditions.ipynb
```

## Environment

The local environment has already been created in `.venv` and qLDPC installed
with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e /home/judah_unmuth/Documents/multistaq/qLDPC
```

Set writable cache directories when running several workers:

```bash
export NUMBA_CACHE_DIR="$PWD/results/numba-cache"
export MPLCONFIGDIR="$PWD/results/matplotlib-cache"
```

## Search

Run the algebraic filters and 20 randomized BP+OSD distance trials:

```bash
.venv/bin/python run_search.py --m 7 11 13 --trials 20
.venv/bin/python collect_results.py
```

The search quotients torus translations and independent reflections of the two
cyclic axes.  Pass `--no-symmetry-quotient` to retain every distinct normalized
polynomial support.

The randomized distance result is always recorded as an upper bound.  The known
weight-`m` column logicals are also applied as an analytic upper bound.

## Exact distance

Certify the known bicycle-chain seed:

```bash
.venv/bin/python certify_distance.py --m 7 --r 3 --s 2 --t 2
```

Or certify every promising `m=7` representative from a search result:

```bash
.venv/bin/python certify_results.py --m 7 --min-bound 6 --workers 2
```

To rule out candidates above a target without spending eight ILPs on each one,
use threshold mode.  Every returned `ilp_distance_upper_bound` is backed by an
explicit logical witness, while `certified_distance` remains empty unless all
eight logical constraints were solved:

```bash
.venv/bin/python certify_results.py \
  --input results/weight8-m13-2000.jsonl \
  --m 13 --min-bound 9 --stop-at 8 --workers 4
```

The certification solves eight binary minimum-weight ILPs.  If `L` is the known
orthonormal logical-support matrix, a zero-syndrome vector `e` is nontrivial iff
at least one component of `L @ e` is one.  The distance is consequently the
minimum of the eight problems

```text
min |e|  subject to  H @ e = 0 and L[i] @ e = 1.
```

Only a complete run with every solver status equal to `optimal` is merged into
the report as `certified_distance`.

## Packed `C_8 x C_4` logical lattice

Search the two-pair weight-eight ansatz over `C_56 x C_4`:

```bash
.venv/bin/python run_packed_search.py --family weight8 --m 7 --workers 8
```

This search is exhaustive modulo translations and axis reflections.  The
two-pair family has an unavoidable disconnected-quotient obstruction.  Search
the minimal connected three-pair family, which has weight-12 checks, with:

```bash
.venv/bin/python run_packed_search.py --family weight12 --m 7 --workers 8
```

Certify selected accepted codes against `d >= 6`:

```bash
.venv/bin/python certify_packed.py \
  --input results/packed-weight12.jsonl \
  --output results/packed-weight12-certifications.jsonl \
  --family packed_weight12 --m 7 --target-distance 6 --workers 8
```

The 32 fibre logicals in either physical half are related by exact physical
translations.  Distance certification therefore requires one ILP from each
half rather than 64 redundant translated problems.

## Tests

```bash
.venv/bin/python -m pytest
```

The regression suite constructs the `[[56,8,6]]` chain seed and the packed
`[[448,64,7]]` code, validates their canonical logical supports and physical
automorphisms, and certifies their exact distances with HiGHS.
