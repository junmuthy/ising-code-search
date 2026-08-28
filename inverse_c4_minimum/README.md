# Direct minimum-size `C4` Ising-code synthesis

This directory searches for a single four-spin logical row without first
choosing a BB/GALA polynomial family or a nonabelian group.  It preserves the
required operational structure as hard constraints:

- four pairwise-disjoint logical representatives related by a physical `C4`
  translation;
- a fixed physical permutation (P) normalizing that translation;
- a permutation logical pairing between the logical (Z) grid and its folded
  (X) grid;
- distinct check spaces allowed, but constrained by

  \[
  \operatorname{row}(H_Z)=\operatorname{row}(H_XP);
  \]

- exact CSS orthogonality and (k=4);
- no logical operator of weight below six;
- a `C4`-orbit generating set of stabilizer weight at most 12.

The physical logical gate is therefore (P H^{\otimes n}): transversal
Hadamard followed by fixed relabeling.  The search never permits unrelated
(H_X) and (H_Z).

## `n=24` certificate

Four disjoint distance-six logical representatives fill all 24 qubits.  For
any physical permutation (P), the logical pairing

\[
M_{ij}=z_i\cdot Pz_j
\]

has zero row augmentation:

\[
\sum_jM_{ij}=z_i\cdot P\mathbf1=\operatorname{wt}(z_i)=0\pmod2.
\]

Hence (M(1,1,1,1)^T=0), ruling out ZX duality on four independent logicals.
The runner writes this as `n24-no-go-certificate.json` before beginning the
`n=28` search.

## `n=28` geometry

The canonical weight-six (Z) grid occupies fibres zero through five of
seven physical `C4` fibres.  The simplest fold exchanges fibres five and six.
The folded (X) grid occupies fibres zero through four and six, so corresponding
logical (X/Z) supports overlap on five fibres.  Their pairing is exactly
the `4 × 4` identity matrix.

The fold catalog also includes physical translation reflection and zero-shift
involutions on the five shared fibres.  Every stored fold is checked for:

- (P^2=I);
- (PTP^{-1}=T^{\pm1});
- permutation logical pairing.

## Exact solver and checkpoints

For each fold and cyclic `F2[C4]` module partition of rank 12, Z3 chooses the
complete (X)-stabilizer module.  It proves rank 12, logical-kernel equations,
CSS orthogonality against its folded image, qubit coverage, and the check-weight
ceiling.  Exact coset enumeration then finds low-weight logicals.  Each such
operator is added as a necessary detection constraint and the solver repeats.

If this counterexample-guided loop reaches `unsat`, that individual
fold/module task is an exact negative result.  `unknown` and `model_limit`
remain explicitly bounded results and are never reported as no-go theorems.

During a run:

- `progress.json` is atomically replaced before and after every solver call;
- each returned model is appended to a task transcript;
- every learned distance constraint is appended immediately to a separate
  JSONL file;
- task summaries and candidates are written without waiting for the whole run;
- an existing run directory is never overwritten.

Example focused run:

```bash
../.venv/bin/python inverse_c4_minimum/run_search.py \
  --output-root inverse_c4_minimum/results \
  --run-name canonical-free-v1 \
  --fold-index 0 --module-type 4,4,4 \
  --solver-timeout-seconds 10 \
  --maximum-solver-timeout-seconds 60 \
  --maximum-models-per-task 100
```

Run the focused tests with:

```bash
../.venv/bin/python -m pytest inverse_c4_minimum/test_search.py -q
```
