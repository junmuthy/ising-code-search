# Paper bicycle-chain `ell=2, m=7` specialization

This directory reconstructs the literal `ell=2`, `m=7` specialization of the
bicycle-chain family in Appendix B of Ismail et al., *Fast and Parallel
High-Rate STAR Architecture for Megaquop Quantum Simulation* (2026),
arXiv:2606.25011v1.

The paper fixes

\[
a(x,y)=1+y^3+xy^2+xy^4,\qquad b(x,y)=a^\dagger(x,y),
\]

over

\[
\mathbb F_2[x,y]/(x^2-1,y^7-1),
\]

and uses

\[
H_X=[A\mid B],\qquad H_Z=[B^T\mid A^T].
\]

Because `B=A.T`, `H_X=H_Z`.  Exact enumeration and `qldpc` independently
show that this specialization is `[[28,4,5]]`, rather than the `[[28,4,6]]`
implied by the paper's family label `[[14*ell,2*ell,6]]`.  The analysis here
always records the verified distance five.

## Frozen structure

- 28 data qubits, labeled `L(0,0)..L(1,6), R(0,0)..R(1,6)` as integers
  `0..27`;
- 14 displayed X checks and 14 displayed Z checks, each of weight eight;
- rank 12 per check type, with two displayed-check dependencies;
- four disjoint weight-seven logical fibres, with identical X and Z support;
- exact transversal H;
- physical x translation acting logically as `(0 1)(2 3)`;
- Appendix-B Table-II syndrome schedule, with 28 ancillas and eight CNOT
  layers.

The frozen arrays and human-readable presentation are in `code_data/`.
The exact paper schedule is `schedule/paper_table_ii_schedule.json`.

## Results

The Table-II schedule is exactly certified at

\[
d_{\mathrm{fault}}=5
\]

in both X and Z bases, including five full noisy syndrome rounds.  See
`stim_fault_distance/RESULTS.md`.

For four simultaneous `M=3` TMR preparations at `theta=pi/32` and
`p=10^-3`, one million full-circuit attempts gave

\[
p_{\mathrm{accept}}=0.123622\pm0.000329,
\]

or 8.089 attempts per accepted four-resource candidate block.  See
`tmr_postselection/RESULTS.md`.  This is an acceptance result only; no
teleportation or conditional logical-fidelity claim is made here.

## Reproduction

Run the regression tests with the installed ClifT environment:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m unittest discover -s bicycle_chain_l2_paper/tests -v
```

Run exact fault certification with the GALA virtual environment:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache \
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/gala-code-search/.venv/bin/python \
  -m bicycle_chain_l2_paper.stim_fault_distance.run_validation \
  --memory-rounds 5 \
  --output bicycle_chain_l2_paper/stim_fault_distance/results/paper_schedule_r5.json
```

Raw result JSON and generated Stim circuits live in ignored `results/`
directories.  The compact Markdown, CSV, SVG, schedule, matrix, and
certificate files are version controlled.
