# Paper bicycle-chain `ell=4, m=7` reproduction

This directory freezes the `[[56,8,6]]` bicycle-chain code used by Ismail et
al., *Fast and Parallel High-Rate STAR Architecture for Megaquop Quantum
Simulation* (2026), [arXiv:2606.25011](https://arxiv.org/abs/2606.25011).

The code is defined over

\[
\mathbb F_2[x,y]/(x^4-1,y^7-1)
\]

by

\[
a(x,y)=1+y^3+xy^2+xy^4,\qquad b(x,y)=a^\dagger(x,y),
\]

with

\[
H_X=[A\mid B],\qquad H_Z=[B^T\mid A^T].
\]

It has 28 displayed checks of each type, rank 24 per type, eight disjoint
weight-seven logical fibres, exact `H_X=H_Z`, weight-eight checks, and two
logical four-cycles under physical `x` translation. An independent HiGHS
certificate finds distance six in a translation representative from each
physical half.

The code matrices and logical basis are frozen in `code_data/`; the exact
depth-eight schedule from Appendix-B Table II is frozen in `schedule/`.

## Post-selection cross-check

The principal reproduction point is the one used explicitly in Fig. 17:

\[
\ell=4,\quad m=7,\quad N=1,\quad M=3,\quad
\theta=\pi/32,\quad p=10^{-3},\quad f=0.
\]

One million ideal attempts give

\[
s_{\rm ideal}=0.687142\pm0.000464,
\]

in agreement with the exact value `0.6871495483`. One million full noisy
attempts, post-selecting on the TMR-sensitive X-check projection, give

\[
s_{\rm sim}=0.294159\pm0.000456.
\]

The paper's Eqs. (C6)--(C7), with `gamma=0.93`, predict

\[
p_{\rm init}=e^{-0.93(32\ell m+13)p}=0.42939845,
\]

and hence

\[
s_{\rm paper}=p_{\rm init}p_{\rm TMR}=0.29506095.
\]

The simulated value is 0.31% below the fit, a difference of 1.98 simulation
standard errors. This is a successful numerical cross-check of the published
acceptance model.

`tmr_postselection/PAPER_CROSSCHECK.md` records the full comparison and an
important acceptance-convention caveat. Raw JSON and Stim circuits are kept
locally under the ignored `results/` directory.

The broader acceptance suite is driven by the manifests in
`tmr_postselection/manifests/`. It compares `N in {1,2,4,8}`, covers the
small-angle Trotter regime, and reports both the paper-matching `tmr-x`
acceptance and the stricter `strict-xz` convention used for BB64.

## Reproduce

Run the tests:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m unittest discover -s bicycle_chain_l4_paper/tests -v
```

Run the paper point with periodic 100,000-shot checkpoints:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bicycle_chain_l4_paper.tmr_postselection.run_postselection \
  --mode full --postselection-policy tmr-x \
  --theta 0.09817477042468103 --partition-count 3 --logical-count 1 \
  --probability 0.001 --max-shots 1000000 --checkpoint-shots 100000 \
  --threads 16 --clifft-batch-size 1 --diagnostic \
  --output bicycle_chain_l4_paper/tmr_postselection/results/reproduction.json
```

Run a resumable suite manifest and regenerate its committed report with:

```bash
PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bicycle_chain_l4_paper.tmr_postselection.run_sweep \
  --manifest bicycle_chain_l4_paper/tmr_postselection/manifests/pilot_small_angles.json \
  --output-dir bicycle_chain_l4_paper/tmr_postselection/results/pilot_small_angles

PYTHONPATH=/home/judah_unmuth/gala-code-search \
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bicycle_chain_l4_paper.tmr_postselection.analyze_suite
```
