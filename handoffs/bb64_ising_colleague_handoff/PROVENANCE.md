# Provenance and evidence boundary

## Published source

The code definition and exact `[[64,8,8]]` parameters are attributed to:

Zijian Liang and Yu-An Chen, “Self-dual bivariate bicycle codes with
transversal Clifford gates,” arXiv:2510.05211v2 (2026), Table 1, Figure 1, and
Appendix A.

Paper landing page: <https://arxiv.org/abs/2510.05211>

The paper itself is not redistributed in this archive.

## Local reconstruction lineage

The raw matrices came from the preserved local result

`results/batched-c4xc2-search/published-selfdual-bb64-grid-presentations-260901-v2/raw-disjoint-grid.npz`

in the `gala-code-search` working tree at local Git HEAD
`f5f5778c85e0d8e8c006abc0d5fec14ccdb4d7c1`.  The relevant search directory
was untracked at packaging time, so the copied scripts and result JSON files in
this archive—not that commit alone—are the reproducibility record.

The original raw NPZ and summaries are copied under `provenance/`; the local
analysis programs are copied under `source_analysis_scripts/`.

## Hadamard basis derivation

The raw same-support ZX pairing is a nonsingular alternating `8 x 8` matrix
`P`.  The packaging script performs GF(2) symplectic Gram–Schmidt to construct
an invertible matrix `S` such that

\[
SPS^T=J_2\oplus J_2\oplus J_2\oplus J_2,
\qquad
J_2=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
\]

This gives a canonical logical basis in which transversal H is logical H plus
four pairwise swaps.  The derivation is reproducible in
`tools/build_package.py` and checked independently by `verify_package.py`.

The resulting Z and X support weights are

\[
(8,8,8,8,16,16,16,16).
\]

## Distance evidence

This package verifies an explicit nontrivial weight-eight logical operator, so
it independently proves `d <= 8`.  The lower bound `d >= 8` is taken from the
published exact-distance result.  The archive does not contain a standalone
MILP certificate for exclusion of weights one through seven.
