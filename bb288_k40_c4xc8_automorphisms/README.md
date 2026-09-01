# Published `[[288,40,8]]` BB code: `C_4 x C_8` screen

This directory reconstructs the exact Campaign-4 verification record for the
`[[288,40,8]]` code of Cruz-Benito *et al.*,
[arXiv:2606.02418](https://arxiv.org/abs/2606.02418):

\[
A=1+x^2+y^4+x^2y^4,
\qquad
B=1+x^4+y^8+x^4y^2
\]

over \(\mathbb F_2[C_{16}\times C_9]\).  The authors' MILP record certifies
\([[288,40,8]]\), with weight-eight stabilizers.

The paper's displayed class-`e` generators are duplicated in its following
class-`f` row.  They reconstruct as `[[288,36,8]]`; that discrepancy is
preserved in `results/run_001`.  The verified machine-readable generators above
are used beginning with `results/run_002`.

The native \(x^2\) translation already has physical order eight.  The search
computes the full color-preserving automorphism group of the canonical Tanner
presentation and seeks a commuting logical order-four partner.  Because the
code has 40 logical qubits but the grid requires 32, the acceptance test is a
32-dimensional regular submodule rather than a basis of the entire logical
space.

The regular-submodule test is exact.  For commuting actions \(X^4=Y^8=1\), the
group algebra is

\[
\mathbb F_2[C_4\times C_8]
\cong
\mathbb F_2[u,v]/(u^4,v^8),
\qquad
u=X+1,\ v=Y+1.
\]

A regular cyclic submodule exists exactly when its socle operator
\(u^3v^7\) acts nontrivially.  This avoids relying on random orbit sampling.

Run from the search-repository root:

```bash
.venv/bin/python bb288_k40_c4xc8_automorphisms/analyze.py \
  --output-dir bb288_k40_c4xc8_automorphisms/results/run_002
```

The script refuses to overwrite output and emits periodic JSON checkpoints.
