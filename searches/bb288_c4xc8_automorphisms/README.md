# Published `[[288,32,8]]` BB code: `C_4 x C_8` automorphism screen

This directory reconstructs the exact-distance code reported as class `j` in
Table 2 of Cruz-Benito *et al.*, [arXiv:2606.02418](https://arxiv.org/abs/2606.02418):

\[
A=1+xy^3+x^5y^3,\qquad
B=1+x^3y+x^3y^5
\]

over \(\mathbb F_2[C_{12}\times C_{12}]\).  The published parameters are
\([[288,32,8]]\), and the stabilizer weight is six.

`analyze.py` performs four checks:

1. reconstructs `H_X=[A|B]` and `H_Z=[B^T|A^T]`;
2. verifies rank, CSS commutation, check weight, connectivity, native shifts,
   and the standard BB ZX fold;
3. computes the full color-preserving automorphism group of the canonical
   translated-check Tanner graph with nauty;
4. screens every automorphism, when the group is small enough, for commuting
   logical actions of orders four and eight that form a regular 32-dimensional
   `C_4 x C_8` module.

The Tanner colors distinguish data, X-check, and Z-check vertices.  The group
therefore contains CSS-preserving physical permutations.  The ZX duality is
checked separately because it exchanges X and Z checks.

Run from the search repository root:

```bash
.venv/bin/python searches/bb288_c4xc8_automorphisms/analyze.py \
  --output-dir searches/bb288_c4xc8_automorphisms/results/run_001
```

The script refuses to overwrite an existing output directory and prints JSON
checkpoints during every potentially long stage.

If the full Tanner graph is disconnected, follow with the exact componentwise
factorization:

```bash
.venv/bin/python searches/bb288_c4xc8_automorphisms/analyze_components.py \
  --output-dir searches/bb288_c4xc8_automorphisms/results/components_001
```

This computes a connected component's complete automorphism group and tests a
twisted component-cycle construction of the requested logical grid.

For a complete search of the cyclic-order-four top action in the full wreath
product, run:

```bash
.venv/bin/python searches/bb288_c4xc8_automorphisms/search_wreath_c4_top.py \
  --output-dir searches/bb288_c4xc8_automorphisms/results/wreath_c4_top_001
```

The only other transitive abelian action on four components has Klein-four
top action.  Complete that case with:

```bash
.venv/bin/python searches/bb288_c4xc8_automorphisms/search_wreath_v4_top.py \
  --output-dir searches/bb288_c4xc8_automorphisms/results/wreath_v4_top_001
```
