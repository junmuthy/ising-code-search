# Notebooks

[`ising_conditions.ipynb`](ising_conditions.ipynb) reconstructs the packed
`[[448,64,7]]` Ising/STAR code and checks its static conditions.

From the repository root:

```bash
.venv/bin/jupyter execute --inplace --timeout=300 notebooks/ising_conditions.ipynb
```

The setup cell locates the repository from either the root or `notebooks/`.
It uses the existing `gala_search` library and saved distance certificate under
`results/`. Execution requires that local certificate and the qLDPC environment.
Saved outputs are retained; moving the notebook does not regenerate them.
