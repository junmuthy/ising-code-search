# BB64 ideal syndrome-conditioned TMR recovery

This package implements the ideal branch-recovery analysis for the frozen preferred `[[64,8,8]]` code and its saved three-piece (`M=3`) logical-`Z` partitions. It does not alter the code basis, stabilizers, syndrome schedule, or earlier postselection results.

The analysis:

1. enumerates all `2^16 = 65,536` attainable TMR syndrome classes;
2. assigns each class a canonical partial-product correction and its 64-bit physical support;
3. derives the eight output logical angles and Pauli-frame metadata;
4. checks all `65,536 x 256 = 16,777,216` syndrome/logical amplitudes for product factorization;
5. implements a noiseless controller that either resets or repairs retained alternatives with deterministic in-code `M=1` rotations;
6. compares zero-syndrome STAR postselection, repair thresholds one through three, and full all-syndrome recovery.

Run the complete reproducible analysis from the repository root:

```bash
.venv/bin/python -m bb64_syndrome_recovery.run_analysis
```

The command reports progress during exhaustive work and refuses to replace nonempty results unless `--overwrite` is supplied. Outputs are written under `bb64_syndrome_recovery/results/ideal_recovery_pi32/`.

The main archive, `syndrome_classes.npz`, uses one row per syndrome class. `syndrome_coordinates` stores the 16 independent syndrome bits, while `displayed_syndromes` expands them to the 32 displayed X-check outcomes. `canonical_piece_masks` identifies the correction as a mask on the 24 saved TMR pieces and `correction_physical_supports` gives the corresponding mask on physical qubits `0` through `63`. `alternative_mask`, `selected_piece`, and `logical_angles_pi32` describe the eight logical outputs. `logical_z_frame` records which complete logical-`Z` triples relate the canonical correction to the deterministic independent-column representative; it is correction-gauge metadata, not an uncorrected output error.

Run the fast regression tests with:

```bash
.venv/bin/python -m unittest discover -s bb64_syndrome_recovery/tests -v
```

## Interpretation boundary

Every retained branch is repaired to all eight requested logical resource states; a threshold does not accept an incomplete resource block. The threshold decides only whether to reset or pay for repair.

This first implementation is noiseless. Its resource estimates include a conservative final syndrome round after `M=1` repair, but it does not yet simulate feed-forward faults. Circuit-level adaptive recovery is a separate next phase.
