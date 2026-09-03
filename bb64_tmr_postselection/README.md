# BB64 TMR postselection simulation

This package measures the acceptance cost of preparing small-angle rotation
resource states in the preferred two-batch `[[64,8,8]]` BB code.  It stops
after post-TMR syndrome extraction.  Logical teleportation, decoding, and
fidelity estimation are deliberately deferred.

## Frozen inputs

The simulation loads the preferred basis and syndrome schedule without
regenerating them:

- basis SHA-256:
  `8cbf96d32e5e16b357c2a4b751a94f09b3e136d8d0d722226a267be3395ad03e`;
- schedule SHA-256:
  `0d07ff49c000900b65928fd5f26cb463315448ffbf2b42c4b822ba3d68fc656c`;
- Z-logical batches: `{0,3,4,7}` and `{1,2,5,6}`.

`partitions_m3.json` records a balanced `2+3+3` partition of every weight-eight
logical Z representative.  Its partial-operator syndrome ranks are `8,8` for
the two batches and `16` jointly.  Thus it certifies both the conservative
two-projection protocol and the more economical protocol that applies the two
TMR batches before one final syndrome projection.

## Circuit modes

- `ideal-projection`: exact encoded plus state, direct multi-Pauli rotations,
  and ideal MPP stabilizer projection;
- `scheduled-tmr-only`: exact encoded plus state, noisy CNOT-ladder TMR, and
  the physical eight-layer syndrome schedule;
- `full`: transversal physical plus initialization, one Z-only initialization
  round, a virtual syndrome-dependent X-frame correction, noisy ladder TMR,
  and scheduled post-TMR extraction.

The initial Z syndrome is used for the frame update and is not postselected.
All 32 displayed X and 32 displayed Z outcomes of every required post-TMR
round are postselection conditions.

For `M=3`, all eight logicals require 80 TMR CNOTs and 24 physical rotations.
The full single-final-check circuit contains 848 CNOTs in 24 entangling layers;
the two-projection circuit contains 1,360 CNOTs in 32 entangling layers.

## Environment

Use the installed ClifT environment:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m unittest discover -s bb64_tmr_postselection/tests -v
```

Stim `1.16.0` is installed in that environment for exact stabilizer-state
synthesis and Clifford cross-checks.

Generate the stored partition certificate with:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_tmr_postselection.select_partitions \
  --output bb64_tmr_postselection/partitions_m3.json
```

Run a small ideal check:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_tmr_postselection.run_postselection \
  --mode ideal-projection \
  --protocol single-final-check \
  --theta 0.09817477042468103 \
  --logical-count 8 \
  --max-shots 100000 \
  --checkpoint-shots 10000 \
  --output bb64_tmr_postselection/results/ideal-n8-pi32.json
```

Run a circuit-level point at `p=10^-3` with:

```bash
/home/judah_unmuth/Documents/multistaq/star-simulators/.venv-clifft/bin/python \
  -m bb64_tmr_postselection.run_postselection \
  --mode full \
  --protocol single-final-check \
  --theta 0.09817477042468103 \
  --logical-count 8 \
  --probability 0.001 \
  --threads 16 \
  --clifft-batch-size 1 \
  --target-survivors 400 \
  --max-shots 1000000 \
  --checkpoint-shots 1000 \
  --output bb64_tmr_postselection/results/full-single-n8-pi32-p1e-3.json
```

The runner refuses to overwrite results.  Pass `--resume` with exactly the
same scientific and execution configuration to continue an interrupted run.
Each checkpoint reports attempts, survivors, acceptance, accepted candidate
resources per attempt, sampling rate, and timing.

The initial noisy `N=8`, `theta=pi/32`, `p=10^-3` benchmark runs at about
38 attempts/s for the one-final-check protocol on this host.  Its observed
acceptance was about one percent, implying approximately 17 minutes for 400
survivors and 1.8 hours for 2,500 survivors.  The first 400-survivor run took
17.7 minutes and measured `0.009780 ± 0.000486` block acceptance.  The
conservative two-projection
protocol simulates far faster because early postselection keeps the ClifT
active state narrower, although it contains more physical gates and has lower
acceptance.

Use `--diagnostic` for ordinary sampling that retains aggregate syndrome-weight
and detector-group statistics.  Survivor mode is faster because ClifT aborts a
trajectory as soon as a postselection detector fires.

## Reported quantities

For `N` rotated logicals and measured block acceptance `s`, the primary yield is

\[
Y_{\mathrm{candidate}}=Ns.
\]

These are accepted candidate resource states, not fidelity-certified states.
The later teleportation study will determine per-logical and collective
infidelity on the surviving blocks.
