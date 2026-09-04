# BB64 ideal syndrome-conditioned TMR recovery

This report studies the frozen preferred `[[64,8,8]]` basis and its saved `M=3` partitions. It is a noiseless branch-algebra and controller analysis; it does not claim circuit-level fault tolerance for the adaptive repair circuit.

## Algebraic certificate

- Partial-product generators: `24` (`3` per logical qubit).
- Displayed X-check outcomes: `32`; attainable syndrome rank: `16`.
- Enumerated syndrome classes: `65,536`.
- Kernel dimension: `8`; it is generated exactly by the eight full logical-`Z` triples.
- Exhaustively compared amplitudes: `16,777,216`.
- Maximum factorization error: `0.000e+00`.
- Independent product rotations after canonical correction: `True`.

Each syndrome has one canonical correction containing zero or one saved TMR piece per logical qubit. A nonzero local branch selects one of three pieces, so the syndrome labels are exactly the `4^8 = 65,536` strings over `{0,1,2,3}`. The correction table and all 64-bit physical supports are stored in `syndrome_classes.npz`.

## Local branches at theta = pi/32

| Branch | Probability per pattern | Multiplicity | Logical angle | Angle-frame Z |
|---|---:|---:|---:|---:|
| target | 0.68715 | 1 | 0.0981748 | `0` |
| alternative piece | 0.104283 | 3 | -0.702148 | `0` |

A retained alternative is repaired deterministically with an in-code `M=1` logical `R_Z(theta - alpha)` rotation. All eight logical resource states are correct at output; the threshold only decides whether an expensive branch is reset or repaired.

## Policy comparison at theta = pi/32

| Threshold | Keep/attempt | Reset/attempt | Initial attempts | Repair given kept | Bad logicals given kept | TMR stages | CNOT layers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0497061 | 0.950294 | 20.1182 | 0 | 0 | 20.1182 | 482.838 |
| 1 | 0.230751 | 0.769249 | 4.33368 | 0.784589 | 0.784589 | 5.11827 | 121.269 |
| 2 | 0.519246 | 0.480754 | 1.92587 | 0.904273 | 1.45988 | 2.83014 | 70.5597 |
| 3 | 0.781943 | 0.218057 | 1.27887 | 0.936433 | 1.97729 | 2.2153 | 58.2773 |
| 8 | 1 | 2.22045e-16 | 1 | 0.950294 | 2.5028 | 1.95029 | 53.3598 |

The analytic moments were independently cross-checked with `1,000,000` Bernoulli samples per angle and threshold. Standard errors are in `policy_comparison.csv` and the full records are in `policy_comparison.json`.

Resource accounting uses the saved BB64 schedule. One initial full `M=3` attempt has `848` CNOTs, `24` CNOT layers, `24` physical rotations, and two syndrome cycles including initialization. A retained repair adds `14` CNOTs per bad logical, `14` CNOT layers per nonempty disjoint-support batch, and conservatively one final 512-CNOT/eight-layer syndrome round.

## Scope

This establishes ideal branch recoverability and compares first-stage reset policies. The next circuit-level phase must synthesize feed-forward, propagate physical noise through correction, and re-evaluate acceptance and logical error. No noisy recovery claim is made here.
