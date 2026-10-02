# Syndrome schedules for the folded `[[32,4,6]]` code

## Outcome

Yes: the code has a translation-symmetric stabilizer presentation and a safe
ZX-fold-symmetric syndrome schedule.  There are three useful operating points.

| Presentation | Ancillas | Check weights per CSS type | CNOTs | CNOT depth | `C4`-symmetric | Even syndrome parity |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Minimum independent basis | `28` | one `4`; thirteen `8` | `216` | `16` | no | no |
| STAR-compatible translated basis | `30` | one `4`; twelve `8`; two `10` | `240` | `20` | checks | yes |
| Strong temporal covariance | `30` | one `4`; twelve `8`; two `10` | `240` | `24` | checks and time layers | yes |

Preparation and measurement layers are not included in the tabulated CNOT
depths.

## Translation-symmetric STAR presentation

The rank-14 `X` stabilizer space decomposes into translated module orbits of
ranks

\[
(4,4,4,2).
\]

Taking the distinct rows in those orbits gives 14 independent checks: twelve
of weight eight and two of weight ten.  Their XOR is the redundant weight-four
check

\[
\{7,15,23,31\}.
\]

This is exactly the spectator fibre.  It is fixed by both the physical
translation `T` and the ZX fold `P`.  Measuring it in addition to the 14
independent checks makes every data-qubit degree even.  Per CSS type, eight
data qubits have degree two, twenty have degree four, and four have degree six.
Therefore every syndrome has even total parity, retaining the structural
one-round STAR postselection protection against a single measurement bit flip.

The translated check set is closed under `T`.  Its check-orbit sizes are

\[
(4,4,4,2,1),
\]

where the final orbit is the redundant spectator-fibre check.

## Safe 20-layer schedule

For one CSS type, the largest check has weight ten and the largest data degree
is six.  The Tanner graph is bipartite, and an exact edge coloring attains the
degree lower bound of ten CNOT layers.  The saved safe cycle executes:

1. the ten `Z` layers;
2. the ten `X` layers.

All cross-type check overlaps are even, so putting one CSS type entirely before
the other cancels cross-ancilla back-action.  The `Z` schedule is the
`P`-image and time reverse of the `X` schedule.  Thus the full circuit is
invariant under exchanging `X` and `Z`, applying `P`, and reversing time.

This is the recommended current schedule for STAR work: it is
translation-symmetric at the check-presentation level, fold-symmetric, has even
syndrome parity, and is cheaper than the stronger temporal-covariance option.

## Strong 24-layer schedule

The stronger certificate also makes physical translation act uniformly on
time.  Within each 12-layer CSS half, `T` permutes the layers as

\[
(0\ 1\ 2\ 3)(4\ 5\ 6\ 7)(8\ 9\ 10\ 11).
\]

The fold exchanges the two CSS halves and reverses all 24 layers.  The layer
actions therefore satisfy the same dihedral relation as the physical actions,

\[
P T P^{-1}=T^{-1}.
\]

An exact finite-domain search excluded 10- and 11-layer one-type schedules in
which `T` acts through a single global permutation of the time layers.  The
stored 12-layer certificate is therefore optimal within this strong temporal
covariance condition.

## What is and is not certified

Both saved full schedules were checked for:

- exact coverage of every check-data Tanner edge;
- no data or ancilla collision within a layer;
- CSS orthogonality;
- cancellation of cross-ancilla back-action;
- exchange under `P` plus time reversal;
- the stated translation action;
- rank 14 per CSS type and even syndrome parity for the STAR presentations.

This is not yet a circuit-level fault-distance certificate.  A single bare
ancilla fault can propagate to several data qubits, so the order within a
collision-free schedule must still be optimized against hook errors.  The
bicycle-chain STAR work followed exactly this stronger procedure: symbolically
filter clean simultaneous schedules and then rank them by circuit fault
distance; for its distance-six, weight-eight code, the selected depth-eight
schedule has circuit fault distance five.  The same Stim/symbolic fault search
is the next required step for this code.

The existence of a collision-free simultaneous 12-layer edge coloring alone
does not certify a syndrome circuit.  In the restricted translation- and
fold-covariant simultaneous ansatz tested here, the short colorings left
cross-ancilla back-action.  A broader simultaneous search remains open.

## Files

- `analyze_schedule.py`: reconstructs the checks and verifies both schedules.
- `minimum-basis-fold-symmetric-16-layer.json`: smallest saved safe cycle.
- `safe-fold-symmetric-20-layer.json`: recommended STAR-compatible cycle.
- `strong-c4-fold-symmetric-24-layer.json`: strong temporal-covariance cycle.
- `schedule-analysis.json`: concise machine-readable summary.

## Literature comparison

The scheduling standard used here follows the distinction made in the
[bicycle-chain STAR paper](https://arxiv.org/abs/2606.25011): Tanner-edge depth
is only the first filter, while clean ancilla propagation and circuit fault
distance require explicit circuit analysis.  The physical translation and
ZX-fold symmetries are interpreted consistently with the
[GALA construction](https://arxiv.org/abs/2608.07431).
