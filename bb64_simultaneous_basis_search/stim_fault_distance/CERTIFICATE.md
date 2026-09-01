# BB64 circuit-fault certificate

The selected depth-eight simultaneous schedule for the Liang--Chen
`[[64,8,8]]` code has exact three-round Clifford memory-circuit fault distance

\[
d_{\mathrm{fault}}^X=d_{\mathrm{fault}}^Z=6.
\]

This statement applies to the `full` elementary Stim noise model implemented
in `circuit.py`: noisy preparation, measurement, CNOTs, and idle locations at
probability `0.001`. The schedule is `baseline_schedule.json`, ID
`67e327dc7cb35335`.

## Lower bound

For each basis, exact enumeration excludes one-, two-, and three-fault logical
mechanisms. A sorted pair meet-in-the-middle table excludes four faults. The
five-fault certificate uses every merged Stim circuit location when constructing
the cell-zero logical anchors; this gives 262 complete anchors rather than only
one representative per detector-error signature.

For each of the 262 anchors, all 51,770,400 unordered pairs of the 10,176
distinct elementary effects are grouped by syndrome. The pair-plus-pair search
checks all 50,913,624 pair-syndrome groups, requires five distinct effect
indices, and finds no undetected nontrivial logical in either basis. Thus

\[
d_{\mathrm{fault}}^X\geq 6,
\qquad
d_{\mathrm{fault}}^Z\geq 6.
\]

## Upper bound

An independent tightly bounded Stim search finds explicit six-mechanism
undetected logical failures in both bases, stored in
`certificates/memory-r3-full-upper-bound6.json`. Hence

\[
d_{\mathrm{fault}}^X\leq 6,
\qquad
d_{\mathrm{fault}}^Z\leq 6.
\]

The matching bounds establish the exact value. No search above the requested
threshold was performed.

## Tracked evidence

- `certificates/memory-r3-full-X-direct4.json`: exact exclusion through four
  faults in the `X` basis.
- `certificates/memory-r3-full-Z-direct4.json`: exact exclusion through four
  faults in the `Z` basis.
- `certificates/memory-r3-full-X-exact5.json`: complete five-fault exclusion in
  the `X` basis.
- `certificates/memory-r3-full-Z-exact5.json`: complete five-fault exclusion in
  the `Z` basis.
- `certificates/memory-r3-full-upper-bound6.json`: concrete six-fault witnesses
  for both bases.

Raw checkpoints, complete schedule-search trees, and multi-gigabyte temporary
pair tables remain outside version control.
