# Bounded `[[30,4,>=6]]` mixed-orbit screen

The `n=30` solver uses seven regular `C4` fibres and one physical `C2` orbit.
It was deliberately stopped after four completed tasks when its timing showed
that a broad continuation would be less efficient than seed-guided `n=32`
refinement.

The four completed tasks tested 18 exact-distance models:

| Exact distance | Models |
| --- | ---: |
| `2` | 4 |
| `3` | 4 |
| `4` | 8 |
| `5` | 2 |
| `>=6` | 0 |

One distance-five model was Tanner-connected.  The other distance-five model
was disconnected.  The completed task statuses were one model-limit task and
three solver-unknown tasks; none is an exact negative result.

The reflected fifth task had only just begun when the run was interrupted and
is excluded from the table.  Its partial files remain on disk, but it has no
completed task summary and must not be counted.

This is a bounded timing decision, not a no-go result for `n=30`.
