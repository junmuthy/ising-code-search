# Existing-ideal `C56 x C4` weight-16 extension

This focused extension raises the physical check-weight ceiling from `12` to
`16` by exhaustively adding normalized four-term, zero-augmentation quotient
polynomials over `GF(8)[C8 x C4]`.  Each quotient term inflates to a binary
weight-four simplex word in the `C7` thickness sector.

The screen uses the new polynomials as possible check generators for the 731
ideals whose distances were already certified in
`exact-selfdual-monomial-sparse-w12-v1`.  It does not add four-term principal
ideals as new code candidates.

## Catalog

- Existing normalized polynomials of quotient weight 1--3: `948`
- New normalized quotient-weight-four polynomials: `48,403`
- Total generator catalog: `49,351`
- Existing code ideals screened: `731`

## Results

Thirty-seven ideals have complete reciprocal-sector presentations with maximum
check weight at most 16:

| Maximum check weight | Ideals |
| ---: | ---: |
| `4` | `2` |
| `8` | `3` |
| `12` | `6` |
| `16` | `26` |

Of the 26 new weight-16 presentations, two have distance at least six.  Both
are exact `[[224,32,7]]` codes:

- `ideal-9b156f440c172fe3`
- `ideal-f5a03f497a332841`

They are the reciprocal ideal/annihilator pair of one another.  Each side is
generated entirely by physical weight-16 checks.

Neither candidate has a connected Tanner graph.  This is not an artifact of
the greedy presentation: an additional audit included every catalog check of
weight at most 16 contained in either sector (two eligible generators in the
14-dimensional sector and four in the 18-dimensional sector), and the graph
remained disconnected.  Thus the count of connected `d >= 6` results in this
focused extension is zero.

## Interpretation

Raising the ceiling to 16 does cross the algebraic presentation barrier for a
small reciprocal pair of distance-seven ideals.  It does not yet produce a
connected Ising candidate among the previously certified 731 ideals.  A truly
complete weight-16 code-family expansion would additionally introduce all
four-term principal ideals as new code candidates and certify their distances.
