"""Batched ``C4 x C2`` logical-grid search for compact GALA codes.

The package is intentionally independent of the older full-half-grid searches.
Its public surface is small so that saved candidates can be reconstructed and
rechecked without depending on a particular search run.
"""

from .model import (
    GRID_ORDER,
    Candidate,
    Monomial,
    analyze_structure,
    build_code,
    quotient_s3_seed,
)
from .halfswap_model import (
    HalfSwapCandidate,
    analyze_half_swap_structure,
    build_half_swap_code,
)
from .twisted_halfswap_model import (
    TwistedHalfSwapCandidate,
    analyze_twisted_half_swap_structure,
    build_twisted_half_swap_code,
)

__all__ = [
    "GRID_ORDER",
    "Candidate",
    "Monomial",
    "analyze_structure",
    "build_code",
    "quotient_s3_seed",
    "HalfSwapCandidate",
    "analyze_half_swap_structure",
    "build_half_swap_code",
    "TwistedHalfSwapCandidate",
    "analyze_twisted_half_swap_structure",
    "build_twisted_half_swap_code",
]
