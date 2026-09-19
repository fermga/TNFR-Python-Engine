"""Experimental spectral factorization lab namespace.

``SpectralPaleyFactorizer`` / ``SpectralAnalysisResult`` expose the detailed
analysis; ``factorize`` / ``FactorizationResult`` provide the high-level
wrapper. Returned structural acceptance is heuristic, not a proof of
divisibility, completeness or autonomous nodal factor recovery."""

from .api import FactorizationResult, factorize
from .spectral_paley import SpectralAnalysisResult, SpectralPaleyFactorizer

__all__ = [
    "SpectralPaleyFactorizer",
    "SpectralAnalysisResult",
    "factorize",
    "FactorizationResult",
]
