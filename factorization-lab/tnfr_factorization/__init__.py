"""TNFR Spectral Factorization namespace.

Public API:
    - SpectralPaleyFactorizer / SpectralAnalysisResult (low-level analysis)
    - factorize / FactorizationResult (high-level nodal factorization)
"""

from .spectral_paley import SpectralPaleyFactorizer, SpectralAnalysisResult
from .api import factorize, FactorizationResult

__all__ = [
    "SpectralPaleyFactorizer",
    "SpectralAnalysisResult",
    "factorize",
    "FactorizationResult",
]
