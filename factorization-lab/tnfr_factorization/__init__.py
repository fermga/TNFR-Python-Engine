"""TNFR Spectral Factorization namespace.

Public API:
    - SpectralPaleyFactorizer / SpectralAnalysisResult (low-level analysis)
    - factorize / FactorizationResult (high-level nodal factorization)
"""

from .api import FactorizationResult, factorize
from .spectral_paley import SpectralAnalysisResult, SpectralPaleyFactorizer

__all__ = [
    "SpectralPaleyFactorizer",
    "SpectralAnalysisResult",
    "factorize",
    "FactorizationResult",
]
