"""Mathematics primitives aligned with TNFR coherence modeling."""

from .operators import CoherenceOperator, FrequencyOperator
from .spaces import BanachSpaceEPI, HilbertSpace

__all__ = [
    "HilbertSpace",
    "BanachSpaceEPI",
    "CoherenceOperator",
    "FrequencyOperator",
]
