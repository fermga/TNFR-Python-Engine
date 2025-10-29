"""Mathematics primitives aligned with TNFR coherence modeling."""

from .operators import CoherenceOperator, FrequencyOperator
from .spaces import BanachSpaceEPI, HilbertSpace
from .transforms import (
    IsometryFactory,
    build_isometry_factory,
    ensure_coherence_monotonicity,
    validate_norm_preservation,
)
from .validators import NFRValidator

__all__ = [
    "HilbertSpace",
    "BanachSpaceEPI",
    "CoherenceOperator",
    "FrequencyOperator",
    "NFRValidator",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "ensure_coherence_monotonicity",
]
