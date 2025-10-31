"""Mathematics primitives aligned with TNFR coherence modeling."""

from .dynamics import MathematicalDynamicsEngine
from .epi import BEPIElement
from .generators import build_delta_nfr
from .operators import CoherenceOperator, FrequencyOperator
from .operators_factory import make_coherence_operator, make_frequency_operator
from .projection import BasicStateProjector, StateProjector
from .runtime import (
    coherence,
    coherence_expectation,
    frequency_expectation,
    frequency_positive,
    normalized,
    stable_unitary,
)
from .spaces import BanachSpaceEPI, HilbertSpace
from .transforms import (
    IsometryFactory,
    build_isometry_factory,
    ensure_coherence_monotonicity,
    validate_norm_preservation,
)
from ..validation import NFRValidator

__all__ = [
    "HilbertSpace",
    "BanachSpaceEPI",
    "BEPIElement",
    "CoherenceOperator",
    "FrequencyOperator",
    "MathematicalDynamicsEngine",
    "build_delta_nfr",
    "make_coherence_operator",
    "make_frequency_operator",
    "NFRValidator",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "ensure_coherence_monotonicity",
    "StateProjector",
    "BasicStateProjector",
    "normalized",
    "coherence",
    "frequency_positive",
    "stable_unitary",
    "coherence_expectation",
    "frequency_expectation",
]
