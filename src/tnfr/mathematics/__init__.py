"""Mathematics primitives aligned with TNFR coherence modeling.

Backend selection
-----------------
Use :func:`get_backend` to retrieve a numerical backend compatible with TNFR's
structural operators.  The selection order is ``name`` → ``TNFR_MATH_BACKEND``
→ :func:`tnfr.config.get_flags`.  NumPy remains the canonical default so
existing code continues to operate even when optional dependencies are absent.
"""

from .backend import (
    MathematicsBackend,
    available_backends,
    ensure_array,
    ensure_numpy,
    get_backend,
    register_backend,
)
from .dynamics import ContractiveDynamicsEngine, MathematicalDynamicsEngine
from .epi import BEPIElement, CoherenceEvaluation, evaluate_coherence_transform
from .generators import build_delta_nfr, build_lindblad_delta_nfr
from .metrics import dcoh
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
    CoherenceMonotonicityReport,
    CoherenceViolation,
    IsometryFactory,
    build_isometry_factory,
    ensure_coherence_monotonicity,
    validate_norm_preservation,
)

__all__ = [
    "MathematicsBackend",
    "ensure_array",
    "ensure_numpy",
    "HilbertSpace",
    "BanachSpaceEPI",
    "BEPIElement",
    "CoherenceEvaluation",
    "CoherenceOperator",
    "ContractiveDynamicsEngine",
    "CoherenceMonotonicityReport",
    "CoherenceViolation",
    "FrequencyOperator",
    "MathematicalDynamicsEngine",
    "build_delta_nfr",
    "build_lindblad_delta_nfr",
    "make_coherence_operator",
    "make_frequency_operator",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "ensure_coherence_monotonicity",
    "evaluate_coherence_transform",
    "StateProjector",
    "BasicStateProjector",
    "normalized",
    "coherence",
    "frequency_positive",
    "stable_unitary",
    "dcoh",
    "coherence_expectation",
    "frequency_expectation",
    "available_backends",
    "get_backend",
    "register_backend",
]
