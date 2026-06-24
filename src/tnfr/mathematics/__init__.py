"""Mathematics primitives aligned with TNFR coherence modeling.

Backend selection
-----------------
Use :func:`get_backend` to retrieve a numerical backend compatible with TNFR's
structural operators.  The selection order is ``name`` → ``TNFR_MATH_BACKEND``
→ :func:`tnfr.config.get_flags`.  NumPy remains the canonical default so
existing code continues to operate even when optional dependencies are absent.

Symbolic Analysis
-----------------
This module also includes symbolic mathematical tools from the tnfr.math module
for analyzing TNFR dynamics, including nodal equation derivations and
convergence analysis.
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
from .liouville import (
    compute_liouvillian_spectrum,
    get_liouvillian_spectrum,
    get_slow_relaxation_mode,
    store_liouvillian_spectrum,
)
from .metrics import dcoh
from .number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRNetwork,
    PrimeCertificate,
    arithmetic_cayley_digraph,
    power_residue_rank,
    power_residue_set,
    quadratic_residue_annotated_rank,
    quadratic_residue_set,
    residue_network_rank,
    run_basic_validation,
    unitary_residue_set,
)
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
from .unified_cache import (
    CacheLevel,
    CacheStats,
    TNFRUnifiedCacheSystem,
    UnifiedLRUCache,
    cache_tnfr_computation,
    clear_unified_caches,
    get_cache_region,
    get_unified_cache_system,
)

# Unified numerical and cache systems
from .unified_numerical import (
    CONSTANTS,
    GAMMA,
    NUMPY_AVAILABLE,
    PHI,
    PI,
    ArrayLike,
    ComplexArray,
    E,
    TNFRConstants,
    TNFRNumericalUtilities,
    clamp_value,
    compute_circular_mean,
    compute_phase_difference,
    generate_random_array,
    get_unified_numerical_utils,
    is_finite_array,
    kahan_sum_nd,
    normalize_phase,
    np,
    npt,
    reset_global_seed,
    safe_divide,
)

"""Symbolic analysis exports
We import from tnfr.math.symbolic and bind the names so lint won't flag them as unused.
"""
try:
    from .. import math as _math
    from ..math import symbolic as _symbolic

    get_nodal_equation = _symbolic.get_nodal_equation
    solve_nodal_equation_constant_params = (
        _symbolic.solve_nodal_equation_constant_params
    )
    integrated_evolution_symbolic = _symbolic.integrated_evolution_symbolic
    check_convergence_exponential = _symbolic.check_convergence_exponential
    compute_second_derivative_symbolic = _symbolic.compute_second_derivative_symbolic
    evaluate_bifurcation_risk = _symbolic.evaluate_bifurcation_risk
    latex_export = _symbolic.latex_export
    pretty_print = _symbolic.pretty_print
    # Re-export the math module under 'math'
    math = _math
    _HAS_SYMBOLIC = True
except Exception:
    _HAS_SYMBOLIC = False

__all__ = [
    # Backend operations
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
    "compute_liouvillian_spectrum",
    "get_liouvillian_spectrum",
    "get_slow_relaxation_mode",
    "store_liouvillian_spectrum",
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
    # Unified numerical and cache systems
    "TNFRConstants",
    "CONSTANTS",
    "TNFRNumericalUtilities",
    "get_unified_numerical_utils",
    "normalize_phase",
    "compute_phase_difference",
    "generate_random_array",
    "safe_divide",
    "compute_circular_mean",
    "is_finite_array",
    "clamp_value",
    "kahan_sum_nd",
    "reset_global_seed",
    "np",
    "npt",
    "NUMPY_AVAILABLE",
    "ArrayLike",
    "ComplexArray",
    "PHI",
    "GAMMA",
    "PI",
    "E",
    "TNFRUnifiedCacheSystem",
    "get_unified_cache_system",
    "get_cache_region",
    "clear_unified_caches",
    "UnifiedLRUCache",
    "CacheStats",
    "CacheLevel",
    "cache_tnfr_computation",
    # Number theory (prime emergence)
    "ArithmeticTNFRFormalism",
    "ArithmeticStructuralTerms",
    "ArithmeticTNFRNetwork",
    "PrimeCertificate",
    "run_basic_validation",
    # Arithmetic residue networks (structural-frequency rank, cyclotomy)
    "quadratic_residue_set",
    "power_residue_set",
    "unitary_residue_set",
    "arithmetic_cayley_digraph",
    "residue_network_rank",
    "power_residue_rank",
    "quadratic_residue_annotated_rank",
]

# Add symbolic analysis functions if available
if _HAS_SYMBOLIC:
    __all__.extend(
        [
            "get_nodal_equation",
            "solve_nodal_equation_constant_params",
            "integrated_evolution_symbolic",
            "check_convergence_exponential",
            "compute_second_derivative_symbolic",
            "evaluate_bifurcation_risk",
            "latex_export",
            "pretty_print",
            "math",  # Also export the entire math module
        ]
    )
