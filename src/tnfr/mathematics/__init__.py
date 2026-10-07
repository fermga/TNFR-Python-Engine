"""Mathematics primitives aligned with TNFR coherence modeling.

Backend selection
-----------------
Use :func:`get_backend` to retrieve a numerical backend compatible with TNFR's
structural operators. The selection order is ``name`` → ``TNFR_MATH_BACKEND``
→ ``tnfr.backend_config.get_config().math_backend``. The configuration defaults
to ``auto``: GPU-capable adapters are preferred, then any available adapter,
in JAX → PyTorch → NumPy order. Request ``numpy`` explicitly for that backend.

Symbolic Analysis
-----------------
When SymPy is installed, this module re-exports the supplied-law symbolic
identities and integration helpers owned by ``tnfr.math.symbolic``.
"""

from .backend import (
    MathematicsBackend,
    available_backends,
    ensure_array,
    ensure_numpy,
    get_backend,
    register_backend,
)
from .cayley import (
    cayley_action,
    cayley_diffusion_action,
    cayley_first_row,
    cayley_laplacian,
    cayley_spectrum,
)
from .dynamics import ContractiveDynamicsEngine, MathematicalDynamicsEngine
from .epi import (
    COMPOSITE_EPI_REGULARITY_KIND,
    COMPOSITE_EPI_REGULARITY_PROVENANCE,
    BEPIElement,
    CoherenceEvaluation,
    CompositeEPIRegularityEvaluation,
    evaluate_coherence_transform,
    evaluate_composite_epi_regularity_transform,
)
from .generators import build_delta_nfr, build_lindblad_delta_nfr
from .liouville import (
    compute_liouvillian_spectrum,
    get_liouvillian_spectrum,
    get_slow_relaxation_mode,
    store_liouvillian_spectrum,
)
from .metrics import dcoh, spectral_weighted_angle
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
    unit_power_residue_set,
    unitary_residue_set,
)
from .operators import CoherenceOperator, FrequencyOperator, SpectralExpectationOperator
from .operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
    make_spectral_expectation_operator,
)
from .projection import BasicStateProjector, StateProjector
from .runtime import (
    coherence,
    coherence_expectation,
    frequency_expectation,
    frequency_positive,
    meets_spectral_expectation_threshold,
    normalized,
    spectral_operator_expectation,
    stable_unitary,
)
from .spaces import BanachSpaceEPI, HilbertSpace
from .transforms import (
    CoherenceMonotonicityReport,
    CoherenceViolation,
    CompositeEPIRegularityTrendReport,
    IsometryFactory,
    RegularityTrendViolation,
    assess_composite_epi_regularity_trend,
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
    NUMPY_AVAILABLE,
    PI,
    ArrayLike,
    ComplexArray,
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

# Keep numerical imports available when the optional symbolic dependency is absent.
try:
    from .. import math
    from ..math import (
        check_convergence_exponential,
        compute_second_derivative_symbolic,
        get_nodal_equation,
        integrated_evolution_symbolic,
        latex_export,
        pretty_print,
        solve_nodal_equation_constant_params,
    )
except ModuleNotFoundError as exc:
    if exc.name != "sympy":
        raise
    _HAS_SYMBOLIC = False
else:
    _HAS_SYMBOLIC = True

__all__ = [
    # Backend operations
    "MathematicsBackend",
    "ensure_array",
    "ensure_numpy",
    "HilbertSpace",
    "BanachSpaceEPI",
    "BEPIElement",
    "COMPOSITE_EPI_REGULARITY_KIND",
    "COMPOSITE_EPI_REGULARITY_PROVENANCE",
    "CompositeEPIRegularityEvaluation",
    "CoherenceEvaluation",
    "CoherenceOperator",
    "SpectralExpectationOperator",
    "ContractiveDynamicsEngine",
    "CompositeEPIRegularityTrendReport",
    "RegularityTrendViolation",
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
    "make_spectral_expectation_operator",
    "make_frequency_operator",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "assess_composite_epi_regularity_trend",
    "evaluate_composite_epi_regularity_transform",
    "ensure_coherence_monotonicity",
    "evaluate_coherence_transform",
    "StateProjector",
    "BasicStateProjector",
    "normalized",
    "coherence",
    "meets_spectral_expectation_threshold",
    "frequency_positive",
    "stable_unitary",
    "dcoh",
    "spectral_weighted_angle",
    "coherence_expectation",
    "spectral_operator_expectation",
    "frequency_expectation",
    "available_backends",
    "get_backend",
    "register_backend",
    # Exact circulant structural operators
    "cayley_laplacian",
    "cayley_first_row",
    "cayley_action",
    "cayley_spectrum",
    "cayley_diffusion_action",
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
    "PI",
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
    "unit_power_residue_set",
    "unitary_residue_set",
    "arithmetic_cayley_digraph",
    "residue_network_rank",
    "power_residue_rank",
    "quadratic_residue_annotated_rank",
]

# Reuse the symbolic facade's existing export order without its module metadata.
if _HAS_SYMBOLIC:
    __all__.extend(
        name for name in math.__all__ if name not in {"symbolic", "__version__"}
    )
    __all__.append("math")
