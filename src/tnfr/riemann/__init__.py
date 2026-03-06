"""TNFR-Riemann spectral analysis module.

Provides the discrete TNFR-Riemann operator H^(k)(sigma) = L_k + V_sigma,
spectral proof framework, alternative graph topologies, per-eigenmode
structural field tetrad, complex-s non-Hermitian extension, and discrete
spectral zeta / trace formula analysis.
See theory/TNFR_RIEMANN_RESEARCH_NOTES.md for the full theoretical background.

Sub-modules
-----------
operator
    Graph builders and discrete operator construction.
spectral_proof
    Four-line spectral analysis and integrated assessment.
topology
    Alternative graph topologies and convergence studies.
eigenmode_fields
    Per-eigenmode structural field tetrad (Phi_s, |grad_phi|, K_phi, xi_C).
complex_extension
    P4 complex-s extension: non-Hermitian operator H(s) for s in C.
spectral_zeta
    P5 discrete spectral zeta, heat kernel trace, and Conjecture 10.1.
random_ensemble
    P6 random prime-graph ensembles and RMT universality analysis.
spectral_conservation
    P7 conservation laws and grammar compliance at criticality.
analytical_convergence
    P8 analytical proof of σ* → 1/2 rate via PNT.
functional_equation
    P9 functional equation analog and spectral reflection symmetry.
convergence_proof
    P10 formal machine-verified proof that σ*(k) → 1/2.
zeta_bridge
    P11 explicit bridge between spectral ζ_H(k) and Riemann ζ_R.
"""

from .operator import (
    build_prime_path_graph,
    build_prime_cycle_graph,
    build_prime_star_graph,
    build_prime_complete_graph,
    build_prime_tree_graph,
    build_prime_random_graph,
    build_h_tnfr,
    build_tridiagonal_h_tnfr,
    default_prime_potential,
    build_h_tnfr_complex,
    build_tridiagonal_h_tnfr_complex,
    default_prime_potential_complex,
)
from .topology import (
    # Data structures
    TopologyResult,
    TopologyConvergenceResult,
    # Registry
    TOPOLOGY_BUILDERS,
    # Analysis
    analyze_graph_topology,
    compare_topologies,
    topology_convergence_study,
)
from .spectral_proof import (
    # Data structures
    EquilibriumResult,
    ThermodynamicResult,
    EigenvalueFlowResult,
    SpectralMomentResult,
    TNFRRiemannAssessment,
    # Core
    compute_eigenspectrum,
    compute_eigensystem,
    # Line 1 - Structural Equilibrium
    verify_equilibrium,
    verify_equilibrium_sequence,
    # Line 2 - Thermodynamic Attractor
    compute_analytic_sigma_star,
    compute_frobenius_energy,
    compute_thermodynamic_landscape,
    verify_thermodynamic_convergence,
    # Line 3 - Eigenvalue Flow
    compute_eigenvalue_velocities,
    analyze_eigenvalue_flow,
    # Line 4 - Spectral Moments
    compute_eigenvalue_spacings,
    compute_spectral_moments,
    # Integration
    run_tnfr_riemann_analysis,
)
from .eigenmode_fields import (
    # Data structures
    EigenmodeTetrad,
    EigenmodeFieldAnalysis,
    # Constants
    PHI_S_VON_KOCH_THRESHOLD,
    PHI_S_GOLDEN_THRESHOLD,
    # Core
    compute_eigenmode_tetrad,
    compute_eigenmode_fields_general,
    # Diagnostics
    check_u6_confinement,
    compare_confinement_at_sigma,
)
from .complex_extension import (
    # Data structures
    ComplexEigenResult,
    CriticalLineScan,
    PseudoSpectrumResult,
    ResolventAnalysis,
    ComplexPlaneAnalysis,
    # Constants
    KNOWN_RIEMANN_ZEROS,
    # Core
    compute_complex_eigenspectrum,
    compute_complex_eigensystem,
    analyze_non_hermiticity,
    # Critical line
    scan_critical_line,
    find_eigenvalue_zero_crossings,
    # Pseudo-spectrum & resolvent
    compute_pseudospectrum,
    compute_resolvent_norm,
    analyze_resolvent_along_critical_line,
    # Riemann zero comparison
    compare_with_riemann_zeros,
    # Integration
    run_complex_plane_analysis,
)
from .spectral_zeta import (
    # Data structures
    SpectralZetaResult,
    HeatKernelResult,
    MellinBridgeResult,
    ConjectureTestResult,
    SpectralZetaAnalysis,
    # Constants
    RIEMANN_ZETA_KNOWN_VALUES,
    # Core computation
    compute_positive_eigenvalues,
    compute_spectral_zeta,
    compute_spectral_zeta_derivative,
    # Heat kernel
    compute_heat_kernel_trace,
    compute_partition_function,
    compute_free_energy,
    # Mellin bridge
    verify_mellin_bridge,
    # Conjecture 10.1
    riemann_zeta_approx,
    test_conjecture_10_1,
    test_conjecture_10_1_sequence,
    # Integration
    run_spectral_zeta_analysis,
)
from .random_ensemble import (
    # Data structures
    EnsembleConfig,
    EnsembleSample,
    SpacingStats,
    RMTComparison,
    EnsembleAnalysis,
    # Reference distributions
    goe_wigner_surmise,
    gue_wigner_surmise,
    poisson_spacing_pdf,
    # Ensemble generation
    generate_er_ensemble,
    generate_wigner_ensemble,
    # Spacing statistics
    compute_ensemble_spacings,
    compute_spacing_ratio,
    compute_mean_spacing_ratio,
    compute_level_repulsion_exponent,
    # Long-range statistics
    compute_number_variance,
    compute_spectral_rigidity,
    # RMT comparison
    ks_test_vs_reference,
    classify_ensemble,
    # Integration
    run_rmt_ensemble_analysis,
    rmt_convergence_study,
)
from .spectral_conservation import (
    # Data structures
    EigenmodeConservation,
    ConservationAtSigma,
    ConservationSigmaScan,
    GrammarComplianceResult,
    CriticalConservationAnalysis,
    # Core eigenmode conservation
    compute_spectral_j_phi,
    compute_spectral_j_dnfr,
    compute_eigenmode_conservation,
    # Sigma scan
    scan_conservation_vs_sigma,
    # Grammar compliance
    test_grammar_conservation,
    # Integration
    run_critical_conservation_analysis,
)
from .analytical_convergence import (
    # Data structures
    TelescopingIdentity,
    PNTAsymptoticBound,
    ConvergenceRateBound,
    EffectiveConstantResult,
    AnalyticalConvergenceProof,
    # Telescoping identity (Theorem 1)
    compute_telescoping_trace,
    verify_telescoping_identity,
    # PNT asymptotics (Theorem 2)
    pnt_prime_estimate,
    euler_maclaurin_log_squared_sum,
    pnt_sum_log_squared,
    # Convergence rate (Theorem 3)
    compute_convergence_rate_bound,
    compute_effective_constant,
    analyze_convergence_sequence,
    # Integration
    run_analytical_convergence_proof,
)
from .functional_equation import (
    # Data structures
    SpectralReflection,
    TraceFormulaResult,
    CompletedXiFunction,
    Conjecture12_1Result,
    Conjecture12_2Result,
    LargeKConvergence,
    FunctionalEquationAnalysis,
    # Core — reflection symmetry
    verify_spectral_reflection,
    verify_reflection_sequence,
    # Core — trace formulas
    compute_trace_formulas,
    verify_trace_formula_pnt,
    # Core — completed xi
    compute_completed_xi,
    verify_xi_functional_equation,
    # Conjectures
    test_conjecture_12_1,
    test_conjecture_12_2,
    # Large-k verification
    verify_large_k_convergence,
    # Integration
    run_functional_equation_analysis,
)
from .convergence_proof import (
    # Data structures
    ProofStep,
    DusartVerification,
    ExplicitBoundResult,
    CurvatureGrowthResult,
    CKAsymptoticFit,
    FormalConvergenceProof,
    # Dusart bounds
    dusart_lower_bound,
    dusart_upper_bound,
    verify_dusart_bounds,
    # Proof steps
    prove_bilinear_decomposition,
    prove_telescoping,
    prove_sum_lower_bound,
    prove_convergence_rate,
    prove_explicit_bound,
    prove_curvature_divergence,
    # Explicit bound
    scan_effective_constant,
    compute_explicit_bound_constant,
    # C(k) asymptotics
    fit_ck_asymptotics,
    # Integration
    run_formal_convergence_proof,
)
from .zeta_bridge import (
    # Data structures
    WeylAsymptotic,
    HeatKernelReflection,
    SpectralZetaReflection,
    ScalingLaw,
    PrimeEncoding,
    ZetaBridgeAnalysis,
    # Functions
    compute_weyl_asymptotic,
    compute_heat_kernel_reflection,
    compute_spectral_zeta_reflection,
    extract_scaling_law,
    compute_prime_encoding,
    run_zeta_bridge_analysis,
)

__all__ = [
    # Graph builders
    "build_prime_path_graph",
    "build_prime_cycle_graph",
    "build_prime_star_graph",
    "build_prime_complete_graph",
    "build_prime_tree_graph",
    "build_prime_random_graph",
    "build_h_tnfr",
    "build_tridiagonal_h_tnfr",
    "default_prime_potential",
    # Topology comparison (P2)
    "TopologyResult",
    "TopologyConvergenceResult",
    "TOPOLOGY_BUILDERS",
    "analyze_graph_topology",
    "compare_topologies",
    "topology_convergence_study",
    # Data structures
    "EquilibriumResult",
    "ThermodynamicResult",
    "EigenvalueFlowResult",
    "SpectralMomentResult",
    "TNFRRiemannAssessment",
    # Core
    "compute_eigenspectrum",
    "compute_eigensystem",
    # Line 1 - Structural Equilibrium
    "verify_equilibrium",
    "verify_equilibrium_sequence",
    # Line 2 - Thermodynamic Attractor
    "compute_analytic_sigma_star",
    "compute_frobenius_energy",
    "compute_thermodynamic_landscape",
    "verify_thermodynamic_convergence",
    # Line 3 - Eigenvalue Flow
    "compute_eigenvalue_velocities",
    "analyze_eigenvalue_flow",
    # Line 4 - Spectral Moments
    "compute_eigenvalue_spacings",
    "compute_spectral_moments",
    # Integration
    "run_tnfr_riemann_analysis",
    # Per-eigenmode tetrad (P3)
    "EigenmodeTetrad",
    "EigenmodeFieldAnalysis",
    "PHI_S_VON_KOCH_THRESHOLD",
    "PHI_S_GOLDEN_THRESHOLD",
    "compute_eigenmode_tetrad",
    "compute_eigenmode_fields_general",
    "check_u6_confinement",
    "compare_confinement_at_sigma",
    # Complex-s extension (P4)
    "ComplexEigenResult",
    "CriticalLineScan",
    "PseudoSpectrumResult",
    "ResolventAnalysis",
    "ComplexPlaneAnalysis",
    "KNOWN_RIEMANN_ZEROS",
    "build_h_tnfr_complex",
    "build_tridiagonal_h_tnfr_complex",
    "default_prime_potential_complex",
    "compute_complex_eigenspectrum",
    "compute_complex_eigensystem",
    "analyze_non_hermiticity",
    "scan_critical_line",
    "find_eigenvalue_zero_crossings",
    "compute_pseudospectrum",
    "compute_resolvent_norm",
    "analyze_resolvent_along_critical_line",
    "compare_with_riemann_zeros",
    "run_complex_plane_analysis",
    # Spectral zeta & trace formula (P5)
    "SpectralZetaResult",
    "HeatKernelResult",
    "MellinBridgeResult",
    "ConjectureTestResult",
    "SpectralZetaAnalysis",
    "RIEMANN_ZETA_KNOWN_VALUES",
    "compute_positive_eigenvalues",
    "compute_spectral_zeta",
    "compute_spectral_zeta_derivative",
    "compute_heat_kernel_trace",
    "compute_partition_function",
    "compute_free_energy",
    "verify_mellin_bridge",
    "riemann_zeta_approx",
    "test_conjecture_10_1",
    "test_conjecture_10_1_sequence",
    "run_spectral_zeta_analysis",
    # Random ensemble RMT (P6)
    "EnsembleConfig",
    "EnsembleSample",
    "SpacingStats",
    "RMTComparison",
    "EnsembleAnalysis",
    "goe_wigner_surmise",
    "gue_wigner_surmise",
    "poisson_spacing_pdf",
    "generate_er_ensemble",
    "generate_wigner_ensemble",
    "compute_ensemble_spacings",
    "compute_spacing_ratio",
    "compute_mean_spacing_ratio",
    "compute_level_repulsion_exponent",
    "compute_number_variance",
    "compute_spectral_rigidity",
    "ks_test_vs_reference",
    "classify_ensemble",
    "run_rmt_ensemble_analysis",
    "rmt_convergence_study",
    # Spectral conservation at criticality (P7)
    "EigenmodeConservation",
    "ConservationAtSigma",
    "ConservationSigmaScan",
    "GrammarComplianceResult",
    "CriticalConservationAnalysis",
    "compute_spectral_j_phi",
    "compute_spectral_j_dnfr",
    "compute_eigenmode_conservation",
    "scan_conservation_vs_sigma",
    "test_grammar_conservation",
    "run_critical_conservation_analysis",
    # Analytical convergence proof (P8)
    "TelescopingIdentity",
    "PNTAsymptoticBound",
    "ConvergenceRateBound",
    "EffectiveConstantResult",
    "AnalyticalConvergenceProof",
    "compute_telescoping_trace",
    "verify_telescoping_identity",
    "pnt_prime_estimate",
    "euler_maclaurin_log_squared_sum",
    "pnt_sum_log_squared",
    "compute_convergence_rate_bound",
    "compute_effective_constant",
    "analyze_convergence_sequence",
    "run_analytical_convergence_proof",
    # Functional equation & reflection symmetry (P9)
    "SpectralReflection",
    "TraceFormulaResult",
    "CompletedXiFunction",
    "Conjecture12_1Result",
    "Conjecture12_2Result",
    "LargeKConvergence",
    "FunctionalEquationAnalysis",
    "verify_spectral_reflection",
    "verify_reflection_sequence",
    "compute_trace_formulas",
    "verify_trace_formula_pnt",
    "compute_completed_xi",
    "verify_xi_functional_equation",
    "test_conjecture_12_1",
    "test_conjecture_12_2",
    "verify_large_k_convergence",
    "run_functional_equation_analysis",
    # Formal convergence proof (P10)
    "ProofStep",
    "DusartVerification",
    "ExplicitBoundResult",
    "CurvatureGrowthResult",
    "CKAsymptoticFit",
    "FormalConvergenceProof",
    "dusart_lower_bound",
    "dusart_upper_bound",
    "verify_dusart_bounds",
    "prove_bilinear_decomposition",
    "prove_telescoping",
    "prove_sum_lower_bound",
    "prove_convergence_rate",
    "prove_explicit_bound",
    "prove_curvature_divergence",
    "scan_effective_constant",
    "compute_explicit_bound_constant",
    "fit_ck_asymptotics",
    "run_formal_convergence_proof",
    # Zeta bridge (P11)
    "WeylAsymptotic",
    "HeatKernelReflection",
    "SpectralZetaReflection",
    "ScalingLaw",
    "PrimeEncoding",
    "ZetaBridgeAnalysis",
    "compute_weyl_asymptotic",
    "compute_heat_kernel_reflection",
    "compute_spectral_zeta_reflection",
    "extract_scaling_law",
    "compute_prime_encoding",
    "run_zeta_bridge_analysis",
]
