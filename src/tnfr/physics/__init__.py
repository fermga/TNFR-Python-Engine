"""TNFR Physics Module — Canonical Structural Telemetry (Expandable)

This package exposes physics-based structural telemetry derived from the
TNFR nodal equation and validated empirically. All functions are centralized
in a single module to avoid duplication and ensure a clear single source of
truth. Documentation is English-only and organized for incremental growth.

Canonical Structural Field Tetrad (Telemetry)
---------------------------------------------
All four fields below are CANONICAL (November 2025) and read-only:

1) Structural Potential (Φ_s)
   - Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α (α=2, inverse-square)
   - Validation: 2,400+ experiments; corr(ΔΦ_s, ΔC) = -0.822; CV = 0.1%
   - Drift scale (U6): Δ Φ_s < π/2 ≈ 1.571 — a conservative bound just inside
     the ζ(2) = π²/6 ≈ 1.6449 saturation of inverse-square accumulation
   - Per-node safety: |Φ_s| < π/4 ≈ 0.785 — π-derived quarter phase-wrap (lies
     within the O(1) ζ(4)=π⁴/90 variance band of inverse-square
     pressure). The earlier empirical 0.7711 / "ψ(x) − x bounds" rationale was
     withdrawn; see theory/MINIMAL_STRUCTURAL_DEGREES.md §4.1.

2) Phase Gradient (|∇φ|)
   - |∇φ|(i) = mean_{j∈N(i)} |wrap(φ_j − φ_i)| (circular differences)
   - Predicts peak stress; |corr| > 0.5 across topologies; threshold 0.38

3) Phase Curvature (K_φ)
   - K_φ(i) = φ_i − mean_circular_{j∈N(i)} φ_j (Laplacian-like curvature)
   - Threshold |K_φ| ≥ 2.8274 (hotspots); multiscale var(K_φ) ~ 1/r^α, α≈2.76

4) Coherence Length (ξ_C)
   - From spatial decay of local coherence correlations C(r) ~ exp(−r/ξ_C)
   - Diverges near I_c (phase transitions); large ξ_C warns system-wide reorg

Physics Foundation
------------------
Nodal equation (per node):  ∂EPI/∂t = ν_f · ΔNFR(t)
EPI changes only via operators; telemetry functions compute read-only fields
from current graph attributes. ν_f uses structural units Hz_str.

Modules
-------
fields : Centralized structural field computations and research utilities
    - compute_structural_potential, compute_phase_gradient,
      compute_phase_curvature, estimate_coherence_length
    - k_φ multiscale helpers and topological winding (Q)
interactions : Canonical operator sequences with telemetry guards
      - em_like, weak_like, strong_like, gravity_like
         (returning InteractionResult)
life : Life emergence detection from autopoietic TNFR dynamics
    - detect_life_emergence, LifeTelemetry, autopoietic coefficients
    - Threshold: A > 1.0 for autopoietic behavior
cell : Cell formation from compartmentalized TNFR life patterns
    - detect_cell_formation, CellTelemetry, membrane selectivity
    - Requires life foundation (A > 1.0) plus spatial organization
phase_transition : Life/non-life phase transition as universal symmetry breaking
    - Order parameter 𝒮, chirality χ, susceptibility, coherence length
    - Critical exponent measured as an observable (audit 2026: NOT the
      universal closed-form scale; the fitted exponent is protocol-dependent)
    - Second-order transition with divergent ξ_C at criticality

See Also
--------
tnfr.operators.grammar : Unified Grammar (U1–U6) and validations
tnfr.dynamics           : Nodal equation integration utilities
docs/STRUCTURAL_FIELDS_TETRAD.md : Canonical tetrad documentation
AGENTS.md               : Canonical invariants and field promotions
src/tnfr/physics/README.md        : Module hub (Patterns,
                                     Interactions, Workflows)

References
----------
- UNIFIED_GRAMMAR_RULES.md (U6: Structural potential confinement)
- docs/STRUCTURAL_FIELDS_TETRAD.md (Φ_s, |∇φ|, K_φ, ξ_C validation)
- AGENTS.md (Structural Fields Tetrad: canonical status, thresholds)
- TNFR.pdf (§2.1: Nodal equation foundation)

Examples
--------
>>> from tnfr.physics.fields import compute_structural_potential
>>> import networkx as nx
>>> G = nx.karate_club_graph()
>>> for node in G.nodes():
...     G.nodes[node]['delta_nfr'] = 0.5
>>> phi_s = compute_structural_potential(G)  # canonical α=2 (inverse-square)
>>> print(f"Potential at node 0: {phi_s[0]:.3f}")

>>> # Telemetry-based U6 safety (ΔΦ_s drift)
>>> phi_before = compute_structural_potential(G)
>>> # ... apply sequence ...
>>> phi_after = compute_structural_potential(G)
>>> drift = sum(
...     abs(phi_after[n] - phi_before[n]) for n in G.nodes()
... ) / G.number_of_nodes()
>>> assert drift < 1.571, "Escape threshold exceeded (π/2)"

"""

from .cell import (
    CellTelemetry,
    apply_membrane_flux,
    compute_boundary_coherence,
    compute_homeostatic_index,
    compute_membrane_integrity,
    compute_selectivity_index,
    detect_cell_formation,
)
from .conservation import (
    ConservationBalance,
    ConservationSnapshot,
    ConservationTimeSeries,
    ConservationTracker,
    LyapunovResult,
    SpectralConservation,
    WardIdentity,
    analyze_sector_coupling,
    capture_conservation_snapshot,
    compute_charge_density,
    compute_conservation_scaling,
    compute_current_divergence,
    compute_energy_functional,
    compute_grammar_conservation_bounds,
    compute_lyapunov_derivative,
    compute_noether_charge,
    compute_spectral_conservation,
    compute_ward_identity,
    decompose_conservation_residual,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
    verify_sequence_ward_identity,
)
from .conservation_gauge_unification import (
    ActionEnergyConsistency,
    ConservationGaugeUnification,
    GaugeConservationCoupling,
    GrammarSymmetryMapping,
    NoetherGaugeDecomposition,
    SymplecticGaugeCompatibility,
    compute_gauge_conservation_coupling,
    compute_grammar_symmetry_mapping,
    compute_noether_gauge_decomposition,
    run_conservation_gauge_unification,
    verify_action_energy_consistency,
    verify_symplectic_gauge_compatibility,
)
from .dissipative_conservation import (
    DissipativeBalance,
    DissipativeConservationTracker,
    DissipativeSnapshot,
    DissipativeTimeSeries,
    analyze_dissipation_rates,
    capture_dissipative_snapshot,
    classify_dissipative_regime,
    compute_dissipation_bound,
    compute_dissipator_action,
    compute_purity_decay_bound,
    predict_amplitude_damping_purity,
    predict_dephasing_purity,
    steady_state_from_generator,
    verify_dissipative_balance,
)
from .fields import (
    compute_k_phi_multiscale_variance,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_phase_winding,
    compute_structural_potential,
    estimate_coherence_length,
    fit_k_phi_asymptotic_alpha,
    k_phi_multiscale_safety,
)
from .gauge import (
    N_REGIMES,
    REGIME_ACTIVITY_SHARE,
    BianchiIdentityResult,
    GaugeInvarianceResult,
    GaugeSnapshot,
    InteractionRegimeMetrics,
    NetworkInteractionProfile,
    YangMillsFieldEquations,
    apply_gauge_transformation,
    capture_gauge_snapshot,
    classify_interaction_regime,
    classify_interaction_regime_formal,
    classify_network_regimes,
    compute_chirality_norm,
    compute_covariant_derivative,
    compute_covariant_derivative_magnitude,
    compute_dual_chirality,
    compute_dual_topological_charge,
    compute_gauge_connection,
    compute_gauge_coupling_constant,
    compute_gauge_curvature,
    compute_gauge_energy_decomposition,
    compute_gauss_law_residual,
    compute_matter_current,
    compute_network_interaction_profile,
    compute_topological_norm,
    compute_yang_mills_action,
    compute_yang_mills_equations,
    verify_bianchi_identity,
    verify_gauge_invariance,
)
from .integrity import (
    IntegrityReport,
    IntegritySummary,
    MonitorMode,
    StructuralIntegrityMonitor,
    StructuralIntegrityViolation,
    enable_integrity_monitor,
)
from .interactions import (
    InteractionResult,
    em_like,
    gravity_like,
    strong_like,
    weak_like,
)
from .life import (
    LifeTelemetry,
    compute_autopoietic_coefficient,
    compute_self_generation,
    compute_self_org_index,
    compute_stability_margin,
    detect_life_emergence,
)
from .lyapunov import (
    OPERATOR_LYAPUNOV_BOUNDS,
    EnergyClass,
    LyapunovSpectralSummary,
    OperatorLyapunovBound,
    OperatorLyapunovVerification,
    SequenceLyapunovProof,
    SpectralGapAnalysis,
    analyze_operator_convergence,
    analyze_spectral_gap,
    compute_operator_energy_bound,
    compute_sequence_energy_bound,
    get_bound,
    prove_sequence_lyapunov,
    verify_operator_lyapunov,
)
from .phase_transition import (
    Z_SIGNIFICANCE,
    Phase,
    PhaseSnapshot,
    PhaseTransitionTelemetry,
    capture_phase_snapshot,
    classify_phase,
    compute_chirality_statistics,
    compute_order_parameter,
    detect_phase_transition,
    fit_critical_exponent,
    symmetry_zscore,
)
from .spectral_conservation import (
    SpectralConservationBalance,
    SpectralLyapunovResult,
    SpectralSectorDecomposition,
    SpectralWardIdentity,
    classify_spectral_modes,
    compute_spectral_energy_conservation,
    compute_spectral_lyapunov,
    compute_spectral_ward_identity,
    decompose_spectral_sectors,
    verify_spectral_conservation_balance,
)
from .structural_diffusion import (
    DiscreteModeCertificate,
    OverdampedRegimeCertificate,
    RandomWalkCertificate,
    StructuralDiffusionCertificate,
    StructuralFlowCertificate,
    StructuralStabilityCertificate,
    commute_time,
    current_divergence,
    degree_weighted_total,
    dispersion_relation,
    effective_resistance,
    fiedler_partition,
    instability_threshold,
    nodal_domain_count,
    random_walk_matrix,
    relaxation_spectrum,
    stationary_distribution,
    structural_current,
    structural_diffusion_operator,
    structural_diffusivity,
    structural_eigenmodes,
    structural_field,
    verify_discrete_modes,
    verify_overdamped_regime,
    verify_structural_diffusion,
    verify_structural_flow,
    verify_structural_random_walk,
    verify_structural_stability,
)
from .symplectic_substrate import (
    CanonicalStructureCertificate,
    HermitianStructureCertificate,
    IntegrabilityCertificate,
    MarsdenWeinsteinCertificate,
    NoetherChargeCertificate,
    PhaseSpacePoint,
    PoincareCartanCertificate,
    PolarizationSymmetryCertificate,
    SubstrateGeometryReport,
    background_potential,
    canonical_bracket_table,
    compatible_metric_matrix,
    complex_structure_matrix,
    diagonal_moment_map,
    evolve_substrate_flow,
    extract_phase_space_point,
    geometric_sector_energy,
    hamiltonian_vector_field,
    kahler_potential,
    liouville_divergence,
    loop_action_integral,
    noether_charges,
    poisson_bracket,
    polarization_density,
    polarization_vector,
    potential_sector_energy,
    reduced_symplectic_form_matrix,
    substrate_flow_matrix,
    substrate_hamiltonian,
    symplectic_form_matrix,
    to_action_angle,
    to_complex_coordinates,
    verify_canonical_structure,
    verify_hermitian_structure,
    verify_integrability,
    verify_noether_conservation,
    verify_poincare_cartan,
    verify_polarization_symmetry,
    verify_substrate_geometry,
    verify_symplectic_reduction,
)
from .unified import (
    compute_action_density,
    compute_chirality_field,
    compute_coherence_coupling_field,
    compute_complex_geometric_field,
    compute_energy_density,
    compute_field_magnitude,
    compute_field_phase,
    compute_symmetry_breaking_field,
    compute_topological_charge,
    compute_unified_field_suite,
)
from .variational import (
    ConjugatePair,
    CriticalPointAnalysis,
    EulerLagrangeResidual,
    GrammarStationarityAnalysis,
    LagrangianSnapshot,
    SymplecticCheck,
    VariationalTimeSeries,
    VariationalTracker,
    analyze_grammar_stationarity,
    analyze_potential_critical_points,
    capture_lagrangian_snapshot,
    check_symplectic_preservation,
    classify_operator_canonical,
    compute_action_functional,
    compute_euler_lagrange_residual,
    compute_hamiltonian_density,
    compute_interaction_density,
    compute_kinetic_density,
    compute_lagrangian_density,
    compute_phase_space_volume,
    compute_poisson_bracket_estimate,
    compute_potential_density,
    compute_variational_suite,
    identify_conjugate_pairs,
    translate_sectors,
)

__all__ = [
    # --- Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) ---
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
    "compute_k_phi_multiscale_variance",
    "fit_k_phi_asymptotic_alpha",
    "k_phi_multiscale_safety",
    "compute_phase_winding",
    # --- Force-like Interactions ---
    "InteractionResult",
    "em_like",
    "weak_like",
    "strong_like",
    "gravity_like",
    # --- Life Emergence ---
    "LifeTelemetry",
    "compute_self_generation",
    "compute_autopoietic_coefficient",
    "compute_self_org_index",
    "compute_stability_margin",
    "detect_life_emergence",
    # --- Cell / Membrane ---
    "CellTelemetry",
    "compute_boundary_coherence",
    "compute_selectivity_index",
    "compute_homeostatic_index",
    "compute_membrane_integrity",
    "detect_cell_formation",
    "apply_membrane_flux",
    # --- Conservation Theorem (Noether-like) ---
    "ConservationSnapshot",
    "ConservationBalance",
    "ConservationTimeSeries",
    "ConservationTracker",
    "compute_charge_density",
    "compute_current_divergence",
    "capture_conservation_snapshot",
    "verify_conservation_balance",
    "decompose_conservation_residual",
    "analyze_sector_coupling",
    "compute_grammar_conservation_bounds",
    "detect_grammar_violations_from_conservation",
    "compute_noether_charge",
    "compute_energy_functional",
    "WardIdentity",
    "LyapunovResult",
    "SpectralConservation",
    "compute_ward_identity",
    "verify_sequence_ward_identity",
    "compute_lyapunov_derivative",
    "compute_spectral_conservation",
    "compute_conservation_scaling",
    # --- Unified Complex Fields (Ψ = K_φ + i·J_φ) ---
    "compute_complex_geometric_field",
    "compute_field_magnitude",
    "compute_field_phase",
    "compute_chirality_field",
    "compute_symmetry_breaking_field",
    "compute_coherence_coupling_field",
    "compute_energy_density",
    "compute_action_density",
    "compute_topological_charge",
    "compute_unified_field_suite",
    # --- Dissipative Conservation ---
    "DissipativeSnapshot",
    "DissipativeBalance",
    "DissipativeTimeSeries",
    "DissipativeConservationTracker",
    "capture_dissipative_snapshot",
    "compute_dissipation_bound",
    "compute_dissipator_action",
    "compute_purity_decay_bound",
    "verify_dissipative_balance",
    "predict_amplitude_damping_purity",
    "predict_dephasing_purity",
    "analyze_dissipation_rates",
    "classify_dissipative_regime",
    "steady_state_from_generator",
    # --- Structural Integrity Monitor ---
    "StructuralIntegrityMonitor",
    "StructuralIntegrityViolation",
    "MonitorMode",
    "IntegrityReport",
    "IntegritySummary",
    "enable_integrity_monitor",
    # --- Spectral Conservation ---
    "SpectralConservationBalance",
    "SpectralWardIdentity",
    "SpectralLyapunovResult",
    "SpectralSectorDecomposition",
    "verify_spectral_conservation_balance",
    "compute_spectral_ward_identity",
    "compute_spectral_lyapunov",
    "decompose_spectral_sectors",
    "compute_spectral_energy_conservation",
    "classify_spectral_modes",
    # --- Variational Principle (Lagrangian Action) ---
    "ConjugatePair",
    "LagrangianSnapshot",
    "EulerLagrangeResidual",
    "SymplecticCheck",
    "GrammarStationarityAnalysis",
    "CriticalPointAnalysis",
    "VariationalTimeSeries",
    "VariationalTracker",
    "compute_kinetic_density",
    "compute_potential_density",
    "compute_lagrangian_density",
    "compute_hamiltonian_density",
    "compute_interaction_density",
    "translate_sectors",
    "identify_conjugate_pairs",
    "compute_phase_space_volume",
    "compute_poisson_bracket_estimate",
    "capture_lagrangian_snapshot",
    "compute_euler_lagrange_residual",
    "compute_action_functional",
    "check_symplectic_preservation",
    "classify_operator_canonical",
    "analyze_grammar_stationarity",
    "analyze_potential_critical_points",
    "compute_variational_suite",
    # --- Emergent Symplectic Substrate (geometry the dynamics generates) ---
    "PhaseSpacePoint",
    "CanonicalStructureCertificate",
    "NoetherChargeCertificate",
    "HermitianStructureCertificate",
    "IntegrabilityCertificate",
    "PoincareCartanCertificate",
    "MarsdenWeinsteinCertificate",
    "PolarizationSymmetryCertificate",
    "SubstrateGeometryReport",
    "extract_phase_space_point",
    "symplectic_form_matrix",
    "complex_structure_matrix",
    "compatible_metric_matrix",
    "substrate_hamiltonian",
    "background_potential",
    "hamiltonian_vector_field",
    "poisson_bracket",
    "canonical_bracket_table",
    "liouville_divergence",
    "verify_canonical_structure",
    "evolve_substrate_flow",
    "geometric_sector_energy",
    "potential_sector_energy",
    "noether_charges",
    "verify_noether_conservation",
    "to_complex_coordinates",
    "kahler_potential",
    "verify_hermitian_structure",
    "to_action_angle",
    "verify_integrability",
    "substrate_flow_matrix",
    "loop_action_integral",
    "verify_poincare_cartan",
    "diagonal_moment_map",
    "reduced_symplectic_form_matrix",
    "verify_symplectic_reduction",
    "polarization_vector",
    "polarization_density",
    "verify_polarization_symmetry",
    "verify_substrate_geometry",
    # --- Structural Diffusion (transport content of the nodal equation) ---
    "StructuralDiffusionCertificate",
    "OverdampedRegimeCertificate",
    "DiscreteModeCertificate",
    "StructuralStabilityCertificate",
    "RandomWalkCertificate",
    "StructuralFlowCertificate",
    "structural_diffusion_operator",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "degree_weighted_total",
    "structural_eigenmodes",
    "nodal_domain_count",
    "dispersion_relation",
    "instability_threshold",
    "fiedler_partition",
    "random_walk_matrix",
    "stationary_distribution",
    "effective_resistance",
    "commute_time",
    "structural_current",
    "current_divergence",
    "verify_structural_diffusion",
    "verify_overdamped_regime",
    "verify_discrete_modes",
    "verify_structural_stability",
    "verify_structural_random_walk",
    "verify_structural_flow",
    # --- Gauge Structure (U(1) Symmetry of Ψ = K_φ + i·J_φ) ---
    "GaugeSnapshot",
    "GaugeInvarianceResult",
    "apply_gauge_transformation",
    "compute_gauge_connection",
    "compute_gauge_curvature",
    "compute_covariant_derivative",
    "compute_covariant_derivative_magnitude",
    "compute_topological_norm",
    "compute_chirality_norm",
    "compute_dual_topological_charge",
    "compute_dual_chirality",
    "verify_gauge_invariance",
    "capture_gauge_snapshot",
    "classify_interaction_regime",
    "classify_network_regimes",
    "compute_yang_mills_action",
    "compute_gauge_energy_decomposition",
    # --- Yang-Mills Formalism (complete field equations) ---
    "YangMillsFieldEquations",
    "BianchiIdentityResult",
    "InteractionRegimeMetrics",
    "NetworkInteractionProfile",
    "N_REGIMES",
    "REGIME_ACTIVITY_SHARE",
    "compute_matter_current",
    "compute_yang_mills_equations",
    "verify_bianchi_identity",
    "compute_gauss_law_residual",
    "compute_gauge_coupling_constant",
    "classify_interaction_regime_formal",
    "compute_network_interaction_profile",
    # --- Phase Transition (Life/Non-Life Symmetry Breaking) ---
    "Phase",
    "PhaseTransitionTelemetry",
    "PhaseSnapshot",
    "Z_SIGNIFICANCE",
    "symmetry_zscore",
    "compute_order_parameter",
    "compute_chirality_statistics",
    "classify_phase",
    "capture_phase_snapshot",
    "detect_phase_transition",
    "fit_critical_exponent",
    # --- Formal Lyapunov Analysis (per-operator bounds + spectral gap) ---
    "EnergyClass",
    "OperatorLyapunovBound",
    "OperatorLyapunovVerification",
    "SpectralGapAnalysis",
    "LyapunovSpectralSummary",
    "SequenceLyapunovProof",
    "OPERATOR_LYAPUNOV_BOUNDS",
    "get_bound",
    "compute_operator_energy_bound",
    "verify_operator_lyapunov",
    "compute_sequence_energy_bound",
    "prove_sequence_lyapunov",
    "analyze_spectral_gap",
    "analyze_operator_convergence",
    # --- Conservation-Gauge Unification ---
    "GrammarSymmetryMapping",
    "ActionEnergyConsistency",
    "NoetherGaugeDecomposition",
    "GaugeConservationCoupling",
    "SymplecticGaugeCompatibility",
    "ConservationGaugeUnification",
    "compute_grammar_symmetry_mapping",
    "verify_action_energy_consistency",
    "compute_noether_gauge_decomposition",
    "compute_gauge_conservation_coupling",
    "verify_symplectic_gauge_compatibility",
    "run_conservation_gauge_unification",
]
