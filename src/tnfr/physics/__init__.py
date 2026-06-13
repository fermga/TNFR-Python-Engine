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
   - Drift scale (U6): Δ Φ_s < φ ≈ 1.618 — a conservative bound just inside
     the ζ(2) = π²/6 ≈ 1.6449 saturation of inverse-square accumulation
   - Per-node safety: |Φ_s| < 0.7711 — EMPIRICAL threshold (no closed form;
     lies within the O(1) ζ(4)=π⁴/90 variance band of inverse-square
     pressure). The earlier "ψ(x) − x bounds" rationale was fabricated and is
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
    - Critical exponent γ_c = γ/π ≈ 0.1837 from Tetrahedral Correspondence
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
>>> assert drift < 1.618, "Escape threshold exceeded (φ)"

"""

from .fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
    compute_k_phi_multiscale_variance,
    fit_k_phi_asymptotic_alpha,
    k_phi_multiscale_safety,
    compute_phase_winding,
)
from .interactions import InteractionResult
from .interactions import em_like
from .interactions import weak_like
from .interactions import strong_like
from .interactions import gravity_like
from .life import LifeTelemetry
from .life import compute_self_generation
from .life import compute_autopoietic_coefficient
from .life import compute_self_org_index
from .life import compute_stability_margin
from .life import detect_life_emergence
from .cell import CellTelemetry
from .cell import compute_boundary_coherence
from .cell import compute_selectivity_index
from .cell import compute_homeostatic_index
from .cell import compute_membrane_integrity
from .cell import detect_cell_formation
from .cell import apply_membrane_flux
from .conservation import (
    ConservationSnapshot,
    ConservationBalance,
    ConservationTimeSeries,
    ConservationTracker,
    WardIdentity,
    LyapunovResult,
    SpectralConservation,
    compute_charge_density,
    compute_current_divergence,
    capture_conservation_snapshot,
    verify_conservation_balance,
    decompose_conservation_residual,
    analyze_sector_coupling,
    compute_grammar_conservation_bounds,
    detect_grammar_violations_from_conservation,
    compute_noether_charge,
    compute_energy_functional,
    compute_ward_identity,
    verify_sequence_ward_identity,
    compute_lyapunov_derivative,
    compute_spectral_conservation,
    compute_conservation_scaling,
)
from .dissipative_conservation import (
    DissipativeSnapshot,
    DissipativeBalance,
    DissipativeTimeSeries,
    DissipativeConservationTracker,
    capture_dissipative_snapshot,
    compute_dissipation_bound,
    compute_dissipator_action,
    compute_purity_decay_bound,
    verify_dissipative_balance,
    predict_amplitude_damping_purity,
    predict_dephasing_purity,
    analyze_dissipation_rates,
    classify_dissipative_regime,
    steady_state_from_generator,
)
from .integrity import (
    StructuralIntegrityMonitor,
    StructuralIntegrityViolation,
    MonitorMode,
    IntegrityReport,
    IntegritySummary,
    enable_integrity_monitor,
)
from .spectral_conservation import (
    SpectralConservationBalance,
    SpectralWardIdentity,
    SpectralLyapunovResult,
    SpectralSectorDecomposition,
    verify_spectral_conservation_balance,
    compute_spectral_ward_identity,
    compute_spectral_lyapunov,
    decompose_spectral_sectors,
    compute_spectral_energy_conservation,
    classify_spectral_modes,
)
from .unified import (
    compute_complex_geometric_field,
    compute_field_magnitude,
    compute_field_phase,
    compute_chirality_field,
    compute_symmetry_breaking_field,
    compute_coherence_coupling_field,
    compute_energy_density,
    compute_action_density,
    compute_topological_charge,
    compute_unified_field_suite,
)
from .variational import (
    ConjugatePair,
    LagrangianSnapshot,
    EulerLagrangeResidual,
    SymplecticCheck,
    GrammarStationarityAnalysis,
    CriticalPointAnalysis,
    VariationalTimeSeries,
    VariationalTracker,
    compute_kinetic_density,
    compute_potential_density,
    compute_lagrangian_density,
    compute_hamiltonian_density,
    compute_interaction_density,
    translate_sectors,
    identify_conjugate_pairs,
    compute_phase_space_volume,
    compute_poisson_bracket_estimate,
    capture_lagrangian_snapshot,
    compute_euler_lagrange_residual,
    compute_action_functional,
    check_symplectic_preservation,
    classify_operator_canonical,
    analyze_grammar_stationarity,
    analyze_potential_critical_points,
    compute_variational_suite,
)
from .symplectic_substrate import (
    PhaseSpacePoint,
    CanonicalStructureCertificate,
    extract_phase_space_point,
    symplectic_form_matrix,
    substrate_hamiltonian,
    background_potential,
    hamiltonian_vector_field,
    poisson_bracket,
    canonical_bracket_table,
    liouville_divergence,
    verify_canonical_structure,
)
from .gauge import (
    GaugeSnapshot,
    GaugeInvarianceResult,
    apply_gauge_transformation,
    compute_gauge_connection,
    compute_gauge_curvature,
    compute_covariant_derivative,
    compute_covariant_derivative_magnitude,
    compute_topological_norm,
    compute_chirality_norm,
    compute_dual_topological_charge,
    compute_dual_chirality,
    verify_gauge_invariance,
    capture_gauge_snapshot,
    classify_interaction_regime,
    classify_network_regimes,
    compute_yang_mills_action,
    compute_gauge_energy_decomposition,
    YangMillsFieldEquations,
    BianchiIdentityResult,
    InteractionRegimeMetrics,
    NetworkInteractionProfile,
    REGIME_DOMINANCE_THRESHOLD,
    REGIME_STRONG_THRESHOLD,
    compute_matter_current,
    compute_yang_mills_equations,
    verify_bianchi_identity,
    compute_gauss_law_residual,
    compute_gauge_coupling_constant,
    classify_interaction_regime_formal,
    compute_network_interaction_profile,
)
from .phase_transition import (
    Phase,
    PhaseTransitionTelemetry,
    PhaseSnapshot,
    GAMMA_C,
    ORDER_PARAMETER_NOISE_FLOOR,
    CHIRALITY_THRESHOLD,
    compute_order_parameter,
    compute_chirality_statistics,
    classify_phase,
    capture_phase_snapshot,
    detect_phase_transition,
    fit_critical_exponent,
)
from .lyapunov import (
    EnergyClass,
    OperatorLyapunovBound,
    OperatorLyapunovVerification,
    SpectralGapAnalysis,
    LyapunovSpectralSummary,
    SequenceLyapunovProof,
    OPERATOR_LYAPUNOV_BOUNDS,
    get_bound,
    compute_operator_energy_bound,
    verify_operator_lyapunov,
    compute_sequence_energy_bound,
    prove_sequence_lyapunov,
    analyze_spectral_gap,
    analyze_operator_convergence,
)
from .conservation_gauge_unification import (
    GrammarSymmetryMapping,
    ActionEnergyConsistency,
    NoetherGaugeDecomposition,
    GaugeConservationCoupling,
    SymplecticGaugeCompatibility,
    ConservationGaugeUnification,
    compute_grammar_symmetry_mapping,
    verify_action_energy_consistency,
    compute_noether_gauge_decomposition,
    compute_gauge_conservation_coupling,
    verify_symplectic_gauge_compatibility,
    run_conservation_gauge_unification,
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
    "extract_phase_space_point",
    "symplectic_form_matrix",
    "substrate_hamiltonian",
    "background_potential",
    "hamiltonian_vector_field",
    "poisson_bracket",
    "canonical_bracket_table",
    "liouville_divergence",
    "verify_canonical_structure",
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
    "REGIME_DOMINANCE_THRESHOLD",
    "REGIME_STRONG_THRESHOLD",
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
    "GAMMA_C",
    "ORDER_PARAMETER_NOISE_FLOOR",
    "CHIRALITY_THRESHOLD",
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
