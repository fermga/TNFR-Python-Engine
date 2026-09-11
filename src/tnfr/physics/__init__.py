"""TNFR physics package — structural diagnostics and scoped models.

This package exposes read-only telemetry, model-specific certificates and
auxiliary dynamics tied to the TNFR nodal equation.  A curated public surface
is re-exported here; module-specific public names remain available from their
defining modules.  Shared numerical primitives are centralized in private
helper modules instead of duplicating them across those implementations.

Canonical Structural Field Tetrad (Telemetry)
---------------------------------------------
The four read-only fields below form the canonical diagnostic interface:

1) Structural Potential (Φ_s)
   - Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α (α=2, inverse-square)
   - U6 drift π/2 and per-node magnitude π/4 are selected monitoring policies,
     not graph-independent bounds. Particular ζ sums do not derive them.

2) Phase Gradient (|∇φ|)
   - |∇φ|(i) = mean_{j∈N(i)} |wrap(φ_j − φ_i)| (circular differences)
   - Exact wrapped-angle bound π; π/16 is a selected early-warning policy
   - A measured synchronization onset near 0.29 is protocol/σ-dependent

3) Phase Curvature (K_φ)
   - K_φ(i) = φ_i − mean_circular_{j∈N(i)} φ_j (Laplacian-like curvature)
   - Exact wrapped-angle bound π; 0.9π is a selected warning margin

4) Coherence Length (ξ_C)
   - From spatial decay of local coherence correlations C(r) ~ exp(−r/ξ_C)
   - Finite-network diagnostic; divergence requires a finite-size limit

Physics Foundation
------------------
Nodal equation (per node):  ∂EPI/∂t = ν_f · ΔNFR(t)
Canonical glyphs are the exclusive semantic transformations at the operator
layer. Declared domain solvers may advance EPI only through the shared nodal
integrator from an explicit DeltaNFR with provenance and residual telemetry.
Telemetry functions are read-only. nu_f uses structural units Hz_str.

Modules
-------
fields : Centralized structural field computations and research utilities
    - compute_structural_potential, compute_phase_gradient,
      compute_phase_curvature, estimate_coherence_length
    - k_φ multiscale helpers; declared-cycle winding is exposed separately
    - historical Q is a continuous bilinear snapshot, not phase winding
interactions : Canonical operator sequences with telemetry guards
      - em_like, weak_like, strong_like, gravity_like
         (returning InteractionResult)
life : Assumption-explicit autopoietic diagnostics
    - detect_life_emergence, LifeTelemetry, declared threshold policies
cell : Compartment and membrane-pressure diagnostics
    - detect_cell_formation, CellTelemetry, measured membrane flux
phase_transition : Operational structural-symmetry transition diagnostics
    - Order parameter 𝒮, chirality χ, susceptibility, coherence length
    - Critical exponent measured as an observable (audit 2026: NOT the
      universal closed-form scale; the fitted exponent is protocol-dependent)
    - No universal transition order or ξ_C divergence is asserted
event_remesh_refinement : Finite three-mesh event/REMESH observations
    - Exact persistent-node checkpoint errors, executed ZHIR gates and modal factors
    - Excludes solver order, mesh convergence and mixed schedule/REMESH gain
reversible_eigenmode_reference : Exact reversible single-eigenmode Euler theorem
    - Derives the reversible metric, eigenvalue, modal factors and error bounds
    - Certifies conditional exact-real convergence, not binary64 asymptotics
runtime_eigenmode_reference : Executed reversible single-eigenmode binding
    - Separates exact pressure, held-input and combined binary64 defects
    - Propagates off-mode defects through complete Euler matrices
    - Excludes runtime convergence, solver order and common causal provenance
event_remesh_reference : Exact P2 event/REMESH reference family
    - Rational continuous/Euler error enclosure and subdivision improvement
    - Exact beta-scaled ideal REMESH error plus bounded runtime residual
    - Excludes generic or binary64 asymptotic convergence
remesh_history_stability : Exact finite uniform delayed-history theorem
    - Stationary-weighted Jensen disagreement balance
    - Temporal mixing for 0 < alpha < 1; pure-delay cycles at alpha = 1
    - Excludes clipping, changing support/metric and runtime identification
remesh_schedule_policy_stability : Uniform exact REMESH/schedule policy theorem
    - Common schedule gain q gives prefix bound 1 and block gain q
    - Repeated spatial-disagreement decay for q < 1 over L = max_delay + 1
    - Conditional on exact maps; excludes binary64 runtime identification
remesh_schedule_relative_defect_stability : Robust exact policy envelope
    - Signed pre-schedule defect delta <= eta J gives q_eff = q(1 + eta)
    - Reuses the common-q theorem; q_eff < 1 gives geometric disagreement decay
    - Assumes a uniform defect bound; excludes runtime forward invariance
binary64_remesh_relative_defect : Exact runtime-rounding boundary
    - Replays one production pair and exposes its signed relative defect
    - Certifies the alpha=1 bounded hard-clip REMESH-only class with eta=0
    - Excludes schedule families, repeated execution and future behavior
binary64_p2_reception_stability : Global restricted P2 kernel composition
    - Half-Reception sends every finite represented pair in its interval to consensus
    - Composes q=0 with the alpha=1, eta=0 REMESH class
    - Excludes complete EN stages, grammar and live graph execution
runtime_p2_reception_stage : Finite executor binding for the P2 kernel
    - Binds one sealed two-phase EN event and its endpoints to the q=0 kernel
    - Verifies P2 neighbours, exact half mix, hard interval and metric ray
    - Excludes REMESH graph binding and repeated/future runtime stability
runtime_p2_reception_remesh_sequence : Finite causal P2 extinction binding
    - Binds each executed EN stage and delayed REMESH to one causal graph trace
    - Verifies active-history disagreement extinction after tau_global + 1 cycles
    - Excludes unobserved repetition, future runtime and auxiliary-state stability
runtime_p2_reception_remesh_policy : Transactional finite P2 policy invocation
    - Revalidates current P2/REMESH preconditions before every finite invocation
    - Encloses execution and post-certification in one graph transaction
    - Excludes future, unobserved-repetition and auxiliary-state stability
runtime_remesh_history_stability : Executed runtime/companion bridge
    - Exact signed binary64 rounding and clipping residual decomposition
    - Lifted one-step augmented-energy balances and sufficient lower bounds
    - Excludes live history advance, repeated stability and schedule composition
remesh_schedule_stability : Exact REMESH-head/schedule-head balance
    - Separates raw, clipping and schedule disagreement-energy defects
    - Retains the schedule-gain slack and a sufficient one-step lower bound
    - Excludes executable provenance, repetition and future stability
runtime_remesh_schedule_stability : Adjacent-cycle runtime/history binding
    - Identifies the next represented schedule and recorded history append
    - Telescopes exact finite augmented-energy balances in one metric
    - Excludes shared execution provenance, global gain and repetition
runtime_remesh_schedule_relative_defect : Finite robust-policy verification
    - Checks signed relative defects, represented gains and vector envelopes
    - Applies the complete-block q_eff endpoint bound to one causal block
    - Excludes forward invariance, repetition and future binary64 stability
runtime_remesh_schedule_block_margin : Causal finite-block margin observation
    - Normalizes a contiguous block lower bound by its initial energy
    - Binds the block to one graph-owned causal finite execution
    - Excludes uniform class coercivity, repetition and future stability

See Also
--------
tnfr.operators.grammar : Unified Grammar (U1–U6) and validations
tnfr.dynamics           : Nodal equation integration utilities
docs/STRUCTURAL_FIELDS_TETRAD.md : Canonical tetrad documentation
AGENTS.md               : Canonical invariants and field promotions
ARCHITECTURE.md          : Package ownership and dependency boundaries

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
>>> assert drift < 1.571, "Selected U6 drift policy exceeded (π/2)"

"""

from .cell import (
    CellTelemetry,
    MembraneFluxResult,
    MembraneNodeFlux,
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
from .coherence_geometry import (
    CoherenceLevelSetCertificate,
    CrossPolytopeStratification,
    FixedCapacityCoherenceLevelSetCertificate,
    NetworkCoherenceLevelSetCertificate,
    coherence_level_set_geometry,
    fixed_capacity_coherence_level_set_geometry,
    network_coherence_level_set_geometry,
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
from .event_duration import (
    ContinuousRelaxationDurationDiagnostic,
    diagnose_continuous_relaxation_duration,
)
from .fields import (
    CoherenceLengthEstimate,
    compute_k_phi_multiscale_variance,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_phase_winding,
    compute_structural_potential,
    estimate_coherence_length,
    estimate_coherence_length_with_provenance,
    fit_k_phi_asymptotic_alpha,
    k_phi_multiscale_safety,
)
from .observability import (
    EpiDiffusionReconstructionCertificate,
    LinearObservabilityCertificate,
    LocalObserverCertificate,
    ObservationSignature,
    finite_difference_observer_certificate,
    epi_diffusion_reconstruction_certificate,
    linear_observability_certificate,
    minimal_distinguishing_channels,
    observation_signature,
    observer_ablation_ranks,
    tetrad_observation_channels,
    tetrad_observation_vector,
    transform_linear_observer,
)
from .reduction_certificates import (
    ComposedReductionCertificate,
    KronReductionCertificate,
    ObserverTransportCertificate,
    composed_reduction_certificate,
    kron_reduction_certificate,
    observer_transport_certificate,
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
    OPERATOR_POLICY_MULTIPLIERS,
    EnergyClass,
    LyapunovSpectralSummary,
    OperatorLyapunovBound,
    OperatorLyapunovVerification,
    OperatorPolicyComparison,
    OperatorPolicyMultiplier,
    OperatorPolicySpectralContext,
    SequenceLyapunovProof,
    SequencePolicyEvaluation,
    SpectralGapAnalysis,
    U2PolicyRole,
    analyze_operator_convergence,
    analyze_operator_policy_context,
    analyze_spectral_gap,
    compute_operator_energy_bound,
    compute_operator_policy_delta,
    compute_sequence_energy_bound,
    compute_sequence_policy_score,
    compare_operator_energy_to_policy,
    evaluate_sequence_policy,
    get_bound,
    get_policy_multiplier,
    prove_sequence_lyapunov,
    verify_operator_lyapunov,
)
from .metriplectic import (
    MetriplecticProductCertificate,
    verify_metriplectic_product,
)
from .multiscale_coherence import (
    U5CoherenceAssessment,
    assess_u5_parent_child_coherence,
)
from .mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerEvidence,
    MutationTriggerInputError,
    certify_mutation_trigger,
)
from .nonnormal_prediction import (
    NonnormalPredictionCertificate,
    NonnormalPredictorRecord,
    benchmark_nonnormal_prediction,
    deterministic_directed_family,
    measure_nonnormal_pressure_prediction,
)
from .operator_quotient import (
    OperatorQuotientCertificate,
    certify_operator_quotient,
)
from .phase_quotient import (
    PhaseNodalCoarseGrainingCertificate,
    certify_phase_nodal_coarse_graining,
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
from .phase_scaling import (
    PhaseScalingDiagnostic,
    SizePowerLawFit,
    analyze_phase_finite_size_scaling,
)
from .spectral_conservation import (
    SpectralConservationBalance,
    SpectralLyapunovResult,
    SpectralStructuralEnergyResult,
    SpectralSectorDecomposition,
    SpectralWardIdentity,
    classify_spectral_modes,
    compute_spectral_energy_conservation,
    compute_spectral_lyapunov,
    compute_spectral_structural_energy,
    compute_spectral_ward_identity,
    decompose_spectral_sectors,
    verify_spectral_conservation_balance,
)
from .structural_morphism import (
    EpiCoarseGrainingCertificate,
    certify_epi_coarse_graining,
)
from .structural_state_distance import (
    StructuralChannelScales,
    StructuralStateDistanceCertificate,
    circular_phase_distance,
    fixed_topology_structural_state_distance,
)
from .temporal_identifiability import (
    NearestSignatureIdentification,
    SignatureMatrixCertificate,
    SignatureNoiseMarginCertificate,
    TemporalOperatorIdentifiabilityCertificate,
    certify_signature_noise_margin,
    certify_temporal_signature_matrix,
    identify_nearest_signature,
    probe_canonical_operator_identifiability,
)
from .topology_transitions import (
    NodalTopologySnapshot,
    NodalTopologyStep,
    NodalTopologyTransitionCertificate,
    capture_nodal_topology_snapshot,
    detect_nodal_topology_transitions,
)
from .structural_diffusion import (
    DiscreteModeCertificate,
    EulerRelaxationWindowDiagnostic,
    HeterogeneousDiffusionStabilityCertificate,
    OverdampedRegimeCertificate,
    RandomWalkCertificate,
    StructuralDiffusionCertificate,
    StructuralFlowCertificate,
    StructuralStabilityCertificate,
    SwitchingDiffusionStabilityCertificate,
    TimeVaryingDiffusionStabilityBound,
    commute_time,
    current_divergence,
    degree_weighted_total,
    derive_time_varying_diffusion_stability_bound,
    diagnose_euler_relaxation_window,
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
    structural_eigenvalues,
    structural_eigenmodes,
    structural_field,
    verify_heterogeneous_diffusion_stability,
    verify_switching_diffusion_stability,
    verify_discrete_modes,
    verify_overdamped_regime,
    verify_structural_diffusion,
    verify_structural_flow,
    verify_structural_random_walk,
    verify_structural_stability,
)
from .core_research_integration import (
    CoreResearchIntegrationCertificate,
    certify_core_research_integration,
)
from .core_research_trajectory import (
    CoreResearchRefinementComparison,
    CoreResearchRefinementSample,
    CoreResearchTrajectoryCertificate,
    CoreResearchTrajectoryIntervalCertificate,
    certify_core_research_trajectory,
    compare_core_research_trajectory_refinement,
)
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate,
    HybridEPIStabilityCertificate,
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
)
from .runtime_flow_stability import (
    NodalFlowIntervalCertificate,
    NodalFlowStateSnapshot,
    capture_nodal_flow_state,
    certify_observed_nodal_flow_interval,
)
from .event_remesh_refinement import (
    EventRemeshEPICheckpointObservation,
    EventRemeshMeshObservation,
    EventRemeshPersistentEPIError,
    EventRemeshThreeMeshModalObservation,
    EventRemeshThreeMeshRefinementObservation,
    EventRemeshThreeMeshZHIRObservation,
    observe_event_remesh_three_mesh_refinement,
)
from .reversible_eigenmode_reference import (
    ReversibleSingleEigenmodeEulerReferenceCertificate,
    certify_reversible_single_eigenmode_euler_reference,
)
from .runtime_eigenmode_reference import (
    ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
    ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
    observe_executed_reversible_single_eigenmode_euler_reference,
)
from .event_remesh_reference import (
    P2EventRemeshMeshReferenceObservation,
    P2EventRemeshReferenceFamilyObservation,
    observe_p2_event_remesh_reference_family,
)
from .remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    UniformRemeshHistoryTransitionObservation,
    certify_uniform_remesh_history_stability,
    observe_uniform_remesh_history_transition,
)
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
    certify_uniform_remesh_schedule_relative_defect_stability,
)
from .binary64_remesh_relative_defect import (
    Binary64RemeshPairRelativeDefectObservation,
    UniformAlphaOneHardClipRemeshClassCertificate,
    certify_alpha_one_hard_clip_remesh_class,
    observe_binary64_remesh_pair_relative_defect,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
    certify_p2_half_reception_remesh_stability,
)
from .runtime_p2_reception_stage import (
    ExecutedP2HalfReceptionStageCertificate,
    certify_executed_p2_half_reception_stage,
)
from .runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate,
    certify_executed_p2_half_reception_remesh_sequence,
)
from .runtime_p2_reception_remesh_policy import (
    execute_p2_half_reception_remesh_policy_invocation,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
    observe_runtime_remesh_history_bridge,
)
from .remesh_schedule_stability import (
    RemeshScheduleHistoryStabilityObservation,
    observe_remesh_schedule_history_transition,
)
from .runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
    RuntimeRemeshScheduleSequenceObservation,
    observe_runtime_remesh_schedule_sequence,
)
from .runtime_remesh_schedule_block_margin import (
    RuntimeRemeshScheduleBlockMarginObservation,
    observe_executed_event_remesh_block_margin,
)
from .runtime_remesh_schedule_relative_defect import (
    RuntimeRemeshScheduleRelativeDefectBlockObservation,
    observe_executed_event_remesh_relative_defect_block,
)
from .network_stage_stability import (
    AllTargetNeighborStageCertificate,
    AllTargetNeighborStageStep,
    NeighborStageDiffusionBridgeCertificate,
    certify_all_target_neighbor_stage,
    certify_reception_all_target_stage,
    certify_resonance_all_target_stage,
    compose_neighbor_stage_diffusion_stability,
)
from .reception_realization import (
    ReceptionEPIRealizationCertificate,
    certify_reception_epi_realization,
)
from .resonance_realization import (
    ResonanceEPIRealizationCertificate,
    certify_resonance_epi_realization,
)
from .winding_certificates import (
    WindingCertificate,
    WindingStepObservation,
    WindingWordObservation,
    certify_phase_winding,
    observe_winding_word,
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
    symplectic_pullback_residual,
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
    compute_historical_q_density,
    compute_symmetry_breaking_field,
    compute_topological_charge,
    compute_unified_field_suite,
)
from .emergent_particles import (
    EmergentParticle,
    WindingSector,
    classify_particle,
    classify_winding_sector,
    winding_number,
    winding_ring,
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
    "estimate_coherence_length_with_provenance",
    "CoherenceLengthEstimate",
    "compute_k_phi_multiscale_variance",
    "fit_k_phi_asymptotic_alpha",
    "k_phi_multiscale_safety",
    "compute_phase_winding",
    # --- Read-only field observability ---
    "EpiDiffusionReconstructionCertificate",
    "LinearObservabilityCertificate",
    "LocalObserverCertificate",
    "ObservationSignature",
    "linear_observability_certificate",
    "finite_difference_observer_certificate",
    "epi_diffusion_reconstruction_certificate",
    "observer_ablation_ranks",
    "tetrad_observation_channels",
    "tetrad_observation_vector",
    "observation_signature",
    "minimal_distinguishing_channels",
    "transform_linear_observer",
    # --- Structural reduction diagnostics ---
    "ObserverTransportCertificate",
    "KronReductionCertificate",
    "ComposedReductionCertificate",
    "observer_transport_certificate",
    "kron_reduction_certificate",
    "composed_reduction_certificate",
    # --- Branch-aware phase topology ---
    "WindingCertificate",
    "WindingStepObservation",
    "WindingWordObservation",
    "certify_phase_winding",
    "observe_winding_word",
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
    "MembraneNodeFlux",
    "MembraneFluxResult",
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
    # --- Constitutive coherence geometry ---
    "CoherenceLevelSetCertificate",
    "CrossPolytopeStratification",
    "FixedCapacityCoherenceLevelSetCertificate",
    "NetworkCoherenceLevelSetCertificate",
    "coherence_level_set_geometry",
    "fixed_capacity_coherence_level_set_geometry",
    "network_coherence_level_set_geometry",
    # --- Unified Complex Fields (Ψ = K_φ + i·J_φ) ---
    "compute_complex_geometric_field",
    "compute_field_magnitude",
    "compute_field_phase",
    "compute_chirality_field",
    "compute_symmetry_breaking_field",
    "compute_coherence_coupling_field",
    "compute_energy_density",
    "compute_action_density",
    "compute_historical_q_density",
    "compute_topological_charge",
    "compute_unified_field_suite",
    # --- Declared-cycle phase winding ---
    "WindingSector",
    "classify_winding_sector",
    "winding_number",
    "winding_ring",
    # Historical compatibility names
    "EmergentParticle",
    "classify_particle",
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
    "SpectralStructuralEnergyResult",
    "SpectralSectorDecomposition",
    "verify_spectral_conservation_balance",
    # --- Exact pure-EPI coarse-graining ---
    "EpiCoarseGrainingCertificate",
    "certify_epi_coarse_graining",
    # --- Fixed-branch circular phase coarse-graining ---
    "PhaseNodalCoarseGrainingCertificate",
    "certify_phase_nodal_coarse_graining",
    # --- Structural-state quotient metric ---
    "StructuralChannelScales",
    "StructuralStateDistanceCertificate",
    "circular_phase_distance",
    "fixed_topology_structural_state_distance",
    # --- Restricted executable S16 integration boundary ---
    "CoreResearchIntegrationCertificate",
    "certify_core_research_integration",
    "CoreResearchTrajectoryIntervalCertificate",
    "CoreResearchTrajectoryCertificate",
    "CoreResearchRefinementSample",
    "CoreResearchRefinementComparison",
    "certify_core_research_trajectory",
    "compare_core_research_trajectory_refinement",
    # --- Affine EPI-reset and hybrid flow/reset stability ---
    "AffineEPIJumpGainCertificate",
    "HybridEPIStabilityCertificate",
    "certify_affine_epi_jump_gain",
    "compose_hybrid_epi_stability",
    # --- Observed nodal-flow interval evidence ---
    "NodalFlowStateSnapshot",
    "NodalFlowIntervalCertificate",
    "capture_nodal_flow_state",
    "certify_observed_nodal_flow_interval",
    # --- Finite event/REMESH three-mesh observations ---
    "EventRemeshEPICheckpointObservation",
    "EventRemeshMeshObservation",
    "EventRemeshPersistentEPIError",
    "EventRemeshThreeMeshModalObservation",
    "EventRemeshThreeMeshRefinementObservation",
    "EventRemeshThreeMeshZHIRObservation",
    "observe_event_remesh_three_mesh_refinement",
    # --- Exact reversible single-eigenmode Euler reference ---
    "ReversibleSingleEigenmodeEulerReferenceCertificate",
    "certify_reversible_single_eigenmode_euler_reference",
    # --- Executed reversible single-eigenmode reference binding ---
    "ExecutedReversibleSingleEigenmodeEulerPartitionObservation",
    "ExecutedReversibleSingleEigenmodeEulerReferenceObservation",
    "observe_executed_reversible_single_eigenmode_euler_reference",
    # --- Exact P2 event/REMESH reference family ---
    "P2EventRemeshMeshReferenceObservation",
    "P2EventRemeshReferenceFamilyObservation",
    "observe_p2_event_remesh_reference_family",
    # --- Exact uniform delayed-REMESH history stability ---
    "UniformRemeshHistoryStabilityCertificate",
    "UniformRemeshHistoryTransitionObservation",
    "certify_uniform_remesh_history_stability",
    "observe_uniform_remesh_history_transition",
    # --- Uniform exact REMESH/schedule policy stability ---
    "UniformRemeshSchedulePolicyStabilityCertificate",
    "certify_uniform_remesh_schedule_policy_stability",
    # --- Relative-defect robust REMESH/schedule policy stability ---
    "UniformRemeshScheduleRelativeDefectStabilityCertificate",
    "certify_uniform_remesh_schedule_relative_defect_stability",
    # --- Exact binary64 REMESH relative-defect boundary ---
    "Binary64RemeshPairRelativeDefectObservation",
    "UniformAlphaOneHardClipRemeshClassCertificate",
    "certify_alpha_one_hard_clip_remesh_class",
    "observe_binary64_remesh_pair_relative_defect",
    # --- Global binary64 P2 half-Reception/REMESH kernel family ---
    "P2HalfReceptionRemeshStabilityCertificate",
    "certify_p2_half_reception_remesh_stability",
    # --- Finite executed P2 half-Reception stage binding ---
    "ExecutedP2HalfReceptionStageCertificate",
    "certify_executed_p2_half_reception_stage",
    # --- Finite causal P2 Reception/REMESH extinction binding ---
    "ExecutedP2HalfReceptionRemeshSequenceCertificate",
    "certify_executed_p2_half_reception_remesh_sequence",
    # --- Transactional finite P2 runtime policy invocation ---
    "execute_p2_half_reception_remesh_policy_invocation",
    # --- Executed REMESH runtime/companion bridge ---
    "RuntimeRemeshHistoryBridgeObservation",
    "observe_runtime_remesh_history_bridge",
    # --- Exact REMESH-to-schedule augmented-history balance ---
    "RemeshScheduleHistoryStabilityObservation",
    "observe_remesh_schedule_history_transition",
    # --- Adjacent-cycle runtime REMESH/schedule history binding ---
    "RuntimeRemeshScheduleBoundaryObservation",
    "RuntimeRemeshScheduleSequenceObservation",
    "observe_runtime_remesh_schedule_sequence",
    # --- Causally executed finite REMESH/schedule block margin ---
    "RuntimeRemeshScheduleBlockMarginObservation",
    "observe_executed_event_remesh_block_margin",
    # --- Finite causal relative-defect REMESH/schedule block ---
    "RuntimeRemeshScheduleRelativeDefectBlockObservation",
    "observe_executed_event_remesh_relative_defect_block",
    # --- Physical operator-event flow duration ---
    "ContinuousRelaxationDurationDiagnostic",
    "diagnose_continuous_relaxation_duration",
    # --- Repeated all-target EN/RA EPI-stage realization ---
    "AllTargetNeighborStageCertificate",
    "AllTargetNeighborStageStep",
    "NeighborStageDiffusionBridgeCertificate",
    "certify_all_target_neighbor_stage",
    "certify_reception_all_target_stage",
    "certify_resonance_all_target_stage",
    "compose_neighbor_stage_diffusion_stability",
    # --- Reception runtime-to-affine realization boundary ---
    "ReceptionEPIRealizationCertificate",
    "certify_reception_epi_realization",
    "ResonanceEPIRealizationCertificate",
    "certify_resonance_epi_realization",
    "compute_spectral_ward_identity",
    "compute_spectral_lyapunov",
    "compute_spectral_structural_energy",
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
    "symplectic_pullback_residual",
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
    "HeterogeneousDiffusionStabilityCertificate",
    "SwitchingDiffusionStabilityCertificate",
    "TimeVaryingDiffusionStabilityBound",
    "OverdampedRegimeCertificate",
    "DiscreteModeCertificate",
    "EulerRelaxationWindowDiagnostic",
    "StructuralStabilityCertificate",
    "RandomWalkCertificate",
    "StructuralFlowCertificate",
    "structural_diffusion_operator",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "degree_weighted_total",
    "derive_time_varying_diffusion_stability_bound",
    "diagnose_euler_relaxation_window",
    "verify_heterogeneous_diffusion_stability",
    "verify_switching_diffusion_stability",
    "structural_eigenvalues",
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
    # --- Auxiliary pure-gauge U(1) coordinates for Ψ = K_φ + i·J_φ ---
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
    # --- Legacy Yang-Mills-named finite snapshot diagnostics ---
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
    # --- Finite-size phase-scaling protocol ---
    "SizePowerLawFit",
    "PhaseScalingDiagnostic",
    "analyze_phase_finite_size_scaling",
    # --- Dynamic nodal-topology observations ---
    "NodalTopologySnapshot",
    "NodalTopologyStep",
    "NodalTopologyTransitionCertificate",
    "capture_nodal_topology_snapshot",
    "detect_nodal_topology_transitions",
    # --- U2 policy multipliers and independent spectral diagnostics ---
    "U2PolicyRole",
    "OperatorPolicyMultiplier",
    "OperatorPolicyComparison",
    "OperatorPolicySpectralContext",
    "SequencePolicyEvaluation",
    "OPERATOR_POLICY_MULTIPLIERS",
    "get_policy_multiplier",
    "compute_operator_policy_delta",
    "compare_operator_energy_to_policy",
    "compute_sequence_policy_score",
    "evaluate_sequence_policy",
    "analyze_operator_policy_context",
    # Historical compatibility names
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
    # --- Restricted dissipative-symplectic product ---
    "MetriplecticProductCertificate",
    "verify_metriplectic_product",
    # --- Explicit U5 parent/child coherence assessment ---
    "U5CoherenceAssessment",
    "assess_u5_parent_child_coherence",
    # --- Mutation prediction/observation certificate ---
    "MutationTriggerCertificate",
    "MutationTriggerEvidence",
    "MutationTriggerInputError",
    "certify_mutation_trigger",
    # --- Directed non-normal pressure prediction ---
    "NonnormalPredictorRecord",
    "NonnormalPredictionCertificate",
    "deterministic_directed_family",
    "measure_nonnormal_pressure_prediction",
    "benchmark_nonnormal_prediction",
    # --- Generic operator quotient closure ---
    "OperatorQuotientCertificate",
    "certify_operator_quotient",
    # --- Temporal operator identifiability ---
    "NearestSignatureIdentification",
    "SignatureMatrixCertificate",
    "SignatureNoiseMarginCertificate",
    "TemporalOperatorIdentifiabilityCertificate",
    "certify_signature_noise_margin",
    "certify_temporal_signature_matrix",
    "identify_nearest_signature",
    "probe_canonical_operator_identifiability",
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
