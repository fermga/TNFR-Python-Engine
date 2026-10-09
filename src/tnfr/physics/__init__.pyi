"""Generated typed re-exports for the lazy physics facade.

Regenerate with: python scripts/generate_physics_stub.py --write
The runtime _EXPORT_MODULES registry is the sole maintained source.
"""

# isort: skip_file
# Preserve the runtime public export order.

from .form_geometry import RegionalAffineClosure as RegionalAffineClosure
from .form_geometry import RegionalFormObservation as RegionalFormObservation
from .form_geometry import RegionalFormRegion as RegionalFormRegion
from .form_geometry import (
    derive_regional_affine_closure as derive_regional_affine_closure,
)
from .form_geometry import observe_regional_form as observe_regional_form
from .source_relative_form import (
    SourceRelativeFormObservation as SourceRelativeFormObservation,
)
from .source_relative_form import (
    observe_source_relative_form as observe_source_relative_form,
)
from .fields import compute_structural_potential as compute_structural_potential
from .fields import compute_phase_gradient as compute_phase_gradient
from .fields import compute_phase_curvature as compute_phase_curvature
from .fields import estimate_coherence_length as estimate_coherence_length
from .fields import (
    estimate_coherence_length_with_provenance as estimate_coherence_length_with_provenance,
)
from .fields import CoherenceLengthEstimate as CoherenceLengthEstimate
from .fields import (
    compute_k_phi_multiscale_variance as compute_k_phi_multiscale_variance,
)
from .fields import fit_k_phi_asymptotic_alpha as fit_k_phi_asymptotic_alpha
from .fields import k_phi_multiscale_safety as k_phi_multiscale_safety
from .fields import compute_phase_winding as compute_phase_winding
from .observability import (
    EpiDiffusionReconstructionCertificate as EpiDiffusionReconstructionCertificate,
)
from .observability import (
    LinearObservabilityCertificate as LinearObservabilityCertificate,
)
from .observability import LocalObserverCertificate as LocalObserverCertificate
from .observability import ObservationSignature as ObservationSignature
from .observability import (
    linear_observability_certificate as linear_observability_certificate,
)
from .observability import (
    finite_difference_observer_certificate as finite_difference_observer_certificate,
)
from .observability import (
    epi_diffusion_reconstruction_certificate as epi_diffusion_reconstruction_certificate,
)
from .observability import observer_ablation_ranks as observer_ablation_ranks
from .observability import tetrad_observation_channels as tetrad_observation_channels
from .observability import tetrad_observation_vector as tetrad_observation_vector
from .observability import observation_signature as observation_signature
from .observability import (
    minimal_distinguishing_channels as minimal_distinguishing_channels,
)
from .observability import transform_linear_observer as transform_linear_observer
from .reduction_certificates import (
    ObserverTransportCertificate as ObserverTransportCertificate,
)
from .reduction_certificates import KronReductionCertificate as KronReductionCertificate
from .reduction_certificates import (
    ComposedReductionCertificate as ComposedReductionCertificate,
)
from .reduction_certificates import (
    observer_transport_certificate as observer_transport_certificate,
)
from .reduction_certificates import (
    kron_reduction_certificate as kron_reduction_certificate,
)
from .reduction_certificates import (
    composed_reduction_certificate as composed_reduction_certificate,
)
from .epi_memory import EpiMemoryObservation as EpiMemoryObservation
from .epi_memory import EpiMemorySample as EpiMemorySample
from .epi_memory import observe_epi_memory as observe_epi_memory
from .p5_memory_truncation import (
    P5MemoryTruncationReference as P5MemoryTruncationReference,
)
from .p5_memory_truncation import P5MemoryTruncationSample as P5MemoryTruncationSample
from .p5_memory_truncation import (
    bound_p5_memory_truncation as bound_p5_memory_truncation,
)
from .p5_hidden_form import P5HiddenFormBound as P5HiddenFormBound
from .p5_hidden_form import P5HiddenFormSample as P5HiddenFormSample
from .p5_hidden_form import bound_p5_hidden_form as bound_p5_hidden_form
from .p2_phase_form import P2PhaseFormModel as P2PhaseFormModel
from .p2_phase_form import P2PhaseFormStep as P2PhaseFormStep
from .p2_phase_form import derive_p2_phase_form_model as derive_p2_phase_form_model
from .p2_phase_form import propose_p2_phase_form_step as propose_p2_phase_form_step
from .p5_reduction import P5ReducedState as P5ReducedState
from .p5_reduction import P5ReductionGeometry as P5ReductionGeometry
from .p5_reduction import P5RemeshReduction as P5RemeshReduction
from .p5_reduction import reduce_p5_state as reduce_p5_state
from .p5_reduction import p5_reduction_geometry as p5_reduction_geometry
from .p5_reduction import observe_p5_remesh_reduction as observe_p5_remesh_reduction
from .winding_certificates import WindingCertificate as WindingCertificate
from .winding_certificates import WindingStepObservation as WindingStepObservation
from .winding_certificates import WindingWordObservation as WindingWordObservation
from .winding_certificates import certify_phase_winding as certify_phase_winding
from .winding_certificates import observe_winding_word as observe_winding_word
from .coupling_winding import CouplingGapStep as CouplingGapStep
from .coupling_winding import observe_coupling_gap_step as observe_coupling_gap_step
from .capacity_localization import CycleCapacityBalance as CycleCapacityBalance
from .capacity_localization import (
    observe_cycle_capacity_balance as observe_cycle_capacity_balance,
)
from .cycle_memory_relaxation import (
    CycleMemoryRelaxationReference as CycleMemoryRelaxationReference,
)
from .cycle_memory_relaxation import (
    certify_cycle_memory_relaxation as certify_cycle_memory_relaxation,
)
from .cycle_support_dynamics import CycleSupportBalance as CycleSupportBalance
from .cycle_support_dynamics import CycleSupportEuler as CycleSupportEuler
from .cycle_support_dynamics import CycleSupportReset as CycleSupportReset
from .cycle_support_dynamics import (
    observe_cycle_support_balance as observe_cycle_support_balance,
)
from .cycle_support_dynamics import (
    observe_cycle_support_euler as observe_cycle_support_euler,
)
from .cycle_support_dynamics import (
    observe_cycle_support_reset as observe_cycle_support_reset,
)
from .support_transport import SupportTransportSnapshot as SupportTransportSnapshot
from .support_transport import SupportTransportReset as SupportTransportReset
from .support_transport import SupportTransportEuler as SupportTransportEuler
from .support_transport import (
    SupportTransportClippedFlow as SupportTransportClippedFlow,
)
from .support_transport import observe_support_transport as observe_support_transport
from .support_transport import (
    observe_support_transport_reset as observe_support_transport_reset,
)
from .support_transport import (
    observe_support_transport_euler as observe_support_transport_euler,
)
from .support_transport import (
    observe_support_transport_clipped_flow as observe_support_transport_clipped_flow,
)
from .forced_support import ForcedSupportBalance as ForcedSupportBalance
from .forced_support import ForcedSupportEvent as ForcedSupportEvent
from .forced_support import ForcedSupportJumpEnergy as ForcedSupportJumpEnergy
from .forced_support import observe_forced_support_event as observe_forced_support_event
from .forced_support import ForcedSupportPattern as ForcedSupportPattern
from .forced_support import ForcedSupportReset as ForcedSupportReset
from .forced_support import ForcedSupportResetEnergy as ForcedSupportResetEnergy
from .forced_support import (
    observe_forced_support_pattern as observe_forced_support_pattern,
)
from .forced_support import observe_forced_support_reset as observe_forced_support_reset
from .forced_support import ForcedSupportState as ForcedSupportState
from .forced_support import ForcedSupportStep as ForcedSupportStep
from .forced_support import ForcedSupportTarget as ForcedSupportTarget
from .forced_support import (
    derive_forced_support_balance as derive_forced_support_balance,
)
from .forced_support import observe_forced_support_state as observe_forced_support_state
from .forced_support import observe_forced_support_step as observe_forced_support_step
from .forced_support import (
    observe_forced_support_target as observe_forced_support_target,
)
from .forcing_realization import NonEpiForcingObservation as NonEpiForcingObservation
from .forcing_realization import ForcingDirichletBalance as ForcingDirichletBalance
from .forcing_realization import ForcingMeanBalance as ForcingMeanBalance
from .forcing_realization import ForcingCapacityDifference as ForcingCapacityDifference
from .forcing_realization import (
    observe_forcing_capacity_difference as observe_forcing_capacity_difference,
)
from .forcing_realization import capture_non_epi_forcing as capture_non_epi_forcing
from .forcing_realization import decompose_non_epi_forcing as decompose_non_epi_forcing
from .forcing_realization import (
    observe_forcing_dirichlet_balance as observe_forcing_dirichlet_balance,
)
from .forcing_realization import (
    observe_forcing_mean_balance as observe_forcing_mean_balance,
)
from .capacity_feedback import P2CapacityFeedbackBound as P2CapacityFeedbackBound
from .capacity_feedback import (
    CapacityIntervalRecoveryBound as CapacityIntervalRecoveryBound,
)
from .capacity_feedback import (
    derive_capacity_interval_recovery_bound as derive_capacity_interval_recovery_bound,
)
from .capacity_feedback import P2CapacityFeedbackCycle as P2CapacityFeedbackCycle
from .capacity_feedback import (
    P2CapacityFeedbackReference as P2CapacityFeedbackReference,
)
from .capacity_feedback import bound_p2_capacity_feedback as bound_p2_capacity_feedback
from .capacity_feedback import (
    derive_p2_capacity_feedback as derive_p2_capacity_feedback,
)
from .capacity_feedback import (
    observe_p2_capacity_feedback_cycle as observe_p2_capacity_feedback_cycle,
)
from .capacity_feedback import (
    P2Binary64CouplingObservation as P2Binary64CouplingObservation,
)
from .capacity_feedback import (
    P2Binary64CouplingReference as P2Binary64CouplingReference,
)
from .capacity_feedback import (
    derive_p2_binary64_coupling_lattice as derive_p2_binary64_coupling_lattice,
)
from .capacity_feedback import (
    observe_p2_binary64_coupling_lattice as observe_p2_binary64_coupling_lattice,
)
from .coupling_support import CompatibleCapacityBalance as CompatibleCapacityBalance
from .coupling_support import CouplingSupportObservation as CouplingSupportObservation
from .coupling_support import (
    derive_compatible_capacity_balance as derive_compatible_capacity_balance,
)
from .coupling_support import observe_coupling_support as observe_coupling_support
from .coupling_support import AntipodalRegionPhaseBalance as AntipodalRegionPhaseBalance
from .coupling_support import (
    AntipodalRegionPhaseResponse as AntipodalRegionPhaseResponse,
)
from .coupling_support import (
    derive_antipodal_region_phase_balance as derive_antipodal_region_phase_balance,
)
from .coupling_support import (
    observe_antipodal_region_phase_response as observe_antipodal_region_phase_response,
)
from .phase_response import PhaseResponseReference as PhaseResponseReference
from .phase_response import derive_phase_response as derive_phase_response
from .coupling_winding import C6WindingPhaseReference as C6WindingPhaseReference
from .coupling_winding import C6WindingPhaseObservation as C6WindingPhaseObservation
from .coupling_winding import (
    derive_c6_winding_phase_response as derive_c6_winding_phase_response,
)
from .coupling_winding import (
    observe_c6_winding_phase_response as observe_c6_winding_phase_response,
)
from .coupling_winding import C6WindingJointDomain as C6WindingJointDomain
from .coupling_winding import C6WindingJointStep as C6WindingJointStep
from .coupling_winding import (
    derive_c6_winding_joint_domain as derive_c6_winding_joint_domain,
)
from .coupling_winding import (
    observe_c6_winding_joint_domain as observe_c6_winding_joint_domain,
)
from .coupling_winding import C6WindingDefect as C6WindingDefect
from .coupling_winding import C6WindingDefectPrefix as C6WindingDefectPrefix
from .coupling_winding import C6WindingUniformDefectBound as C6WindingUniformDefectBound
from .coupling_winding import C6WindingPairingReference as C6WindingPairingReference
from .coupling_winding import C6WindingPairingObservation as C6WindingPairingObservation
from .coupling_winding import c6_centered_opposite_pairs as c6_centered_opposite_pairs
from .coupling_winding import derive_c6_winding_pairing as derive_c6_winding_pairing
from .coupling_winding import observe_c6_winding_pairing as observe_c6_winding_pairing
from .binary64_nodal_flow import Binary64AdditionCell as Binary64AdditionCell
from .nodal_remainder import NodalRemainderPrefix as NodalRemainderPrefix
from .nodal_remainder import NodalRemainderSequence as NodalRemainderSequence
from .nodal_remainder import (
    observe_nodal_remainder_sequence as observe_nodal_remainder_sequence,
)
from .nodal_remainder import NodalRemainderCellHorizon as NodalRemainderCellHorizon
from .nodal_remainder import (
    derive_nodal_remainder_cell_horizon as derive_nodal_remainder_cell_horizon,
)
from .nodal_remainder import NodalRemainderCellExit as NodalRemainderCellExit
from .nodal_remainder import (
    observe_nodal_remainder_cell_exit as observe_nodal_remainder_cell_exit,
)
from .nodal_remainder import NodalRemainderItineraryCell as NodalRemainderItineraryCell
from .nodal_remainder import NodalRemainderItinerary as NodalRemainderItinerary
from .nodal_remainder import (
    derive_nodal_remainder_itinerary as derive_nodal_remainder_itinerary,
)
from .binary64_pressure_equilibrium import (
    Binary64PressureEquilibriumRow as Binary64PressureEquilibriumRow,
)
from .binary64_pressure_equilibrium import (
    Binary64C6PressureEquilibriumObstruction as Binary64C6PressureEquilibriumObstruction,
)
from .binary64_pressure_equilibrium import (
    derive_binary64_c6_pressure_equilibrium_obstruction as derive_binary64_c6_pressure_equilibrium_obstruction,
)
from .nodal_remainder_pressure import (
    NodalRemainderPressureReadout as NodalRemainderPressureReadout,
)
from .nodal_remainder_pressure import (
    observe_nodal_remainder_pressure_readout as observe_nodal_remainder_pressure_readout,
)
from .nodal_remainder_pressure import (
    PeriodicPhaseSourceBudget as PeriodicPhaseSourceBudget,
)
from .nodal_remainder_pressure import (
    PeriodicPhaseSourceCompensation as PeriodicPhaseSourceCompensation,
)
from .nodal_remainder_pressure import (
    derive_periodic_phase_source_budget as derive_periodic_phase_source_budget,
)
from .nodal_remainder_pressure import (
    observe_periodic_phase_source_compensation as observe_periodic_phase_source_compensation,
)
from .nodal_remainder_pressure import (
    FiniteNodalPressureDrift as FiniteNodalPressureDrift,
)
from .nodal_remainder_pressure import (
    observe_finite_nodal_pressure_drift as observe_finite_nodal_pressure_drift,
)
from .nodal_remainder_pressure import NodalAreaCrossing as NodalAreaCrossing
from .nodal_remainder_pressure import NodalAreaCrossings as NodalAreaCrossings
from .nodal_remainder_pressure import (
    derive_nodal_area_crossings as derive_nodal_area_crossings,
)
from .nodal_remainder_pressure import TwoLevelNodalReturn as TwoLevelNodalReturn
from .nodal_remainder_pressure import (
    derive_two_level_nodal_return as derive_two_level_nodal_return,
)
from .nodal_remainder_pressure import FiniteLevelNodalReturn as FiniteLevelNodalReturn
from .nodal_remainder_pressure import (
    derive_finite_level_nodal_return as derive_finite_level_nodal_return,
)
from .nodal_remainder_pressure import (
    NodalRemainderCycleGradient as NodalRemainderCycleGradient,
)
from .nodal_remainder_pressure import (
    observe_nodal_remainder_cycle_gradient as observe_nodal_remainder_cycle_gradient,
)
from .c6_pressure_lattice import C6PressureLatticeRow as C6PressureLatticeRow
from .c6_pressure_lattice import (
    C6PressureLatticeReference as C6PressureLatticeReference,
)
from .c6_pressure_lattice import (
    C6PressureLatticeObservation as C6PressureLatticeObservation,
)
from .c6_pressure_lattice import (
    derive_c6_pressure_lattice as derive_c6_pressure_lattice,
)
from .c6_pressure_lattice import (
    observe_c6_pressure_lattice as observe_c6_pressure_lattice,
)
from .c6_pressure_lattice import C6PressureSignSector as C6PressureSignSector
from .c6_pressure_lattice import C6PressureSectorExit as C6PressureSectorExit
from .c6_pressure_lattice import (
    derive_c6_pressure_sign_sector as derive_c6_pressure_sign_sector,
)
from .c6_pressure_lattice import (
    observe_c6_pressure_sector_exit as observe_c6_pressure_sector_exit,
)
from .c6_pressure_lattice import C6FrozenPressureStencil as C6FrozenPressureStencil
from .c6_pressure_lattice import (
    observe_c6_frozen_pressure_stencil as observe_c6_frozen_pressure_stencil,
)
from .c6_phase_orbit import C6CouplingCoherencePhaseStep as C6CouplingCoherencePhaseStep
from .c6_phase_orbit import (
    C6CouplingCoherencePhaseOrbit as C6CouplingCoherencePhaseOrbit,
)
from .c6_phase_orbit import (
    observe_c6_coupling_coherence_phase_step as observe_c6_coupling_coherence_phase_step,
)
from .c6_phase_orbit import (
    derive_c6_coupling_coherence_phase_orbit as derive_c6_coupling_coherence_phase_orbit,
)
from .c6_carried_profile import C6CarriedProfile as C6CarriedProfile
from .c6_carried_profile import derive_c6_carried_profile as derive_c6_carried_profile
from .c6_carried_profile import C6CarriedProfileStep as C6CarriedProfileStep
from .c6_carried_profile import (
    observe_c6_carried_profile_step as observe_c6_carried_profile_step,
)
from .c6_carried_tube import C6CarriedContraction as C6CarriedContraction
from .c6_carried_tube import (
    derive_c6_carried_contraction as derive_c6_carried_contraction,
)
from .c6_carried_tube import C6CarriedTube as C6CarriedTube
from .c6_carried_tube import derive_c6_carried_tube as derive_c6_carried_tube
from .c6_carried_tube import C6CarriedBandHorizon as C6CarriedBandHorizon
from .c6_carried_tube import (
    derive_c6_carried_band_horizon as derive_c6_carried_band_horizon,
)
from .c6_carried_tube import C6CarriedCutExclusion as C6CarriedCutExclusion
from .c6_carried_tube import (
    observe_c6_carried_cut_exclusion as observe_c6_carried_cut_exclusion,
)
from .c6_carried_closure import C6CarriedClosure as C6CarriedClosure
from .c6_carried_closure import derive_c6_carried_closure as derive_c6_carried_closure
from .c6_carried_balance import C6CarriedPressurePoint as C6CarriedPressurePoint
from .c6_carried_balance import (
    observe_c6_carried_pressure_point as observe_c6_carried_pressure_point,
)
from .c6_carried_balance import C6CarriedPressureBalance as C6CarriedPressureBalance
from .c6_carried_balance import (
    derive_c6_carried_pressure_balance as derive_c6_carried_pressure_balance,
)
from .c6_carried_passage import (
    C6CarriedPositivePressurePassage as C6CarriedPositivePressurePassage,
)
from .c6_carried_passage import (
    derive_c6_carried_positive_pressure_passage as derive_c6_carried_positive_pressure_passage,
)
from .c6_carried_cell_escape import (
    C6CarriedCompleteCellObstruction as C6CarriedCompleteCellObstruction,
)
from .c6_carried_cell_escape import (
    derive_c6_carried_complete_cell_obstruction as derive_c6_carried_complete_cell_obstruction,
)
from .c6_carried_cell_escape import (
    C6CarriedCompleteCellEscape as C6CarriedCompleteCellEscape,
)
from .c6_carried_cell_escape import (
    observe_c6_carried_complete_cell_escape as observe_c6_carried_complete_cell_escape,
)
from .c6_carried_cell_graph import C6CarriedCellGraph as C6CarriedCellGraph
from .c6_carried_cell_graph import (
    derive_c6_carried_cell_graph as derive_c6_carried_cell_graph,
)
from .c6_carried_viability import C6CarriedViabilityBox as C6CarriedViabilityBox
from .c6_carried_viability import (
    C6CarriedViabilityIteration as C6CarriedViabilityIteration,
)
from .c6_carried_viability import C6CarriedViability as C6CarriedViability
from .c6_carried_viability import (
    derive_c6_carried_viability as derive_c6_carried_viability,
)
from .c6_carried_viability import C6CarriedForwardZone as C6CarriedForwardZone
from .c6_carried_viability import C6CarriedPairBarrier as C6CarriedPairBarrier
from .c6_carried_viability import C6CarriedForwardIteration as C6CarriedForwardIteration
from .c6_carried_viability import C6CarriedForwardEnvelope as C6CarriedForwardEnvelope
from .c6_carried_viability import (
    derive_c6_carried_forward_envelope as derive_c6_carried_forward_envelope,
)
from .c6_carried_viability import C6CarriedPredecessorLayer as C6CarriedPredecessorLayer
from .c6_carried_viability import C6CarriedPredecessors as C6CarriedPredecessors
from .c6_carried_viability import (
    derive_c6_carried_predecessors as derive_c6_carried_predecessors,
)
from .c6_carried_viability import C6CarriedRegionIteration as C6CarriedRegionIteration
from .c6_carried_viability import C6CarriedRegionExclusion as C6CarriedRegionExclusion
from .c6_carried_viability import C6CarriedRegionExclusions as C6CarriedRegionExclusions
from .c6_carried_viability import (
    derive_c6_carried_region_exclusions as derive_c6_carried_region_exclusions,
)
from .c6_carried_viability import (
    C6CarriedReachableIteration as C6CarriedReachableIteration,
)
from .c6_carried_viability import (
    C6CarriedReachableEnvelope as C6CarriedReachableEnvelope,
)
from .c6_carried_viability import (
    derive_c6_carried_reachable_envelope as derive_c6_carried_reachable_envelope,
)
from .c6_carried_excursion import C6CarriedLinearExtremum as C6CarriedLinearExtremum
from .c6_carried_excursion import C6CarriedExcursionIngress as C6CarriedExcursionIngress
from .c6_carried_excursion import (
    C6CarriedExcursionExclusion as C6CarriedExcursionExclusion,
)
from .c6_carried_excursion import (
    derive_c6_carried_excursion_exclusion as derive_c6_carried_excursion_exclusion,
)
from .c6_carried_excursion import (
    C6CarriedExcursionTransition as C6CarriedExcursionTransition,
)
from .c6_carried_excursion import (
    C6CarriedModeExcursionExclusion as C6CarriedModeExcursionExclusion,
)
from .c6_carried_excursion import (
    derive_c6_carried_mode_excursion_exclusion as derive_c6_carried_mode_excursion_exclusion,
)
from .c6_carried_return import C6CarriedReturnTransition as C6CarriedReturnTransition
from .c6_carried_return import C6CarriedReturnIteration as C6CarriedReturnIteration
from .c6_carried_return import C6CarriedReturnEnvelope as C6CarriedReturnEnvelope
from .c6_carried_return import (
    derive_c6_carried_return_envelope as derive_c6_carried_return_envelope,
)
from .c6_carried_return import (
    C6CarriedReturnRegionExclusion as C6CarriedReturnRegionExclusion,
)
from .c6_carried_return import (
    C6CarriedReturnRegionExclusions as C6CarriedReturnRegionExclusions,
)
from .c6_carried_return import (
    derive_c6_carried_return_region_exclusions as derive_c6_carried_return_region_exclusions,
)
from .c6_carried_return import (
    C6CarriedReturnUnionExclusion as C6CarriedReturnUnionExclusion,
)
from .c6_carried_return import (
    C6CarriedReturnUnionExclusions as C6CarriedReturnUnionExclusions,
)
from .c6_carried_return import (
    derive_c6_carried_return_union_exclusions as derive_c6_carried_return_union_exclusions,
)
from .c6_carried_return import (
    C6CarriedReturnCountWitness as C6CarriedReturnCountWitness,
)
from .c6_carried_return import (
    C6CarriedReturnCountRelaxation as C6CarriedReturnCountRelaxation,
)
from .c6_carried_return import (
    derive_c6_carried_return_count_relaxation as derive_c6_carried_return_count_relaxation,
)
from .c6_carried_return import C6CarriedReturnWordBudget as C6CarriedReturnWordBudget
from .c6_carried_return import (
    derive_c6_carried_return_word_budget as derive_c6_carried_return_word_budget,
)
from .c6_carried_return import (
    C6CarriedReturnMemoryEnvelope as C6CarriedReturnMemoryEnvelope,
)
from .c6_carried_return import (
    derive_c6_carried_return_memory_envelope as derive_c6_carried_return_memory_envelope,
)
from .c6_carried_return import (
    C6CarriedReturnMemoryRegionExclusion as C6CarriedReturnMemoryRegionExclusion,
)
from .c6_carried_return import (
    C6CarriedReturnMemoryRegionExclusions as C6CarriedReturnMemoryRegionExclusions,
)
from .c6_carried_return import (
    derive_c6_carried_return_memory_region_exclusions as derive_c6_carried_return_memory_region_exclusions,
)
from .c6_carried_return import (
    C6CarriedReturnExcludedPiece as C6CarriedReturnExcludedPiece,
)
from .c6_carried_return import (
    C6CarriedReturnSafeSubtraction as C6CarriedReturnSafeSubtraction,
)
from .c6_carried_return import (
    C6CarriedReturnSafePartition as C6CarriedReturnSafePartition,
)
from .c6_carried_return import (
    C6CarriedReturnSafePartitionRegionExclusions as C6CarriedReturnSafePartitionRegionExclusions,
)
from .c6_carried_return import (
    C6CarriedReturnPredecessorPiece as C6CarriedReturnPredecessorPiece,
)
from .c6_carried_return import (
    C6CarriedReturnPredecessorLayer as C6CarriedReturnPredecessorLayer,
)
from .c6_carried_return import (
    C6CarriedReturnPredecessorPartition as C6CarriedReturnPredecessorPartition,
)
from .c6_carried_return import (
    derive_c6_carried_return_safe_partition as derive_c6_carried_return_safe_partition,
)
from .c6_carried_return import (
    derive_c6_carried_return_safe_partition_region_exclusions as derive_c6_carried_return_safe_partition_region_exclusions,
)
from .c6_carried_return import (
    C6CarriedReturnCoverSchedule as C6CarriedReturnCoverSchedule,
)
from .c6_carried_return import C6CarriedReturnCoverCheck as C6CarriedReturnCoverCheck
from .c6_carried_return import C6CarriedReturnSafeCover as C6CarriedReturnSafeCover
from .c6_carried_return import (
    C6CarriedReturnSafeCoverRegionExclusions as C6CarriedReturnSafeCoverRegionExclusions,
)
from .c6_carried_return import (
    derive_c6_carried_return_safe_cover as derive_c6_carried_return_safe_cover,
)
from .c6_carried_return import (
    derive_c6_carried_return_safe_cover_region_exclusions as derive_c6_carried_return_safe_cover_region_exclusions,
)
from .c6_carried_mean_cylinder import (
    C6CarriedMeanCylinderObstruction as C6CarriedMeanCylinderObstruction,
)
from .c6_carried_mean_cylinder import (
    derive_c6_carried_mean_cylinder_obstruction as derive_c6_carried_mean_cylinder_obstruction,
)
from .c6_carried_mean_cylinder import (
    C6CarriedMeanCylinderBoundary as C6CarriedMeanCylinderBoundary,
)
from .c6_carried_mean_cylinder import (
    C6CarriedMeanCylinderEscape as C6CarriedMeanCylinderEscape,
)
from .c6_carried_mean_cylinder import (
    observe_c6_carried_mean_cylinder_escape as observe_c6_carried_mean_cylinder_escape,
)
from .c6_carried_affine_mean import (
    C6CarriedAffineMeanObstruction as C6CarriedAffineMeanObstruction,
)
from .c6_carried_affine_mean import (
    derive_c6_carried_affine_mean_obstruction as derive_c6_carried_affine_mean_obstruction,
)
from .c6_carried_affine_mean import (
    C6CarriedAffineMeanBoundary as C6CarriedAffineMeanBoundary,
)
from .c6_carried_affine_mean import (
    C6CarriedAffineMeanEscape as C6CarriedAffineMeanEscape,
)
from .c6_carried_affine_mean import (
    observe_c6_carried_affine_mean_escape as observe_c6_carried_affine_mean_escape,
)
from .c6_carried_relay import C6CarriedRelay as C6CarriedRelay
from .c6_carried_relay import C6CarriedRelayAxis as C6CarriedRelayAxis
from .c6_carried_relay import C6CarriedRelayPoint as C6CarriedRelayPoint
from .c6_carried_relay import derive_c6_carried_relay as derive_c6_carried_relay
from .c6_carried_relay import C6CarriedRelayExit as C6CarriedRelayExit
from .c6_carried_relay import (
    observe_c6_carried_relay_exit as observe_c6_carried_relay_exit,
)
from .c6_carried_relay import C6CarriedLocalRelayBudget as C6CarriedLocalRelayBudget
from .c6_carried_relay import (
    derive_c6_carried_local_relay_budget as derive_c6_carried_local_relay_budget,
)
from .c6_carried_relay import C6CarriedLocalRelayPoint as C6CarriedLocalRelayPoint
from .c6_carried_relay import C6CarriedLocalRelayExit as C6CarriedLocalRelayExit
from .c6_carried_relay import (
    observe_c6_carried_local_relay_exit as observe_c6_carried_local_relay_exit,
)
from .binary64_nodal_flow import Binary64QuarterSubstep as Binary64QuarterSubstep
from .binary64_nodal_flow import Binary64UnitQuarterFlow as Binary64UnitQuarterFlow
from .binary64_nodal_flow import (
    observe_binary64_unit_quarter_flow as observe_binary64_unit_quarter_flow,
)
from .binary64_nodal_flow import Binary64PairedC6Diffusion as Binary64PairedC6Diffusion
from .binary64_nodal_flow import (
    observe_binary64_paired_c6_diffusion as observe_binary64_paired_c6_diffusion,
)
from .binary64_nodal_flow import Binary64PressureTraceCell as Binary64PressureTraceCell
from .binary64_nodal_flow import (
    Binary64QuarterPressureBox as Binary64QuarterPressureBox,
)
from .binary64_nodal_flow import (
    derive_binary64_quarter_pressure_box as derive_binary64_quarter_pressure_box,
)
from .coupling_winding import observe_c6_winding_defect as observe_c6_winding_defect
from .coupling_winding import (
    bound_c6_winding_defect_prefix as bound_c6_winding_defect_prefix,
)
from .coupling_winding import (
    bound_c6_winding_uniform_defects as bound_c6_winding_uniform_defects,
)
from .interactions import InteractionResult as InteractionResult
from .interactions import em_like as em_like
from .interactions import weak_like as weak_like
from .interactions import strong_like as strong_like
from .interactions import gravity_like as gravity_like
from .life import LifeTelemetry as LifeTelemetry
from .life import compute_self_generation as compute_self_generation
from .life import compute_autopoietic_coefficient as compute_autopoietic_coefficient
from .life import compute_self_org_index as compute_self_org_index
from .life import compute_stability_margin as compute_stability_margin
from .life import detect_life_emergence as detect_life_emergence
from .cell import CellTelemetry as CellTelemetry
from .cell import MembraneNodeFlux as MembraneNodeFlux
from .cell import MembraneFluxResult as MembraneFluxResult
from .cell import compute_boundary_coherence as compute_boundary_coherence
from .cell import compute_selectivity_index as compute_selectivity_index
from .cell import compute_homeostatic_index as compute_homeostatic_index
from .cell import compute_membrane_integrity as compute_membrane_integrity
from .cell import detect_cell_formation as detect_cell_formation
from .cell import apply_membrane_flux as apply_membrane_flux
from .conservation import ConservationSnapshot as ConservationSnapshot
from .conservation import ConservationBalance as ConservationBalance
from .conservation import ConservationTimeSeries as ConservationTimeSeries
from .conservation import ConservationTracker as ConservationTracker
from .conservation import compute_charge_density as compute_charge_density
from .conservation import compute_current_divergence as compute_current_divergence
from .conservation import capture_conservation_snapshot as capture_conservation_snapshot
from .conservation import verify_conservation_balance as verify_conservation_balance
from .conservation import (
    decompose_conservation_residual as decompose_conservation_residual,
)
from .conservation import analyze_sector_coupling as analyze_sector_coupling
from .conservation import (
    compute_grammar_conservation_bounds as compute_grammar_conservation_bounds,
)
from .conservation import (
    detect_grammar_violations_from_conservation as detect_grammar_violations_from_conservation,
)
from .conservation import compute_noether_charge as compute_noether_charge
from .conservation import compute_energy_functional as compute_energy_functional
from .conservation import WardIdentity as WardIdentity
from .conservation import LyapunovResult as LyapunovResult
from .conservation import SpectralConservation as SpectralConservation
from .conservation import compute_ward_identity as compute_ward_identity
from .conservation import verify_sequence_ward_identity as verify_sequence_ward_identity
from .conservation import compute_lyapunov_derivative as compute_lyapunov_derivative
from .conservation import compute_spectral_conservation as compute_spectral_conservation
from .coherence_geometry import (
    CoherenceLevelSetCertificate as CoherenceLevelSetCertificate,
)
from .coherence_geometry import (
    CrossPolytopeStratification as CrossPolytopeStratification,
)
from .coherence_geometry import (
    FixedCapacityCoherenceLevelSetCertificate as FixedCapacityCoherenceLevelSetCertificate,
)
from .coherence_geometry import (
    NetworkCoherenceLevelSetCertificate as NetworkCoherenceLevelSetCertificate,
)
from .coherence_geometry import (
    coherence_level_set_geometry as coherence_level_set_geometry,
)
from .coherence_geometry import (
    fixed_capacity_coherence_level_set_geometry as fixed_capacity_coherence_level_set_geometry,
)
from .coherence_geometry import (
    network_coherence_level_set_geometry as network_coherence_level_set_geometry,
)
from .unified import compute_complex_geometric_field as compute_complex_geometric_field
from .unified import compute_field_magnitude as compute_field_magnitude
from .unified import compute_field_phase as compute_field_phase
from .unified import compute_chirality_field as compute_chirality_field
from .unified import compute_symmetry_breaking_field as compute_symmetry_breaking_field
from .unified import (
    compute_coherence_coupling_field as compute_coherence_coupling_field,
)
from .unified import compute_energy_density as compute_energy_density
from .unified import compute_action_density as compute_action_density
from .unified import compute_historical_q_density as compute_historical_q_density
from .unified import compute_topological_charge as compute_topological_charge
from .unified import compute_unified_field_suite as compute_unified_field_suite
from .emergent_particles import WindingSector as WindingSector
from .emergent_particles import classify_winding_sector as classify_winding_sector
from .emergent_particles import winding_number as winding_number
from .emergent_particles import winding_ring as winding_ring
from .emergent_particles import EmergentParticle as EmergentParticle
from .emergent_particles import classify_particle as classify_particle
from .dissipative_conservation import DissipativeSnapshot as DissipativeSnapshot
from .dissipative_conservation import DissipativeBalance as DissipativeBalance
from .dissipative_conservation import DissipativeTimeSeries as DissipativeTimeSeries
from .dissipative_conservation import (
    DissipativeConservationTracker as DissipativeConservationTracker,
)
from .dissipative_conservation import (
    capture_dissipative_snapshot as capture_dissipative_snapshot,
)
from .dissipative_conservation import (
    compute_dissipation_bound as compute_dissipation_bound,
)
from .dissipative_conservation import (
    compute_dissipator_action as compute_dissipator_action,
)
from .dissipative_conservation import (
    compute_purity_decay_bound as compute_purity_decay_bound,
)
from .dissipative_conservation import (
    verify_dissipative_balance as verify_dissipative_balance,
)
from .dissipative_conservation import (
    predict_amplitude_damping_purity as predict_amplitude_damping_purity,
)
from .dissipative_conservation import (
    predict_dephasing_purity as predict_dephasing_purity,
)
from .dissipative_conservation import (
    analyze_dissipation_rates as analyze_dissipation_rates,
)
from .dissipative_conservation import (
    classify_dissipative_regime as classify_dissipative_regime,
)
from .dissipative_conservation import (
    steady_state_from_generator as steady_state_from_generator,
)
from .integrity import StructuralIntegrityMonitor as StructuralIntegrityMonitor
from .integrity import StructuralIntegrityViolation as StructuralIntegrityViolation
from .integrity import MonitorMode as MonitorMode
from .integrity import IntegrityReport as IntegrityReport
from .integrity import IntegritySummary as IntegritySummary
from .integrity import enable_integrity_monitor as enable_integrity_monitor
from .spectral_conservation import (
    SpectralConservationBalance as SpectralConservationBalance,
)
from .spectral_conservation import SpectralWardIdentity as SpectralWardIdentity
from .spectral_conservation import SpectralLyapunovResult as SpectralLyapunovResult
from .spectral_conservation import (
    SpectralStructuralEnergyResult as SpectralStructuralEnergyResult,
)
from .spectral_conservation import (
    SpectralSectorDecomposition as SpectralSectorDecomposition,
)
from .spectral_conservation import (
    verify_spectral_conservation_balance as verify_spectral_conservation_balance,
)
from .structural_morphism import (
    EpiCoarseGrainingCertificate as EpiCoarseGrainingCertificate,
)
from .structural_morphism import (
    certify_epi_coarse_graining as certify_epi_coarse_graining,
)
from .phase_quotient import (
    PhaseNodalCoarseGrainingCertificate as PhaseNodalCoarseGrainingCertificate,
)
from .phase_quotient import (
    certify_phase_nodal_coarse_graining as certify_phase_nodal_coarse_graining,
)
from .structural_state_distance import (
    StructuralChannelScales as StructuralChannelScales,
)
from .structural_state_distance import (
    StructuralStateDistanceCertificate as StructuralStateDistanceCertificate,
)
from .structural_state_distance import (
    circular_phase_distance as circular_phase_distance,
)
from .structural_state_distance import (
    fixed_topology_structural_state_distance as fixed_topology_structural_state_distance,
)
from .core_research_integration import (
    CoreResearchIntegrationCertificate as CoreResearchIntegrationCertificate,
)
from .core_research_integration import (
    certify_core_research_integration as certify_core_research_integration,
)
from .core_research_trajectory import (
    CoreResearchTrajectoryIntervalCertificate as CoreResearchTrajectoryIntervalCertificate,
)
from .core_research_trajectory import (
    CoreResearchTrajectoryCertificate as CoreResearchTrajectoryCertificate,
)
from .core_research_trajectory import (
    CoreResearchRefinementSample as CoreResearchRefinementSample,
)
from .core_research_trajectory import (
    CoreResearchRefinementComparison as CoreResearchRefinementComparison,
)
from .core_research_trajectory import (
    certify_core_research_trajectory as certify_core_research_trajectory,
)
from .core_research_trajectory import (
    compare_core_research_trajectory_refinement as compare_core_research_trajectory_refinement,
)
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate as AffineEPIJumpGainCertificate,
)
from .hybrid_operator_stability import (
    HybridEPIStabilityCertificate as HybridEPIStabilityCertificate,
)
from .hybrid_operator_stability import (
    certify_affine_epi_jump_gain as certify_affine_epi_jump_gain,
)
from .hybrid_operator_stability import (
    compose_hybrid_epi_stability as compose_hybrid_epi_stability,
)
from .runtime_flow_stability import NodalFlowStateSnapshot as NodalFlowStateSnapshot
from .runtime_flow_stability import (
    NodalFlowIntervalCertificate as NodalFlowIntervalCertificate,
)
from .runtime_flow_stability import capture_nodal_flow_state as capture_nodal_flow_state
from .runtime_flow_stability import (
    certify_observed_nodal_flow_interval as certify_observed_nodal_flow_interval,
)
from .event_remesh_refinement import (
    EventRemeshEPICheckpointObservation as EventRemeshEPICheckpointObservation,
)
from .event_remesh_refinement import (
    EventRemeshMeshObservation as EventRemeshMeshObservation,
)
from .event_remesh_refinement import (
    EventRemeshPersistentEPIError as EventRemeshPersistentEPIError,
)
from .event_remesh_refinement import (
    EventRemeshThreeMeshModalObservation as EventRemeshThreeMeshModalObservation,
)
from .event_remesh_refinement import (
    EventRemeshThreeMeshRefinementObservation as EventRemeshThreeMeshRefinementObservation,
)
from .event_remesh_refinement import (
    EventRemeshThreeMeshZHIRObservation as EventRemeshThreeMeshZHIRObservation,
)
from .event_remesh_refinement import (
    observe_event_remesh_three_mesh_refinement as observe_event_remesh_three_mesh_refinement,
)
from .reversible_eigenmode_reference import (
    ReversibleSingleEigenmodeEulerReferenceCertificate as ReversibleSingleEigenmodeEulerReferenceCertificate,
)
from .reversible_eigenmode_reference import (
    certify_reversible_single_eigenmode_euler_reference as certify_reversible_single_eigenmode_euler_reference,
)
from .runtime_eigenmode_reference import (
    ExecutedReversibleSingleEigenmodeEulerPartitionObservation as ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
)
from .runtime_eigenmode_reference import (
    ExecutedReversibleSingleEigenmodeEulerReferenceObservation as ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
)
from .runtime_eigenmode_reference import (
    observe_executed_reversible_single_eigenmode_euler_reference as observe_executed_reversible_single_eigenmode_euler_reference,
)
from .event_remesh_reference import (
    P2EventRemeshMeshReferenceObservation as P2EventRemeshMeshReferenceObservation,
)
from .event_remesh_reference import (
    P2EventRemeshReferenceFamilyObservation as P2EventRemeshReferenceFamilyObservation,
)
from .event_remesh_reference import (
    observe_p2_event_remesh_reference_family as observe_p2_event_remesh_reference_family,
)
from .remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate as UniformRemeshHistoryStabilityCertificate,
)
from .remesh_history_stability import (
    UniformRemeshHistoryTransitionObservation as UniformRemeshHistoryTransitionObservation,
)
from .remesh_history_stability import (
    certify_uniform_remesh_history_stability as certify_uniform_remesh_history_stability,
)
from .remesh_history_stability import (
    observe_uniform_remesh_history_transition as observe_uniform_remesh_history_transition,
)
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate as UniformRemeshSchedulePolicyStabilityCertificate,
)
from .remesh_schedule_policy_stability import (
    certify_uniform_remesh_schedule_policy_stability as certify_uniform_remesh_schedule_policy_stability,
)
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate as UniformRemeshScheduleRelativeDefectStabilityCertificate,
)
from .remesh_schedule_relative_defect_stability import (
    certify_uniform_remesh_schedule_relative_defect_stability as certify_uniform_remesh_schedule_relative_defect_stability,
)
from .binary64_remesh_relative_defect import (
    Binary64RemeshPairRelativeDefectObservation as Binary64RemeshPairRelativeDefectObservation,
)
from .binary64_remesh_relative_defect import (
    UniformAlphaOneHardClipRemeshClassCertificate as UniformAlphaOneHardClipRemeshClassCertificate,
)
from .binary64_remesh_relative_defect import (
    UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate as UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate,
)
from .binary64_remesh_relative_defect import (
    certify_alpha_one_hard_clip_remesh_class as certify_alpha_one_hard_clip_remesh_class,
)
from .binary64_remesh_relative_defect import (
    certify_half_alpha_antisymmetric_hard_clip_remesh_class as certify_half_alpha_antisymmetric_hard_clip_remesh_class,
)
from .binary64_remesh_relative_defect import (
    observe_binary64_remesh_pair_relative_defect as observe_binary64_remesh_pair_relative_defect,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate as P2HalfReceptionRemeshStabilityCertificate,
)
from .binary64_p2_reception_stability import (
    certify_p2_half_reception_remesh_stability as certify_p2_half_reception_remesh_stability,
)
from .runtime_p2_reception_stage import (
    ExecutedP2HalfReceptionStageCertificate as ExecutedP2HalfReceptionStageCertificate,
)
from .runtime_p2_reception_stage import (
    certify_executed_p2_half_reception_stage as certify_executed_p2_half_reception_stage,
)
from .runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate as ExecutedP2HalfReceptionRemeshSequenceCertificate,
)
from .runtime_p2_reception_remesh_sequence import (
    certify_executed_p2_half_reception_remesh_sequence as certify_executed_p2_half_reception_remesh_sequence,
)
from .runtime_p2_reception_remesh_policy import (
    execute_p2_half_reception_remesh_policy_invocation as execute_p2_half_reception_remesh_policy_invocation,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation as RuntimeRemeshHistoryBridgeObservation,
)
from .runtime_remesh_history_stability import (
    observe_runtime_remesh_history_bridge as observe_runtime_remesh_history_bridge,
)
from .remesh_schedule_stability import (
    RemeshScheduleHistoryStabilityObservation as RemeshScheduleHistoryStabilityObservation,
)
from .remesh_schedule_stability import (
    observe_remesh_schedule_history_transition as observe_remesh_schedule_history_transition,
)
from .runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation as RuntimeRemeshScheduleBoundaryObservation,
)
from .runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleSequenceObservation as RuntimeRemeshScheduleSequenceObservation,
)
from .runtime_remesh_schedule_stability import (
    observe_runtime_remesh_schedule_sequence as observe_runtime_remesh_schedule_sequence,
)
from .runtime_remesh_schedule_block_margin import (
    RuntimeRemeshScheduleBlockMarginObservation as RuntimeRemeshScheduleBlockMarginObservation,
)
from .runtime_remesh_schedule_block_margin import (
    observe_executed_event_remesh_block_margin as observe_executed_event_remesh_block_margin,
)
from .runtime_remesh_schedule_relative_defect import (
    RuntimeRemeshScheduleRelativeDefectBlockObservation as RuntimeRemeshScheduleRelativeDefectBlockObservation,
)
from .runtime_remesh_schedule_relative_defect import (
    observe_executed_event_remesh_relative_defect_block as observe_executed_event_remesh_relative_defect_block,
)
from .event_duration import (
    ContinuousRelaxationDurationDiagnostic as ContinuousRelaxationDurationDiagnostic,
)
from .event_duration import (
    diagnose_continuous_relaxation_duration as diagnose_continuous_relaxation_duration,
)
from .network_stage_stability import (
    AllTargetNeighborStageCertificate as AllTargetNeighborStageCertificate,
)
from .network_stage_stability import (
    AllTargetNeighborStageStep as AllTargetNeighborStageStep,
)
from .network_stage_stability import (
    NeighborStageDiffusionBridgeCertificate as NeighborStageDiffusionBridgeCertificate,
)
from .network_stage_stability import (
    certify_all_target_neighbor_stage as certify_all_target_neighbor_stage,
)
from .network_stage_stability import (
    certify_reception_all_target_stage as certify_reception_all_target_stage,
)
from .network_stage_stability import (
    certify_resonance_all_target_stage as certify_resonance_all_target_stage,
)
from .network_stage_stability import (
    compose_neighbor_stage_diffusion_stability as compose_neighbor_stage_diffusion_stability,
)
from .reception_realization import (
    ReceptionEPIRealizationCertificate as ReceptionEPIRealizationCertificate,
)
from .reception_realization import (
    certify_reception_epi_realization as certify_reception_epi_realization,
)
from .resonance_realization import (
    ResonanceEPIRealizationCertificate as ResonanceEPIRealizationCertificate,
)
from .resonance_realization import (
    certify_resonance_epi_realization as certify_resonance_epi_realization,
)
from .spectral_conservation import (
    compute_spectral_ward_identity as compute_spectral_ward_identity,
)
from .spectral_conservation import (
    compute_spectral_lyapunov as compute_spectral_lyapunov,
)
from .spectral_conservation import (
    compute_spectral_structural_energy as compute_spectral_structural_energy,
)
from .spectral_conservation import (
    decompose_spectral_sectors as decompose_spectral_sectors,
)
from .spectral_conservation import (
    compute_spectral_energy_conservation as compute_spectral_energy_conservation,
)
from .spectral_conservation import classify_spectral_modes as classify_spectral_modes
from .variational import ConjugatePair as ConjugatePair
from .variational import LagrangianSnapshot as LagrangianSnapshot
from .variational import EulerLagrangeResidual as EulerLagrangeResidual
from .variational import SymplecticCheck as SymplecticCheck
from .variational import GrammarStationarityAnalysis as GrammarStationarityAnalysis
from .variational import CriticalPointAnalysis as CriticalPointAnalysis
from .variational import VariationalTimeSeries as VariationalTimeSeries
from .variational import VariationalTracker as VariationalTracker
from .variational import compute_kinetic_density as compute_kinetic_density
from .variational import compute_potential_density as compute_potential_density
from .variational import compute_lagrangian_density as compute_lagrangian_density
from .variational import compute_hamiltonian_density as compute_hamiltonian_density
from .variational import compute_interaction_density as compute_interaction_density
from .variational import translate_sectors as translate_sectors
from .variational import identify_conjugate_pairs as identify_conjugate_pairs
from .variational import compute_phase_space_volume as compute_phase_space_volume
from .variational import (
    compute_poisson_bracket_estimate as compute_poisson_bracket_estimate,
)
from .variational import capture_lagrangian_snapshot as capture_lagrangian_snapshot
from .variational import (
    compute_euler_lagrange_residual as compute_euler_lagrange_residual,
)
from .variational import compute_action_functional as compute_action_functional
from .variational import check_symplectic_preservation as check_symplectic_preservation
from .variational import classify_operator_canonical as classify_operator_canonical
from .variational import analyze_grammar_stationarity as analyze_grammar_stationarity
from .variational import (
    analyze_potential_critical_points as analyze_potential_critical_points,
)
from .variational import compute_variational_suite as compute_variational_suite
from .symplectic_substrate import PhaseSpacePoint as PhaseSpacePoint
from .symplectic_substrate import (
    CanonicalStructureCertificate as CanonicalStructureCertificate,
)
from .symplectic_substrate import NoetherChargeCertificate as NoetherChargeCertificate
from .symplectic_substrate import (
    HermitianStructureCertificate as HermitianStructureCertificate,
)
from .symplectic_substrate import IntegrabilityCertificate as IntegrabilityCertificate
from .symplectic_substrate import PoincareCartanCertificate as PoincareCartanCertificate
from .symplectic_substrate import (
    MarsdenWeinsteinCertificate as MarsdenWeinsteinCertificate,
)
from .symplectic_substrate import (
    PolarizationSymmetryCertificate as PolarizationSymmetryCertificate,
)
from .symplectic_substrate import SubstrateGeometryReport as SubstrateGeometryReport
from .symplectic_substrate import extract_phase_space_point as extract_phase_space_point
from .symplectic_substrate import symplectic_form_matrix as symplectic_form_matrix
from .symplectic_substrate import (
    symplectic_pullback_residual as symplectic_pullback_residual,
)
from .symplectic_substrate import complex_structure_matrix as complex_structure_matrix
from .symplectic_substrate import compatible_metric_matrix as compatible_metric_matrix
from .symplectic_substrate import substrate_hamiltonian as substrate_hamiltonian
from .symplectic_substrate import background_potential as background_potential
from .symplectic_substrate import hamiltonian_vector_field as hamiltonian_vector_field
from .symplectic_substrate import poisson_bracket as poisson_bracket
from .symplectic_substrate import canonical_bracket_table as canonical_bracket_table
from .symplectic_substrate import liouville_divergence as liouville_divergence
from .symplectic_substrate import (
    verify_canonical_structure as verify_canonical_structure,
)
from .symplectic_substrate import evolve_substrate_flow as evolve_substrate_flow
from .symplectic_substrate import geometric_sector_energy as geometric_sector_energy
from .symplectic_substrate import potential_sector_energy as potential_sector_energy
from .symplectic_substrate import noether_charges as noether_charges
from .symplectic_substrate import (
    verify_noether_conservation as verify_noether_conservation,
)
from .symplectic_substrate import to_complex_coordinates as to_complex_coordinates
from .symplectic_substrate import kahler_potential as kahler_potential
from .symplectic_substrate import (
    verify_hermitian_structure as verify_hermitian_structure,
)
from .symplectic_substrate import to_action_angle as to_action_angle
from .symplectic_substrate import verify_integrability as verify_integrability
from .symplectic_substrate import substrate_flow_matrix as substrate_flow_matrix
from .symplectic_substrate import loop_action_integral as loop_action_integral
from .symplectic_substrate import verify_poincare_cartan as verify_poincare_cartan
from .symplectic_substrate import diagonal_moment_map as diagonal_moment_map
from .symplectic_substrate import (
    reduced_symplectic_form_matrix as reduced_symplectic_form_matrix,
)
from .symplectic_substrate import (
    verify_symplectic_reduction as verify_symplectic_reduction,
)
from .symplectic_substrate import polarization_vector as polarization_vector
from .symplectic_substrate import polarization_density as polarization_density
from .symplectic_substrate import (
    verify_polarization_symmetry as verify_polarization_symmetry,
)
from .symplectic_substrate import verify_substrate_geometry as verify_substrate_geometry
from .structural_diffusion import (
    StructuralDiffusionCertificate as StructuralDiffusionCertificate,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate as HeterogeneousDiffusionStabilityCertificate,
)
from .structural_diffusion import (
    SwitchingDiffusionStabilityCertificate as SwitchingDiffusionStabilityCertificate,
)
from .structural_diffusion import (
    TimeVaryingDiffusionStabilityBound as TimeVaryingDiffusionStabilityBound,
)
from .structural_diffusion import (
    OverdampedRegimeCertificate as OverdampedRegimeCertificate,
)
from .structural_diffusion import DiscreteModeCertificate as DiscreteModeCertificate
from .structural_diffusion import (
    EulerRelaxationWindowDiagnostic as EulerRelaxationWindowDiagnostic,
)
from .structural_diffusion import (
    StructuralStabilityCertificate as StructuralStabilityCertificate,
)
from .structural_diffusion import RandomWalkCertificate as RandomWalkCertificate
from .structural_diffusion import StructuralFlowCertificate as StructuralFlowCertificate
from .structural_diffusion import (
    structural_diffusion_operator as structural_diffusion_operator,
)
from .structural_diffusion import structural_field as structural_field
from .structural_diffusion import structural_diffusivity as structural_diffusivity
from .structural_diffusion import relaxation_spectrum as relaxation_spectrum
from .structural_diffusion import degree_weighted_total as degree_weighted_total
from .structural_diffusion import (
    derive_time_varying_diffusion_stability_bound as derive_time_varying_diffusion_stability_bound,
)
from .structural_diffusion import (
    diagnose_euler_relaxation_window as diagnose_euler_relaxation_window,
)
from .structural_diffusion import (
    verify_heterogeneous_diffusion_stability as verify_heterogeneous_diffusion_stability,
)
from .structural_diffusion import (
    verify_switching_diffusion_stability as verify_switching_diffusion_stability,
)
from .structural_diffusion import structural_eigenvalues as structural_eigenvalues
from .structural_diffusion import structural_eigenmodes as structural_eigenmodes
from .structural_diffusion import nodal_domain_count as nodal_domain_count
from .structural_diffusion import dispersion_relation as dispersion_relation
from .structural_diffusion import instability_threshold as instability_threshold
from .structural_diffusion import fiedler_partition as fiedler_partition
from .structural_diffusion import random_walk_matrix as random_walk_matrix
from .structural_diffusion import stationary_distribution as stationary_distribution
from .structural_diffusion import effective_resistance as effective_resistance
from .structural_diffusion import commute_time as commute_time
from .structural_diffusion import structural_current as structural_current
from .structural_diffusion import current_divergence as current_divergence
from .structural_diffusion import (
    verify_structural_diffusion as verify_structural_diffusion,
)
from .structural_diffusion import verify_overdamped_regime as verify_overdamped_regime
from .structural_diffusion import verify_discrete_modes as verify_discrete_modes
from .structural_diffusion import (
    verify_structural_stability as verify_structural_stability,
)
from .structural_diffusion import (
    verify_structural_random_walk as verify_structural_random_walk,
)
from .structural_diffusion import verify_structural_flow as verify_structural_flow
from .gauge import GaugeSnapshot as GaugeSnapshot
from .gauge import GaugeInvarianceResult as GaugeInvarianceResult
from .gauge import apply_gauge_transformation as apply_gauge_transformation
from .gauge import compute_gauge_connection as compute_gauge_connection
from .gauge import compute_gauge_curvature as compute_gauge_curvature
from .gauge import compute_covariant_derivative as compute_covariant_derivative
from .gauge import (
    compute_covariant_derivative_magnitude as compute_covariant_derivative_magnitude,
)
from .gauge import compute_topological_norm as compute_topological_norm
from .gauge import compute_chirality_norm as compute_chirality_norm
from .gauge import compute_dual_topological_charge as compute_dual_topological_charge
from .gauge import compute_dual_chirality as compute_dual_chirality
from .gauge import verify_gauge_invariance as verify_gauge_invariance
from .gauge import capture_gauge_snapshot as capture_gauge_snapshot
from .gauge import classify_interaction_regime as classify_interaction_regime
from .gauge import classify_network_regimes as classify_network_regimes
from .gauge import compute_yang_mills_action as compute_yang_mills_action
from .gauge import (
    compute_gauge_energy_decomposition as compute_gauge_energy_decomposition,
)
from .gauge import YangMillsFieldEquations as YangMillsFieldEquations
from .gauge import BianchiIdentityResult as BianchiIdentityResult
from .gauge import InteractionRegimeMetrics as InteractionRegimeMetrics
from .gauge import NetworkInteractionProfile as NetworkInteractionProfile
from .gauge import N_REGIMES as N_REGIMES
from .gauge import REGIME_ACTIVITY_SHARE as REGIME_ACTIVITY_SHARE
from .gauge import compute_matter_current as compute_matter_current
from .gauge import compute_yang_mills_equations as compute_yang_mills_equations
from .gauge import verify_bianchi_identity as verify_bianchi_identity
from .gauge import compute_gauss_law_residual as compute_gauss_law_residual
from .gauge import compute_gauge_coupling_constant as compute_gauge_coupling_constant
from .gauge import (
    classify_interaction_regime_formal as classify_interaction_regime_formal,
)
from .gauge import (
    compute_network_interaction_profile as compute_network_interaction_profile,
)
from .phase_transition import Phase as Phase
from .phase_transition import PhaseTransitionTelemetry as PhaseTransitionTelemetry
from .phase_transition import PhaseSnapshot as PhaseSnapshot
from .phase_transition import Z_SIGNIFICANCE as Z_SIGNIFICANCE
from .phase_transition import symmetry_zscore as symmetry_zscore
from .phase_transition import compute_order_parameter as compute_order_parameter
from .phase_transition import (
    compute_chirality_statistics as compute_chirality_statistics,
)
from .phase_transition import classify_phase as classify_phase
from .phase_transition import capture_phase_snapshot as capture_phase_snapshot
from .phase_transition import detect_phase_transition as detect_phase_transition
from .phase_transition import fit_critical_exponent as fit_critical_exponent
from .phase_scaling import SizePowerLawFit as SizePowerLawFit
from .phase_scaling import PhaseScalingDiagnostic as PhaseScalingDiagnostic
from .phase_scaling import (
    analyze_phase_finite_size_scaling as analyze_phase_finite_size_scaling,
)
from .topology_transitions import NodalTopologySnapshot as NodalTopologySnapshot
from .topology_transitions import NodalTopologyStep as NodalTopologyStep
from .topology_transitions import (
    NodalTopologyTransitionCertificate as NodalTopologyTransitionCertificate,
)
from .topology_transitions import (
    capture_nodal_topology_snapshot as capture_nodal_topology_snapshot,
)
from .topology_transitions import (
    detect_nodal_topology_transitions as detect_nodal_topology_transitions,
)
from .lyapunov import U2PolicyRole as U2PolicyRole
from .lyapunov import OperatorPolicyMultiplier as OperatorPolicyMultiplier
from .lyapunov import OperatorPolicyComparison as OperatorPolicyComparison
from .lyapunov import OperatorPolicySpectralContext as OperatorPolicySpectralContext
from .lyapunov import SequencePolicyEvaluation as SequencePolicyEvaluation
from .lyapunov import OPERATOR_POLICY_MULTIPLIERS as OPERATOR_POLICY_MULTIPLIERS
from .lyapunov import get_policy_multiplier as get_policy_multiplier
from .lyapunov import compute_operator_policy_delta as compute_operator_policy_delta
from .lyapunov import (
    compare_operator_energy_to_policy as compare_operator_energy_to_policy,
)
from .lyapunov import compute_sequence_policy_score as compute_sequence_policy_score
from .lyapunov import evaluate_sequence_policy as evaluate_sequence_policy
from .lyapunov import analyze_operator_policy_context as analyze_operator_policy_context
from .lyapunov import EnergyClass as EnergyClass
from .lyapunov import OperatorLyapunovBound as OperatorLyapunovBound
from .lyapunov import OperatorLyapunovVerification as OperatorLyapunovVerification
from .lyapunov import SpectralGapAnalysis as SpectralGapAnalysis
from .lyapunov import LyapunovSpectralSummary as LyapunovSpectralSummary
from .lyapunov import SequenceLyapunovProof as SequenceLyapunovProof
from .lyapunov import OPERATOR_LYAPUNOV_BOUNDS as OPERATOR_LYAPUNOV_BOUNDS
from .lyapunov import get_bound as get_bound
from .lyapunov import compute_operator_energy_bound as compute_operator_energy_bound
from .lyapunov import verify_operator_lyapunov as verify_operator_lyapunov
from .lyapunov import compute_sequence_energy_bound as compute_sequence_energy_bound
from .lyapunov import prove_sequence_lyapunov as prove_sequence_lyapunov
from .metriplectic import (
    MetriplecticProductCertificate as MetriplecticProductCertificate,
)
from .metriplectic import verify_metriplectic_product as verify_metriplectic_product
from .multiscale_coherence import U5CoherenceAssessment as U5CoherenceAssessment
from .multiscale_coherence import (
    assess_u5_parent_child_coherence as assess_u5_parent_child_coherence,
)
from .mutation_trigger import MutationTriggerCertificate as MutationTriggerCertificate
from .mutation_trigger import MutationTriggerEvidence as MutationTriggerEvidence
from .mutation_trigger import MutationTriggerInputError as MutationTriggerInputError
from .mutation_trigger import certify_mutation_trigger as certify_mutation_trigger
from .nonnormal_prediction import NonnormalPredictorRecord as NonnormalPredictorRecord
from .nonnormal_prediction import (
    NonnormalPredictionCertificate as NonnormalPredictionCertificate,
)
from .nonnormal_prediction import (
    deterministic_directed_family as deterministic_directed_family,
)
from .nonnormal_prediction import (
    measure_nonnormal_pressure_prediction as measure_nonnormal_pressure_prediction,
)
from .nonnormal_prediction import (
    benchmark_nonnormal_prediction as benchmark_nonnormal_prediction,
)
from .operator_quotient import (
    OperatorQuotientCertificate as OperatorQuotientCertificate,
)
from .operator_quotient import certify_operator_quotient as certify_operator_quotient
from .temporal_identifiability import (
    NearestSignatureIdentification as NearestSignatureIdentification,
)
from .temporal_identifiability import (
    SignatureMatrixCertificate as SignatureMatrixCertificate,
)
from .temporal_identifiability import (
    SignatureNoiseMarginCertificate as SignatureNoiseMarginCertificate,
)
from .temporal_identifiability import (
    TemporalOperatorIdentifiabilityCertificate as TemporalOperatorIdentifiabilityCertificate,
)
from .temporal_identifiability import (
    certify_signature_noise_margin as certify_signature_noise_margin,
)
from .temporal_identifiability import (
    certify_temporal_signature_matrix as certify_temporal_signature_matrix,
)
from .temporal_identifiability import (
    identify_nearest_signature as identify_nearest_signature,
)
from .temporal_identifiability import (
    probe_canonical_operator_identifiability as probe_canonical_operator_identifiability,
)
from .lyapunov import analyze_spectral_gap as analyze_spectral_gap
from .lyapunov import analyze_operator_convergence as analyze_operator_convergence
from .conservation_gauge_unification import (
    GrammarSymmetryMapping as GrammarSymmetryMapping,
)
from .conservation_gauge_unification import (
    ActionEnergyConsistency as ActionEnergyConsistency,
)
from .conservation_gauge_unification import (
    NoetherGaugeDecomposition as NoetherGaugeDecomposition,
)
from .conservation_gauge_unification import (
    GaugeConservationCoupling as GaugeConservationCoupling,
)
from .conservation_gauge_unification import (
    SymplecticGaugeCompatibility as SymplecticGaugeCompatibility,
)
from .conservation_gauge_unification import (
    ConservationGaugeUnification as ConservationGaugeUnification,
)
from .conservation_gauge_unification import (
    compute_grammar_symmetry_mapping as compute_grammar_symmetry_mapping,
)
from .conservation_gauge_unification import (
    verify_action_energy_consistency as verify_action_energy_consistency,
)
from .conservation_gauge_unification import (
    compute_noether_gauge_decomposition as compute_noether_gauge_decomposition,
)
from .conservation_gauge_unification import (
    compute_gauge_conservation_coupling as compute_gauge_conservation_coupling,
)
from .conservation_gauge_unification import (
    verify_symplectic_gauge_compatibility as verify_symplectic_gauge_compatibility,
)
from .conservation_gauge_unification import (
    run_conservation_gauge_unification as run_conservation_gauge_unification,
)

__all__ = [
    "RegionalAffineClosure",
    "RegionalFormObservation",
    "RegionalFormRegion",
    "derive_regional_affine_closure",
    "observe_regional_form",
    "SourceRelativeFormObservation",
    "observe_source_relative_form",
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
    "ObserverTransportCertificate",
    "KronReductionCertificate",
    "ComposedReductionCertificate",
    "observer_transport_certificate",
    "kron_reduction_certificate",
    "composed_reduction_certificate",
    "EpiMemoryObservation",
    "EpiMemorySample",
    "observe_epi_memory",
    "P5MemoryTruncationReference",
    "P5MemoryTruncationSample",
    "bound_p5_memory_truncation",
    "P5HiddenFormBound",
    "P5HiddenFormSample",
    "bound_p5_hidden_form",
    "P2PhaseFormModel",
    "P2PhaseFormStep",
    "derive_p2_phase_form_model",
    "propose_p2_phase_form_step",
    "P5ReducedState",
    "P5ReductionGeometry",
    "P5RemeshReduction",
    "reduce_p5_state",
    "p5_reduction_geometry",
    "observe_p5_remesh_reduction",
    "WindingCertificate",
    "WindingStepObservation",
    "WindingWordObservation",
    "certify_phase_winding",
    "observe_winding_word",
    "CouplingGapStep",
    "observe_coupling_gap_step",
    "CycleCapacityBalance",
    "observe_cycle_capacity_balance",
    "CycleMemoryRelaxationReference",
    "certify_cycle_memory_relaxation",
    "CycleSupportBalance",
    "CycleSupportEuler",
    "CycleSupportReset",
    "observe_cycle_support_balance",
    "observe_cycle_support_euler",
    "observe_cycle_support_reset",
    "SupportTransportSnapshot",
    "SupportTransportReset",
    "SupportTransportEuler",
    "SupportTransportClippedFlow",
    "observe_support_transport",
    "observe_support_transport_reset",
    "observe_support_transport_euler",
    "observe_support_transport_clipped_flow",
    "ForcedSupportBalance",
    "ForcedSupportEvent",
    "ForcedSupportJumpEnergy",
    "observe_forced_support_event",
    "ForcedSupportPattern",
    "ForcedSupportReset",
    "ForcedSupportResetEnergy",
    "observe_forced_support_pattern",
    "observe_forced_support_reset",
    "ForcedSupportState",
    "ForcedSupportStep",
    "ForcedSupportTarget",
    "derive_forced_support_balance",
    "observe_forced_support_state",
    "observe_forced_support_step",
    "observe_forced_support_target",
    "NonEpiForcingObservation",
    "ForcingDirichletBalance",
    "ForcingMeanBalance",
    "ForcingCapacityDifference",
    "observe_forcing_capacity_difference",
    "capture_non_epi_forcing",
    "decompose_non_epi_forcing",
    "observe_forcing_dirichlet_balance",
    "observe_forcing_mean_balance",
    "P2CapacityFeedbackBound",
    "CapacityIntervalRecoveryBound",
    "derive_capacity_interval_recovery_bound",
    "P2CapacityFeedbackCycle",
    "P2CapacityFeedbackReference",
    "bound_p2_capacity_feedback",
    "derive_p2_capacity_feedback",
    "observe_p2_capacity_feedback_cycle",
    "P2Binary64CouplingObservation",
    "P2Binary64CouplingReference",
    "derive_p2_binary64_coupling_lattice",
    "observe_p2_binary64_coupling_lattice",
    "CompatibleCapacityBalance",
    "CouplingSupportObservation",
    "derive_compatible_capacity_balance",
    "observe_coupling_support",
    "AntipodalRegionPhaseBalance",
    "AntipodalRegionPhaseResponse",
    "derive_antipodal_region_phase_balance",
    "observe_antipodal_region_phase_response",
    "PhaseResponseReference",
    "derive_phase_response",
    "C6WindingPhaseReference",
    "C6WindingPhaseObservation",
    "derive_c6_winding_phase_response",
    "observe_c6_winding_phase_response",
    "C6WindingJointDomain",
    "C6WindingJointStep",
    "derive_c6_winding_joint_domain",
    "observe_c6_winding_joint_domain",
    "C6WindingDefect",
    "C6WindingDefectPrefix",
    "C6WindingUniformDefectBound",
    "C6WindingPairingReference",
    "C6WindingPairingObservation",
    "c6_centered_opposite_pairs",
    "derive_c6_winding_pairing",
    "observe_c6_winding_pairing",
    "Binary64AdditionCell",
    "NodalRemainderPrefix",
    "NodalRemainderSequence",
    "observe_nodal_remainder_sequence",
    "NodalRemainderCellHorizon",
    "derive_nodal_remainder_cell_horizon",
    "NodalRemainderCellExit",
    "observe_nodal_remainder_cell_exit",
    "NodalRemainderItineraryCell",
    "NodalRemainderItinerary",
    "derive_nodal_remainder_itinerary",
    "Binary64PressureEquilibriumRow",
    "Binary64C6PressureEquilibriumObstruction",
    "derive_binary64_c6_pressure_equilibrium_obstruction",
    "NodalRemainderPressureReadout",
    "observe_nodal_remainder_pressure_readout",
    "PeriodicPhaseSourceBudget",
    "PeriodicPhaseSourceCompensation",
    "derive_periodic_phase_source_budget",
    "observe_periodic_phase_source_compensation",
    "FiniteNodalPressureDrift",
    "observe_finite_nodal_pressure_drift",
    "NodalAreaCrossing",
    "NodalAreaCrossings",
    "derive_nodal_area_crossings",
    "TwoLevelNodalReturn",
    "derive_two_level_nodal_return",
    "FiniteLevelNodalReturn",
    "derive_finite_level_nodal_return",
    "NodalRemainderCycleGradient",
    "observe_nodal_remainder_cycle_gradient",
    "C6PressureLatticeRow",
    "C6PressureLatticeReference",
    "C6PressureLatticeObservation",
    "derive_c6_pressure_lattice",
    "observe_c6_pressure_lattice",
    "C6PressureSignSector",
    "C6PressureSectorExit",
    "derive_c6_pressure_sign_sector",
    "observe_c6_pressure_sector_exit",
    "C6FrozenPressureStencil",
    "observe_c6_frozen_pressure_stencil",
    "C6CouplingCoherencePhaseStep",
    "C6CouplingCoherencePhaseOrbit",
    "observe_c6_coupling_coherence_phase_step",
    "derive_c6_coupling_coherence_phase_orbit",
    "C6CarriedProfile",
    "derive_c6_carried_profile",
    "C6CarriedProfileStep",
    "observe_c6_carried_profile_step",
    "C6CarriedContraction",
    "derive_c6_carried_contraction",
    "C6CarriedTube",
    "derive_c6_carried_tube",
    "C6CarriedBandHorizon",
    "derive_c6_carried_band_horizon",
    "C6CarriedCutExclusion",
    "observe_c6_carried_cut_exclusion",
    "C6CarriedClosure",
    "derive_c6_carried_closure",
    "C6CarriedPressurePoint",
    "observe_c6_carried_pressure_point",
    "C6CarriedPressureBalance",
    "derive_c6_carried_pressure_balance",
    "C6CarriedPositivePressurePassage",
    "derive_c6_carried_positive_pressure_passage",
    "C6CarriedCompleteCellObstruction",
    "derive_c6_carried_complete_cell_obstruction",
    "C6CarriedCompleteCellEscape",
    "observe_c6_carried_complete_cell_escape",
    "C6CarriedCellGraph",
    "derive_c6_carried_cell_graph",
    "C6CarriedViabilityBox",
    "C6CarriedViabilityIteration",
    "C6CarriedViability",
    "derive_c6_carried_viability",
    "C6CarriedForwardZone",
    "C6CarriedPairBarrier",
    "C6CarriedForwardIteration",
    "C6CarriedForwardEnvelope",
    "derive_c6_carried_forward_envelope",
    "C6CarriedPredecessorLayer",
    "C6CarriedPredecessors",
    "derive_c6_carried_predecessors",
    "C6CarriedRegionIteration",
    "C6CarriedRegionExclusion",
    "C6CarriedRegionExclusions",
    "derive_c6_carried_region_exclusions",
    "C6CarriedReachableIteration",
    "C6CarriedReachableEnvelope",
    "derive_c6_carried_reachable_envelope",
    "C6CarriedLinearExtremum",
    "C6CarriedExcursionIngress",
    "C6CarriedExcursionExclusion",
    "derive_c6_carried_excursion_exclusion",
    "C6CarriedExcursionTransition",
    "C6CarriedModeExcursionExclusion",
    "derive_c6_carried_mode_excursion_exclusion",
    "C6CarriedReturnTransition",
    "C6CarriedReturnIteration",
    "C6CarriedReturnEnvelope",
    "derive_c6_carried_return_envelope",
    "C6CarriedReturnRegionExclusion",
    "C6CarriedReturnRegionExclusions",
    "derive_c6_carried_return_region_exclusions",
    "C6CarriedReturnUnionExclusion",
    "C6CarriedReturnUnionExclusions",
    "derive_c6_carried_return_union_exclusions",
    "C6CarriedReturnCountWitness",
    "C6CarriedReturnCountRelaxation",
    "derive_c6_carried_return_count_relaxation",
    "C6CarriedReturnWordBudget",
    "derive_c6_carried_return_word_budget",
    "C6CarriedReturnMemoryEnvelope",
    "derive_c6_carried_return_memory_envelope",
    "C6CarriedReturnMemoryRegionExclusion",
    "C6CarriedReturnMemoryRegionExclusions",
    "derive_c6_carried_return_memory_region_exclusions",
    "C6CarriedReturnExcludedPiece",
    "C6CarriedReturnSafeSubtraction",
    "C6CarriedReturnSafePartition",
    "C6CarriedReturnSafePartitionRegionExclusions",
    "C6CarriedReturnPredecessorPiece",
    "C6CarriedReturnPredecessorLayer",
    "C6CarriedReturnPredecessorPartition",
    "derive_c6_carried_return_safe_partition",
    "derive_c6_carried_return_safe_partition_region_exclusions",
    "C6CarriedReturnCoverSchedule",
    "C6CarriedReturnCoverCheck",
    "C6CarriedReturnSafeCover",
    "C6CarriedReturnSafeCoverRegionExclusions",
    "derive_c6_carried_return_safe_cover",
    "derive_c6_carried_return_safe_cover_region_exclusions",
    "C6CarriedMeanCylinderObstruction",
    "derive_c6_carried_mean_cylinder_obstruction",
    "C6CarriedMeanCylinderBoundary",
    "C6CarriedMeanCylinderEscape",
    "observe_c6_carried_mean_cylinder_escape",
    "C6CarriedAffineMeanObstruction",
    "derive_c6_carried_affine_mean_obstruction",
    "C6CarriedAffineMeanBoundary",
    "C6CarriedAffineMeanEscape",
    "observe_c6_carried_affine_mean_escape",
    "C6CarriedRelay",
    "C6CarriedRelayAxis",
    "C6CarriedRelayPoint",
    "derive_c6_carried_relay",
    "C6CarriedRelayExit",
    "observe_c6_carried_relay_exit",
    "C6CarriedLocalRelayBudget",
    "derive_c6_carried_local_relay_budget",
    "C6CarriedLocalRelayPoint",
    "C6CarriedLocalRelayExit",
    "observe_c6_carried_local_relay_exit",
    "Binary64QuarterSubstep",
    "Binary64UnitQuarterFlow",
    "observe_binary64_unit_quarter_flow",
    "Binary64PairedC6Diffusion",
    "observe_binary64_paired_c6_diffusion",
    "Binary64PressureTraceCell",
    "Binary64QuarterPressureBox",
    "derive_binary64_quarter_pressure_box",
    "observe_c6_winding_defect",
    "bound_c6_winding_defect_prefix",
    "bound_c6_winding_uniform_defects",
    "InteractionResult",
    "em_like",
    "weak_like",
    "strong_like",
    "gravity_like",
    "LifeTelemetry",
    "compute_self_generation",
    "compute_autopoietic_coefficient",
    "compute_self_org_index",
    "compute_stability_margin",
    "detect_life_emergence",
    "CellTelemetry",
    "MembraneNodeFlux",
    "MembraneFluxResult",
    "compute_boundary_coherence",
    "compute_selectivity_index",
    "compute_homeostatic_index",
    "compute_membrane_integrity",
    "detect_cell_formation",
    "apply_membrane_flux",
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
    "CoherenceLevelSetCertificate",
    "CrossPolytopeStratification",
    "FixedCapacityCoherenceLevelSetCertificate",
    "NetworkCoherenceLevelSetCertificate",
    "coherence_level_set_geometry",
    "fixed_capacity_coherence_level_set_geometry",
    "network_coherence_level_set_geometry",
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
    "WindingSector",
    "classify_winding_sector",
    "winding_number",
    "winding_ring",
    "EmergentParticle",
    "classify_particle",
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
    "StructuralIntegrityMonitor",
    "StructuralIntegrityViolation",
    "MonitorMode",
    "IntegrityReport",
    "IntegritySummary",
    "enable_integrity_monitor",
    "SpectralConservationBalance",
    "SpectralWardIdentity",
    "SpectralLyapunovResult",
    "SpectralStructuralEnergyResult",
    "SpectralSectorDecomposition",
    "verify_spectral_conservation_balance",
    "EpiCoarseGrainingCertificate",
    "certify_epi_coarse_graining",
    "PhaseNodalCoarseGrainingCertificate",
    "certify_phase_nodal_coarse_graining",
    "StructuralChannelScales",
    "StructuralStateDistanceCertificate",
    "circular_phase_distance",
    "fixed_topology_structural_state_distance",
    "CoreResearchIntegrationCertificate",
    "certify_core_research_integration",
    "CoreResearchTrajectoryIntervalCertificate",
    "CoreResearchTrajectoryCertificate",
    "CoreResearchRefinementSample",
    "CoreResearchRefinementComparison",
    "certify_core_research_trajectory",
    "compare_core_research_trajectory_refinement",
    "AffineEPIJumpGainCertificate",
    "HybridEPIStabilityCertificate",
    "certify_affine_epi_jump_gain",
    "compose_hybrid_epi_stability",
    "NodalFlowStateSnapshot",
    "NodalFlowIntervalCertificate",
    "capture_nodal_flow_state",
    "certify_observed_nodal_flow_interval",
    "EventRemeshEPICheckpointObservation",
    "EventRemeshMeshObservation",
    "EventRemeshPersistentEPIError",
    "EventRemeshThreeMeshModalObservation",
    "EventRemeshThreeMeshRefinementObservation",
    "EventRemeshThreeMeshZHIRObservation",
    "observe_event_remesh_three_mesh_refinement",
    "ReversibleSingleEigenmodeEulerReferenceCertificate",
    "certify_reversible_single_eigenmode_euler_reference",
    "ExecutedReversibleSingleEigenmodeEulerPartitionObservation",
    "ExecutedReversibleSingleEigenmodeEulerReferenceObservation",
    "observe_executed_reversible_single_eigenmode_euler_reference",
    "P2EventRemeshMeshReferenceObservation",
    "P2EventRemeshReferenceFamilyObservation",
    "observe_p2_event_remesh_reference_family",
    "UniformRemeshHistoryStabilityCertificate",
    "UniformRemeshHistoryTransitionObservation",
    "certify_uniform_remesh_history_stability",
    "observe_uniform_remesh_history_transition",
    "UniformRemeshSchedulePolicyStabilityCertificate",
    "certify_uniform_remesh_schedule_policy_stability",
    "UniformRemeshScheduleRelativeDefectStabilityCertificate",
    "certify_uniform_remesh_schedule_relative_defect_stability",
    "Binary64RemeshPairRelativeDefectObservation",
    "UniformAlphaOneHardClipRemeshClassCertificate",
    "UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate",
    "certify_alpha_one_hard_clip_remesh_class",
    "certify_half_alpha_antisymmetric_hard_clip_remesh_class",
    "observe_binary64_remesh_pair_relative_defect",
    "P2HalfReceptionRemeshStabilityCertificate",
    "certify_p2_half_reception_remesh_stability",
    "ExecutedP2HalfReceptionStageCertificate",
    "certify_executed_p2_half_reception_stage",
    "ExecutedP2HalfReceptionRemeshSequenceCertificate",
    "certify_executed_p2_half_reception_remesh_sequence",
    "execute_p2_half_reception_remesh_policy_invocation",
    "RuntimeRemeshHistoryBridgeObservation",
    "observe_runtime_remesh_history_bridge",
    "RemeshScheduleHistoryStabilityObservation",
    "observe_remesh_schedule_history_transition",
    "RuntimeRemeshScheduleBoundaryObservation",
    "RuntimeRemeshScheduleSequenceObservation",
    "observe_runtime_remesh_schedule_sequence",
    "RuntimeRemeshScheduleBlockMarginObservation",
    "observe_executed_event_remesh_block_margin",
    "RuntimeRemeshScheduleRelativeDefectBlockObservation",
    "observe_executed_event_remesh_relative_defect_block",
    "ContinuousRelaxationDurationDiagnostic",
    "diagnose_continuous_relaxation_duration",
    "AllTargetNeighborStageCertificate",
    "AllTargetNeighborStageStep",
    "NeighborStageDiffusionBridgeCertificate",
    "certify_all_target_neighbor_stage",
    "certify_reception_all_target_stage",
    "certify_resonance_all_target_stage",
    "compose_neighbor_stage_diffusion_stability",
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
    "SizePowerLawFit",
    "PhaseScalingDiagnostic",
    "analyze_phase_finite_size_scaling",
    "NodalTopologySnapshot",
    "NodalTopologyStep",
    "NodalTopologyTransitionCertificate",
    "capture_nodal_topology_snapshot",
    "detect_nodal_topology_transitions",
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
    "MetriplecticProductCertificate",
    "verify_metriplectic_product",
    "U5CoherenceAssessment",
    "assess_u5_parent_child_coherence",
    "MutationTriggerCertificate",
    "MutationTriggerEvidence",
    "MutationTriggerInputError",
    "certify_mutation_trigger",
    "NonnormalPredictorRecord",
    "NonnormalPredictionCertificate",
    "deterministic_directed_family",
    "measure_nonnormal_pressure_prediction",
    "benchmark_nonnormal_prediction",
    "OperatorQuotientCertificate",
    "certify_operator_quotient",
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
