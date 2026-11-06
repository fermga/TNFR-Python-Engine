from __future__ import annotations

from typing import Any, TypeAlias

from ..node import NodeProtocol
from ..types import Glyph
from ._operator_growth_trigger import (
    OperatorGrowthTriggerCertificate as OperatorGrowthTriggerCertificate,
)
from .birth_access_coupling import (
    BirthAccessCouplingDispatch as BirthAccessCouplingDispatch,
)
from .birth_access_coupling import BirthAssociationAccess as BirthAssociationAccess
from .birth_access_coupling import (
    execute_birth_access_coupling as execute_birth_access_coupling,
)
from .birth_access_coupling import (
    observe_birth_association_access as observe_birth_association_access,
)
from .derived_expansion import DerivedExpansionDispatch as DerivedExpansionDispatch
from .derived_expansion import DerivedExpansionPolicy as DerivedExpansionPolicy
from .derived_expansion import (
    execute_derived_expansion_stage as execute_derived_expansion_stage,
)
from .emission_episode import EmissionEpisodeResult as EmissionEpisodeResult
from .emission_episode import execute_emission_episode as execute_emission_episode
from .emission_observation import (
    EmissionEpisodeObservation as EmissionEpisodeObservation,
)
from .emission_observation import observe_emission_episode as observe_emission_episode
from .event_remesh_causal_runtime import (
    CausalEventRemeshCycleReceipt as CausalEventRemeshCycleReceipt,
)
from .event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec as EventRemeshCycleExecutionSpec,
)
from .event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence as ExecutedEventRemeshCycleSequence,
)
from .event_remesh_causal_runtime import (
    execute_event_remesh_cycle_sequence as execute_event_remesh_cycle_sequence,
)
from .event_remesh_runtime import EventRemeshCycleResult as EventRemeshCycleResult
from .event_remesh_runtime import (
    RemeshHistoryTransitionObservation as RemeshHistoryTransitionObservation,
)
from .event_remesh_runtime import WeightedEPIObservation as WeightedEPIObservation
from .event_remesh_runtime import (
    execute_event_remesh_cycle as execute_event_remesh_cycle,
)
from .event_remesh_sequence import (
    EventRemeshCycleBoundaryObservation as EventRemeshCycleBoundaryObservation,
)
from .event_remesh_sequence import (
    ObservedEventRemeshCycleSequence as ObservedEventRemeshCycleSequence,
)
from .event_remesh_sequence import (
    compose_event_remesh_cycle_observations as compose_event_remesh_cycle_observations,
)
from .event_runtime import ExecutedGlyphStage as ExecutedGlyphStage
from .event_runtime import ExecutedNodalFlowInterval as ExecutedNodalFlowInterval
from .event_runtime import ExecutedOperatorEvent as ExecutedOperatorEvent
from .event_runtime import (
    ExecutedPressureRefreshedFlowPartition as ExecutedPressureRefreshedFlowPartition,
)
from .event_runtime import (
    ObservedRepresentedEPIScheduleComposition as _ObservedComposition,
)
from .event_runtime import OperatorEventExecutionResult as OperatorEventExecutionResult
from .event_runtime import (
    PhysicalEulerModalObservation as PhysicalEulerModalObservation,
)
from .event_runtime import (
    PressureRefreshBoundaryObservation as PressureRefreshBoundaryObservation,
)
from .event_runtime import (
    RepresentedEPIScheduleOperation as RepresentedEPIScheduleOperation,
)
from .event_runtime import (
    execute_operator_event_schedule as execute_operator_event_schedule,
)
from .event_timing import (
    OperatorEventRuntimeClockDiagnostic as OperatorEventRuntimeClockDiagnostic,
)
from .event_timing import OperatorEventSchedule as OperatorEventSchedule
from .event_timing import PhysicalFlowPartition as PhysicalFlowPartition
from .event_timing import ScheduledOperatorEvent as ScheduledOperatorEvent
from .event_timing import StructuralFlowInterval as StructuralFlowInterval
from .event_timing import build_operator_event_schedule as build_operator_event_schedule
from .event_timing import build_physical_flow_partition as build_physical_flow_partition
from .event_timing import (
    diagnose_operator_event_runtime_clock as diagnose_operator_event_runtime_clock,
)
from .factor_contracts import GLYPH_FACTOR_SPECS as GLYPH_FACTOR_SPECS
from .factor_contracts import GLYPH_FACTORS_BY_GLYPH as GLYPH_FACTORS_BY_GLYPH
from .factor_contracts import GlyphFactorSpec as GlyphFactorSpec
from .factor_contracts import GlyphFactorValidationError as GlyphFactorValidationError
from .factor_contracts import (
    canonical_glyph_factor_defaults as canonical_glyph_factor_defaults,
)
from .factor_contracts import resolve_operator_factors as resolve_operator_factors
from .factor_contracts import (
    resolve_runtime_operator_factors as resolve_runtime_operator_factors,
)
from .factor_contracts import (
    runtime_active_glyph_factor_keys as runtime_active_glyph_factor_keys,
)
from .factor_contracts import validate_glyph_factor as validate_glyph_factor
from .factor_contracts import validate_glyph_factors as validate_glyph_factors
from .grammar_evidence import StructuralGrammarEvidence as StructuralGrammarEvidence
from .grammar_evidence import (
    assess_structural_grammar_evidence as assess_structural_grammar_evidence,
)
from .grammar_observations import GrammarObservation as GrammarObservation
from .grammar_observations import observe_grammar as observe_grammar
from .native_emission_window import (
    NativeEmissionWindowDispatch as NativeEmissionWindowDispatch,
)
from .native_emission_window import (
    NativeEmissionWindowGrowth as NativeEmissionWindowGrowth,
)
from .native_emission_window import (
    NativeEmissionWindowSource as NativeEmissionWindowSource,
)
from .native_emission_window import (
    execute_native_emission_window as execute_native_emission_window,
)
from .native_positive_window import (
    NativePositiveWindowDispatch as NativePositiveWindowDispatch,
)
from .native_positive_window import (
    NativePositiveWindowGrowth as NativePositiveWindowGrowth,
)
from .native_positive_window import (
    NativePositiveWindowSource as NativePositiveWindowSource,
)
from .native_positive_window import (
    execute_native_positive_window as execute_native_positive_window,
)
from .operator_growth import OperatorEPIContribution as OperatorEPIContribution
from .operator_growth import (
    OperatorEPIGrowthObservation as OperatorEPIGrowthObservation,
)
from .operator_growth import observe_operator_epi_growth as observe_operator_epi_growth
from .operator_growth_mutation import (
    EligibleOperatorGrowthMutationDispatch as EligibleOperatorGrowthMutationDispatch,
)
from .operator_growth_mutation import (
    OperatorGrowthMutationCandidate as OperatorGrowthMutationCandidate,
)
from .operator_growth_mutation import (
    OperatorGrowthMutationDispatch as OperatorGrowthMutationDispatch,
)
from .operator_growth_mutation import (
    execute_eligible_operator_growth_mutation as execute_eligible_operator_growth_mutation,
)
from .operator_growth_mutation import (
    execute_operator_growth_mutation as execute_operator_growth_mutation,
)
from .remesh import DelayedRemeshNodeProposal as DelayedRemeshNodeProposal
from .remesh import DelayedRemeshPlan as DelayedRemeshPlan
from .remesh import DelayedRemeshResult as DelayedRemeshResult
from .remesh import DelayedRemeshStabilityEvidence as DelayedRemeshStabilityEvidence
from .remesh import apply_network_remesh as apply_network_remesh
from .remesh import plan_network_remesh as plan_network_remesh
from .retained_emission import RetainedEmissionDispatch as RetainedEmissionDispatch
from .retained_emission import (
    execute_retained_emission_stage as execute_retained_emission_stage,
)
from .retained_mutation import RetainedMutationDispatch as RetainedMutationDispatch
from .retained_mutation import (
    execute_retained_mutation_stage as execute_retained_mutation_stage,
)
from .retained_organization import RetainedMemberState as RetainedMemberState
from .retained_organization import RetainedOrganization as RetainedOrganization
from .retained_organization import RetainedScaleDispatch as RetainedScaleDispatch
from .retained_organization import (
    capture_retained_organization as capture_retained_organization,
)
from .retained_organization import (
    execute_retained_scale_stage as execute_retained_scale_stage,
)
from .retained_organization import (
    retained_organization_matches_graph as retained_organization_matches_graph,
)
from .retained_reception import RetainedInputPort as RetainedInputPort
from .retained_reception import RetainedOutputPort as RetainedOutputPort
from .retained_reception import (
    RetainedOutputReceptionDispatch as RetainedOutputReceptionDispatch,
)
from .retained_reception import RetainedReceptionDispatch as RetainedReceptionDispatch
from .retained_reception import (
    execute_retained_output_reception_stage as execute_retained_output_reception_stage,
)
from .retained_reception import (
    execute_retained_reception_stage as execute_retained_reception_stage,
)
from .retained_reception import retained_input_ports as retained_input_ports
from .retained_reception import retained_output_ports as retained_output_ports
from .retained_synthesis import RetainedPairState as RetainedPairState
from .retained_synthesis import RetainedSynthesisDispatch as RetainedSynthesisDispatch
from .retained_synthesis import capture_retained_pair as capture_retained_pair
from .retained_synthesis import (
    execute_retained_synthesis_stage as execute_retained_synthesis_stage,
)
from .self_organization_selection import (
    EligibleSelfOrganizationDispatch as EligibleSelfOrganizationDispatch,
)
from .self_organization_selection import (
    SelfOrganizationCandidate as SelfOrganizationCandidate,
)
from .self_organization_selection import (
    SelfOrganizationEligibility as SelfOrganizationEligibility,
)
from .self_organization_selection import (
    execute_eligible_self_organization_stage as execute_eligible_self_organization_stage,
)
from .self_organization_selection import (
    observe_self_organization_eligibility as observe_self_organization_eligibility,
)
from .word_execution import (
    preflight_network_mutation_sequence as preflight_network_mutation_sequence,
)
from .word_execution import run_network_sequence as run_network_sequence

ObservedRepresentedEPIScheduleComposition: TypeAlias = _ObservedComposition

Operator: Any
Emission: Any
Reception: Any
Coherence: Any
Dissonance: Any
Coupling: Any
Resonance: Any
Silence: Any
Expansion: Any
Contraction: Any
SelfOrganization: Any
Mutation: Any
Transition: Any
Recursivity: Any
GLYPH_OPERATIONS: Any
JitterCache: Any
JitterCacheManager: Any
OPERATORS: Any
apply_glyph: Any
apply_glyph_obj: Any
apply_remesh_if_globally_stable: Any
apply_topological_remesh: Any

def get_glyph_factors(
    node: NodeProtocol, glyph: Glyph | str | None = ...
) -> dict[str, Any]: ...

get_jitter_manager: Any
get_neighbor_epi: Any
random_jitter: Any
reset_jitter_manager: Any
