"""Exact JSON projections of detached relational model reports.

Projection preserves rational evidence and node order. It neither authenticates
the report nor reconstructs a live graph, model admission or execution history.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from fractions import Fraction
from importlib import import_module
from typing import Any

__all__ = ("relational_report_to_dict",)


# One dispatch registry: values are trusted import owners, never report metadata.
# The owning class identity is checked through the actual MRO before delegation.
_OWNER_MANAGED_REPORTS = {
    "BridgeClockLawDiscrimination": "tnfr.physics.relational_bridge_discrimination",
    "BridgeFiniteLawDiscrimination": "tnfr.physics.relational_bridge_discrimination",
    "BridgeStorageFamilyAssessment": "tnfr.physics.relational_sine_resonance",
    "JointPairObservation": "tnfr.physics.relational_sine_scale",
    "PhaseInformationResponse": "tnfr.physics.phase_response",
    "PhaseMomentInformationAssessment": "tnfr.physics.phase_response",
    "PhaseMomentMotion": "tnfr.physics.phase_response",
    "PhasePairObservation": "tnfr.physics.relational_sine_scale",
    "ReturnPathGeometryResponseAssessment": "tnfr.physics.phase_cycle_geometry",
    "ReturnPathStorageGeometryAssessment": "tnfr.physics.phase_cycle_geometry",
    "SaddleStorageDiscriminator": "tnfr.physics.relational_phase_storage",
    "SineApertureBudget": "tnfr.physics.relational_sine_aperture_budget",
    "SineApertureInference": "tnfr.physics.relational_sine_aperture_inference",
    "SineApertureReadout": "tnfr.physics.relational_sine_aperture_readout",
    "SineBridgeChannelAssessment": "tnfr.physics.relational_sine_resonance",
    "SineBridgeMemoryAssessment": "tnfr.physics.relational_sine_bridge_memory",
    "SineBudgetConsensus": "tnfr.physics.relational_sine_budget",
    "SineC5LeafSaddle": "tnfr.physics.relational_sine_resonance",
    "SineClassAmplitudeFeasibility": "tnfr.physics.relational_sine_class_amplitude_feasibility",
    "SineClassComparisonReadout": "tnfr.physics.relational_sine_class_comparison_readout",
    "SineClassCubicResponse": "tnfr.physics.relational_sine_class_cubic_response",
    "SineClassFourHistoryReadout": "tnfr.physics.relational_sine_class_readout",
    "SineClassMediatedMemory": "tnfr.physics.relational_sine_class_memory",
    "SineClassMediatedMemoryBound": "tnfr.physics.relational_sine_class_memory",
    "SineClassMediation": "tnfr.physics.relational_sine_class_mediation",
    "SineClassNonlinearOrganization": "tnfr.physics.relational_sine_class_nonlinear_organization",
    "SineClassNonlinearProtocol": "tnfr.physics.relational_sine_class_nonlinear_protocol",
    "SineClassPortReadout": "tnfr.physics.relational_sine_class_port_readout",
    "SineClassSpatialObservation": "tnfr.physics.relational_sine_class_spatial_observation",
    "SineClassStorageReadout": "tnfr.physics.relational_sine_class_storage_readout",
    "SineClassSuperposition": "tnfr.physics.relational_sine_class_superposition",
    "SineClockDriftInference": "tnfr.physics.relational_sine_clock_drift_inference",
    "SineClockInference": "tnfr.physics.relational_sine_clock_inference",
    "SineCollectivePulseBalance": "tnfr.physics.relational_sine_partition",
    "SineConservativeHandoff": "tnfr.physics.relational_sine_entry",
    "SineConservativePhaseTransport": "tnfr.physics.relational_sine_entry",
    "SineConservativeSourceGeometry": "tnfr.physics.relational_sine_entry",
    "SineConservativeWindingEntry": "tnfr.physics.relational_sine_entry",
    "SineConstitutiveRobustness": "tnfr.research.sine_constitutive_robustness",
    "SineContactAveraging": "tnfr.physics.relational_sine_partition",
    "SineCurvatureInference": "tnfr.physics.relational_sine_curvature_inference",
    "SineCycleBarrier": "tnfr.physics.relational_sine_regional",
    "SineCycleIdentityAssessment": "tnfr.physics.relational_sine_recovery",
    "SineCycleResonance": "tnfr.physics.relational_sine_resonance",
    "SineCycleRetention": "tnfr.physics.relational_sine_regional",
    "SineCycleSymmetryAssessment": "tnfr.physics.relational_sine_symmetry",
    "SineFormIncrementAssessment": "tnfr.physics.relational_sine_comparison",
    "SineFormationResponse": "tnfr.physics.relational_sine_formation_response",
    "SineFormedClassContact": "tnfr.physics.relational_sine_formed_class_contact",
    "SineFormedClassMaintenance": "tnfr.physics.relational_sine_formed_class_maintenance",
    "SineFormedClassPair": "tnfr.physics.relational_sine_formed_classes",
    "SineFormedClassResponse": "tnfr.physics.relational_sine_formed_classes",
    "SineGlobalPairState": "tnfr.physics.relational_sine_pair",
    "SineInvolutionReduction": "tnfr.physics.relational_sine_symmetry",
    "SineInvolutionState": "tnfr.physics.relational_sine_symmetry",
    "SineJointPairingProjection": "tnfr.physics.relational_sine_scale",
    "SineJointPairingWindowAssessment": "tnfr.physics.relational_sine_scale",
    "SineMediatedResponse": "tnfr.physics.relational_sine_resonance",
    "SineMetricConnection": "tnfr.physics.relational_sine_metric_connection",
    "SineMixedPairStateAssessment": "tnfr.physics.relational_sine_scale",
    "SineMobilityComparison": "tnfr.physics.relational_sine_comparison",
    "SineMobilityGeometryAssessment": "tnfr.physics.relational_sine_scale",
    "SineMobilityRelativeBalance": "tnfr.physics.relational_sine_comparison",
    "SineModeGain": "tnfr.physics.relational_sine_resonance",
    "SineMovingPatternWindow": "tnfr.physics.relational_sine_partition",
    "SinePairCancellationObservation": "tnfr.physics.relational_sine_pair",
    "SinePairEmissionAssessment": "tnfr.physics.relational_sine_scale",
    "SinePairFiniteExchange": "tnfr.physics.relational_sine_pair",
    "SinePairPersistentResponse": "tnfr.physics.relational_sine_pair",
    "SinePairPulseAssessment": "tnfr.physics.relational_sine_resonance",
    "SinePairReceiverConfounding": "tnfr.physics.relational_sine_pair",
    "SinePairReceiverDefect": "tnfr.physics.relational_sine_pair",
    "SinePairReceiverReadout": "tnfr.physics.relational_sine_pair",
    "SinePairReceiverTwoLaw": "tnfr.physics.relational_sine_pair",
    "SinePairReceiverTwoTime": "tnfr.physics.relational_sine_pair",
    "SinePairSupportSymmetryAssessment": "tnfr.physics.relational_sine_scale",
    "SinePairingMobilityAssessment": "tnfr.physics.relational_sine_scale",
    "SinePairingTransitionAssessment": "tnfr.physics.relational_sine_scale",
    "SinePairingWindowAssessment": "tnfr.physics.relational_sine_scale",
    "SinePathMemoryAssessment": "tnfr.physics.relational_sine_resonance",
    "SinePatternComposition": "tnfr.physics.relational_sine_composition",
    "SinePhaseOffsetPartition": "tnfr.physics.relational_sine_partition",
    "SinePhaseOffsetState": "tnfr.physics.relational_sine_partition",
    "SinePortComposition": "tnfr.physics.relational_sine_port_composition",
    "SinePortCompositionState": "tnfr.physics.relational_sine_port_composition",
    "SinePortFormTracking": "tnfr.physics.relational_sine_port_form_tracking",
    "SinePortRelaxation": "tnfr.physics.relational_sine_port_relaxation",
    "SinePreparedComposition": "tnfr.physics.relational_sine_composition",
    "SinePreparedEntry": "tnfr.physics.relational_sine_entry",
    "SineRecoveryResonance": "tnfr.physics.relational_sine_resonance",
    "SineRecurrenceAssessment": "tnfr.physics.relational_sine_resonance",
    "SineReducedClassPortState": "tnfr.physics.relational_sine_reduced_class_ports",
    "SineReducedClassPorts": "tnfr.physics.relational_sine_reduced_class_ports",
    "SineRegionalChannelHistory": "tnfr.physics.relational_sine_regional",
    "SineRegionalOrganization": "tnfr.physics.relational_sine_regional",
    "SineRegionalStorageBalance": "tnfr.physics.relational_sine_comparison",
    "SineRegionalTransfer": "tnfr.physics.relational_sine_comparison",
    "SineReplicaCapacityAssessment": "tnfr.physics.relational_sine_scale",
    "SineReplicaEquilibriaAssessment": "tnfr.physics.relational_sine_scale",
    "SineReplicaPersistenceAssessment": "tnfr.physics.relational_sine_scale",
    "SineReplicaPulseAssessment": "tnfr.physics.relational_sine_replica_pulse",
    "SineReplicaPulseFiniteWorkResponse": "tnfr.physics.relational_sine_replica_pulse",
    "SineReplicaPulseSplitting": "tnfr.physics.relational_sine_replica_pulse",
    "SineReplicaPulseVariation": "tnfr.physics.relational_sine_replica_pulse",
    "SineReplicaPulseWorkResponse": "tnfr.physics.relational_sine_replica_pulse",
    "SineReplicaScaleAssessment": "tnfr.physics.relational_sine_scale",
    "SineReplicaStiffnessTraceCurve": "tnfr.physics.relational_sine_replica_pulse",
    "SineReversiblePreparation": "tnfr.physics.relational_sine_regional",
    "SineSaddleCorridor": "tnfr.physics.relational_sine_corridor",
    "SineSaddleFormation": "tnfr.physics.relational_sine_corridor",
    "SineSaddleMetricForecast": "tnfr.physics.relational_sine_metric_forecast",
    "SineSaddlePreparation": "tnfr.physics.relational_sine_corridor",
    "SineSaddleRetentionBand": "tnfr.physics.relational_sine_corridor",
    "SineSaddleSensitivity": "tnfr.physics.relational_sine_sensitivity",
    "SineSlowCapture": "tnfr.physics.relational_sine_reduction",
    "SineSlowPhaseBound": "tnfr.physics.relational_sine_reduction",
    "SineStarMomentClosure": "tnfr.physics.phase_response",
    "SineStatePairingAssessment": "tnfr.physics.relational_sine_scale",
    "SineTwoPortCapture": "tnfr.physics.relational_sine_two_port_capture",
    "SineTwoPortCompatibility": "tnfr.physics.relational_sine_two_port_compatibility",
    "SineTwoPortDipole": "tnfr.physics.relational_sine_two_port_dipole",
    "SineTwoPortHandoffObstruction": "tnfr.physics.relational_sine_two_port_compatibility",
    "SineTwoPortInference": "tnfr.physics.relational_sine_two_port_inference",
    "SineTwoPortProbe": "tnfr.physics.relational_sine_two_port_probe",
    "SineTwoPortReadout": "tnfr.physics.relational_sine_two_port_readout",
    "SineTwoPortTransit": "tnfr.physics.relational_sine_two_port_transit",
    "SineTwoPulseInference": "tnfr.physics.relational_sine_two_pulse_inference",
}


def _is_owner_managed_report(report: Any) -> bool:
    """Recognize registered classes and their real subclasses lazily.

    A matching name/module only selects a trusted owner; it is not admission.
    An arbitrary object exposing ``to_dict`` remains unsupported.
    """
    # Bypass metaclass attribute overrides and metaclass MRO properties alike.
    for base in type.__dict__["__mro__"].__get__(type(report)):
        owner_name = _OWNER_MANAGED_REPORTS.get(base.__name__)
        if owner_name is not None and base.__module__ == owner_name:
            owner = import_module(owner_name)
            if base is getattr(owner, base.__name__):
                return True
    return False


def _validate_label(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _validate_label(item)
    elif value is None or type(value) in (bool, int, str):
        return
    elif type(value) is float and math.isfinite(value):
        return
    else:
        raise TypeError(
            "report export requires JSON scalar node labels or tuples of them"
        )


def _project(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": int(value.numerator),
            "denominator": int(value.denominator),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _project(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_project(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("relational report contains a nonfinite float")
        return value
    raise TypeError(
        f"unsupported relational report value or node label: {type(value).__name__}; "
        "use JSON scalar labels or tuples of them"
    )


def _validate_label_groups(*groups):
    for labels in groups:
        for node in labels:
            _validate_label(node)


def _validate_cut_labels(cut):
    _validate_label_groups(cut.nodes, cut.region, cut.environment)
    for a, b, _ in cut.cut_edges:
        _validate_label(a)
        _validate_label(b)


def _validate_pattern_labels(report):
    for region in report.regions:
        _validate_label_groups(region.nodes)
        if region.transport is not None:
            transport = region.transport
            _validate_label_groups(
                transport.source.nodes, transport.region, transport.environment
            )
        if region.boundary is not None:
            _validate_cut_labels(region.boundary.cut)


def _validate_support_change_labels(report, bridges, cuts, partitions=()):
    _validate_label_groups(*bridges, *partitions)
    for port in report.ports:
        _validate_label(port.before.node)
        _validate_label(port.after.node)
    for snapshot in (report.transport_reset.before, report.transport_reset.after):
        _validate_label_groups(snapshot.nodes)
    for cut in cuts:
        _validate_cut_labels(cut)


def _reset_states(report):
    """Share label validation for standalone and nested support budgets."""
    for edge in (*report.edges_before, *report.edges_after):
        for node in edge:
            _validate_label(node)
    return (
        report.before,
        report.after,
        report.transport_reset.before,
        report.transport_reset.after,
    )


def relational_report_to_dict(report: Any) -> dict[str, Any]:
    """Project a supported relational report without changing its verdicts.

    The envelope has schema ``tnfr.relational-report.v1``, its concrete
    ``report_type`` and an exact projected ``report`` body. Owner-managed
    reports delegate to their ``to_dict`` method, preserving label admission,
    nested evidence, unavailable values and model-specific scope.

    Fractions become ``{numerator, denominator}`` records; tuples become
    ordered arrays, including tuple node labels. Opaque labels and nonfinite
    numbers reject. Save with ``export_to_json(relational_report_to_dict(report),
    path)`` to use the shared atomic writer. Projection is detached, not a
    typed round trip, resumable checkpoint or provenance authentication.
    No theorem eligibility is recomputed and no producer or trajectory runs.
    """
    if _is_owner_managed_report(report):
        return {
            "schema": "tnfr.relational-report.v1",
            "report_type": type(report).__name__,
            "report": report.to_dict()["report"],
        }

    from ..dynamics.relational import (
        RelationalConsensusTangent,
        RelationalExchangeField,
        RelationalExchangeStep,
        RelationalUniformTangent,
    )
    from ..physics.relational_capture import (
        RelationalCaptureCertificate,
        RelationalConsensusCaptureCertificate,
        RelationalConsensusFormationObstruction,
        RelationalCycleCaptureCertificate,
        RelationalDetachmentObservation,
        RelationalLocalCaptureCertificate,
        RelationalSectorCaptureCertificate,
        RelationalSectorGeometry,
        RelationalSeededFormationObstruction,
    )
    from ..physics.relational_cycle_memory import (
        RelationalCycleMemoryBounds,
        RelationalFiniteMemoryCertificate,
        RelationalMemoryReadoutCertificate,
    )
    from ..physics.relational_memory_contact import (
        RelationalMemoryContactCertificate,
        RelationalMemoryRetentionCertificate,
    )
    from ..physics.relational_observations import (
        RelationalAttachmentObservation,
        RelationalAttachmentSupplyAssessment,
        RelationalCoefficientJetBounds,
        RelationalCoefficientSampleBounds,
        RelationalJetSampleBounds,
        RelationalPatternObservation,
        RelationalRateContrastBounds,
        RelationalRateSampleBounds,
        RelationalRelocationObservation,
        RelationalResetObservation,
        RelationalSampleJetBudget,
    )
    from ..physics.relational_transit import RelationalTransitCertificate

    if not isinstance(
        report,
        (
            RelationalExchangeField,
            RelationalExchangeStep,
            RelationalUniformTangent,
            RelationalConsensusTangent,
            RelationalPatternObservation,
            RelationalAttachmentObservation,
            RelationalAttachmentSupplyAssessment,
            RelationalCoefficientJetBounds,
            RelationalCoefficientSampleBounds,
            RelationalJetSampleBounds,
            RelationalRateContrastBounds,
            RelationalRateSampleBounds,
            RelationalRelocationObservation,
            RelationalResetObservation,
            RelationalSampleJetBudget,
            RelationalCaptureCertificate,
            RelationalConsensusCaptureCertificate,
            RelationalConsensusFormationObstruction,
            RelationalCycleCaptureCertificate,
            RelationalDetachmentObservation,
            RelationalCycleMemoryBounds,
            RelationalFiniteMemoryCertificate,
            RelationalMemoryReadoutCertificate,
            RelationalMemoryContactCertificate,
            RelationalMemoryRetentionCertificate,
            RelationalLocalCaptureCertificate,
            RelationalSeededFormationObstruction,
            RelationalSectorCaptureCertificate,
            RelationalSectorGeometry,
            RelationalTransitCertificate,
        ),
    ):
        raise TypeError(
            "expected a relational field, step, joint tangent, pattern, attachment, relocation, reset, supply assessment, "
            "coefficient-jet, coefficient-sample, rate-contrast, sample-rate, sample-jet, sample-jet-budget, cycle-memory, sector-geometry, capture, detachment or continuous-transit report"
        )
    observation = (
        report.initial
        if isinstance(
            report,
            (RelationalTransitCertificate, RelationalConsensusCaptureCertificate),
        )
        else report
    )
    if isinstance(
        observation,
        (
            RelationalAttachmentSupplyAssessment,
            RelationalCoefficientJetBounds,
            RelationalCoefficientSampleBounds,
            RelationalJetSampleBounds,
            RelationalRateContrastBounds,
            RelationalRateSampleBounds,
            RelationalSampleJetBudget,
            RelationalCycleMemoryBounds,
            RelationalFiniteMemoryCertificate,
            RelationalMemoryReadoutCertificate,
            RelationalMemoryContactCertificate,
            RelationalMemoryRetentionCertificate,
            RelationalSeededFormationObstruction,
        ),
    ):
        states = ()
    elif isinstance(observation, RelationalDetachmentObservation):
        states = (
            observation.before,
            *(component.field for component in observation.components),
            *_reset_states(observation.reset),
        )
        _validate_label_groups(
            *observation.removed_bridges,
            *(component.cycle for component in observation.components),
        )
    elif isinstance(observation, RelationalConsensusFormationObstruction):
        states = (observation.field,)
        _validate_label_groups(*observation.cycles)
    elif isinstance(observation, RelationalCycleCaptureCertificate):
        states = (observation.field,)
        _validate_label_groups(observation.cycle)
    elif isinstance(observation, RelationalSectorGeometry):
        states = (observation,)
        # Public dataclasses can be constructed with labels outside their nodes.
        _validate_label_groups(*observation.cycles, observation.bridge_cycle)
    elif isinstance(observation, RelationalAttachmentObservation):
        states = (*observation.components, observation.joined)
        _validate_support_change_labels(
            observation, (observation.bridge,), (observation.cut,)
        )
    elif isinstance(observation, RelationalRelocationObservation):
        states = (observation.before, observation.after)
        _validate_support_change_labels(
            observation,
            (observation.remove_bridge, observation.add_bridge),
            (observation.cut_before, observation.cut_after),
            observation.components,
        )
    elif isinstance(observation, RelationalResetObservation):
        states = _reset_states(observation)
    elif isinstance(
        observation, (RelationalUniformTangent, RelationalConsensusTangent)
    ):
        states = (observation.field,)
    elif isinstance(observation, RelationalExchangeStep):
        states = (observation.before, observation.after)
    elif isinstance(
        observation,
        (
            RelationalPatternObservation,
            RelationalCaptureCertificate,
            RelationalLocalCaptureCertificate,
            RelationalSectorCaptureCertificate,
        ),
    ):
        states = (observation.field,)
        # Undefined cycles can retain nodes absent from the admitted graph.
        # Those labels must not bypass admission through dataclass projection.
        for certificate in observation.winding:
            _validate_label_groups(certificate.cycle_nodes)
        if isinstance(observation, RelationalPatternObservation):
            _validate_pattern_labels(observation)
        else:
            _validate_label_groups(*observation.cycles)
            if isinstance(observation, RelationalSectorCaptureCertificate):
                _validate_label_groups(observation.bridge_cycle)
    else:
        states = (observation,)
    for state in states:
        _validate_label_groups(state.nodes)
        if isinstance(state, (RelationalExchangeField, RelationalSectorGeometry)):
            _validate_label_groups(*state.edges)
    projected = _project(report)
    if isinstance(
        report,
        (
            RelationalFiniteMemoryCertificate,
            RelationalMemoryReadoutCertificate,
            RelationalMemoryContactCertificate,
            RelationalMemoryRetentionCertificate,
        ),
    ):
        projected["admitted"] = report.admitted
    if isinstance(report, RelationalDetachmentObservation):
        projected["capture_admitted"] = report.capture_admitted
    if isinstance(
        report, (RelationalAttachmentObservation, RelationalRelocationObservation)
    ):
        projected["continuous_loss_change"] = _project(report.continuous_loss_change)
    if isinstance(
        report,
        (
            RelationalAttachmentObservation,
            RelationalRelocationObservation,
            RelationalResetObservation,
        ),
    ):
        projected["represented_zero_supply_passive"] = (
            report.represented_zero_supply_passive
        )
    return {
        "schema": "tnfr.relational-report.v1",
        "report_type": type(report).__name__,
        "report": projected,
    }
