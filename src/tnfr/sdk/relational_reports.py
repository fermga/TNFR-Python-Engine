"""Exact JSON projections of detached relational model reports.

Projection preserves rational evidence and node order. It neither authenticates
the report nor reconstructs a live graph, model admission or execution history.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from fractions import Fraction
from typing import Any

__all__ = ("relational_report_to_dict",)


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
    from ..dynamics.relational import (
        RelationalConsensusTangent,
        RelationalExchangeField,
        RelationalExchangeStep,
        RelationalUniformTangent,
    )
    from ..physics.phase_cycle_geometry import (
        ReturnPathGeometryResponseAssessment,
        ReturnPathStorageGeometryAssessment,
    )
    from ..physics.phase_response import (
        PhaseInformationResponse,
        PhaseMomentInformationAssessment,
        PhaseMomentMotion,
        SineStarMomentClosure,
    )
    from ..physics.relational_bridge_discrimination import (
        BridgeClockLawDiscrimination,
        BridgeFiniteLawDiscrimination,
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
    from ..physics.relational_phase_storage import SaddleStorageDiscriminator
    from ..physics.relational_sine_aperture_budget import SineApertureBudget
    from ..physics.relational_sine_aperture_inference import SineApertureInference
    from ..physics.relational_sine_aperture_readout import SineApertureReadout
    from ..physics.relational_sine_bridge_memory import SineBridgeMemoryAssessment
    from ..physics.relational_sine_budget import SineBudgetConsensus
    from ..physics.relational_sine_class_amplitude_feasibility import (
        SineClassAmplitudeFeasibility,
    )
    from ..physics.relational_sine_class_cubic_response import SineClassCubicResponse
    from ..physics.relational_sine_class_mediation import SineClassMediation
    from ..physics.relational_sine_class_memory import (
        SineClassMediatedMemory,
        SineClassMediatedMemoryBound,
    )
    from ..physics.relational_sine_class_nonlinear_organization import (
        SineClassNonlinearOrganization,
    )
    from ..physics.relational_sine_class_nonlinear_protocol import (
        SineClassNonlinearProtocol,
    )
    from ..physics.relational_sine_class_readout import SineClassFourHistoryReadout
    from ..physics.relational_sine_class_spatial_observation import (
        SineClassSpatialObservation,
    )
    from ..physics.relational_sine_class_superposition import SineClassSuperposition
    from ..physics.relational_sine_clock_drift_inference import SineClockDriftInference
    from ..physics.relational_sine_clock_inference import SineClockInference
    from ..physics.relational_sine_comparison import (
        SineFormIncrementAssessment,
        SineMobilityComparison,
        SineMobilityRelativeBalance,
        SineRegionalStorageBalance,
        SineRegionalTransfer,
    )
    from ..physics.relational_sine_composition import (
        SinePatternComposition,
        SinePreparedComposition,
    )
    from ..physics.relational_sine_corridor import (
        SineSaddleCorridor,
        SineSaddleFormation,
        SineSaddlePreparation,
        SineSaddleRetentionBand,
    )
    from ..physics.relational_sine_curvature_inference import SineCurvatureInference
    from ..physics.relational_sine_entry import (
        SineConservativeHandoff,
        SineConservativePhaseTransport,
        SineConservativeSourceGeometry,
        SineConservativeWindingEntry,
        SinePreparedEntry,
    )
    from ..physics.relational_sine_formation_response import SineFormationResponse
    from ..physics.relational_sine_formed_class_contact import SineFormedClassContact
    from ..physics.relational_sine_formed_class_maintenance import (
        SineFormedClassMaintenance,
    )
    from ..physics.relational_sine_formed_classes import (
        SineFormedClassPair,
        SineFormedClassResponse,
    )
    from ..physics.relational_sine_metric_connection import SineMetricConnection
    from ..physics.relational_sine_metric_forecast import SineSaddleMetricForecast
    from ..physics.relational_sine_pair import (
        SineGlobalPairState,
        SinePairCancellationObservation,
        SinePairFiniteExchange,
        SinePairPersistentResponse,
        SinePairReceiverConfounding,
        SinePairReceiverDefect,
        SinePairReceiverReadout,
        SinePairReceiverTwoLaw,
        SinePairReceiverTwoTime,
    )
    from ..physics.relational_sine_partition import (
        SineCollectivePulseBalance,
        SineContactAveraging,
        SineMovingPatternWindow,
        SinePhaseOffsetPartition,
        SinePhaseOffsetState,
    )
    from ..physics.relational_sine_port_composition import (
        SinePortComposition,
        SinePortCompositionState,
    )
    from ..physics.relational_sine_port_form_tracking import SinePortFormTracking
    from ..physics.relational_sine_port_relaxation import SinePortRelaxation
    from ..physics.relational_sine_recovery import SineCycleIdentityAssessment
    from ..physics.relational_sine_reduced_class_ports import (
        SineReducedClassPorts,
        SineReducedClassPortState,
    )
    from ..physics.relational_sine_reduction import SineSlowCapture, SineSlowPhaseBound
    from ..physics.relational_sine_regional import (
        SineCycleBarrier,
        SineCycleRetention,
        SineRegionalChannelHistory,
        SineRegionalOrganization,
        SineReversiblePreparation,
    )
    from ..physics.relational_sine_replica_pulse import (
        SineReplicaPulseAssessment,
        SineReplicaPulseFiniteWorkResponse,
        SineReplicaPulseSplitting,
        SineReplicaPulseVariation,
        SineReplicaPulseWorkResponse,
        SineReplicaStiffnessTraceCurve,
    )
    from ..physics.relational_sine_resonance import (
        BridgeStorageFamilyAssessment,
        SineBridgeChannelAssessment,
        SineC5LeafSaddle,
        SineCycleResonance,
        SineMediatedResponse,
        SineModeGain,
        SinePairPulseAssessment,
        SinePathMemoryAssessment,
        SineRecoveryResonance,
        SineRecurrenceAssessment,
    )
    from ..physics.relational_sine_scale import (
        JointPairObservation,
        PhasePairObservation,
        SineJointPairingProjection,
        SineJointPairingWindowAssessment,
        SineMixedPairStateAssessment,
        SineMobilityGeometryAssessment,
        SinePairEmissionAssessment,
        SinePairingMobilityAssessment,
        SinePairingTransitionAssessment,
        SinePairingWindowAssessment,
        SinePairSupportSymmetryAssessment,
        SineReplicaCapacityAssessment,
        SineReplicaEquilibriaAssessment,
        SineReplicaPersistenceAssessment,
        SineReplicaScaleAssessment,
        SineStatePairingAssessment,
    )
    from ..physics.relational_sine_sensitivity import SineSaddleSensitivity
    from ..physics.relational_sine_symmetry import (
        SineCycleSymmetryAssessment,
        SineInvolutionReduction,
        SineInvolutionState,
    )
    from ..physics.relational_sine_two_port_capture import SineTwoPortCapture
    from ..physics.relational_sine_two_port_compatibility import (
        SineTwoPortCompatibility,
        SineTwoPortHandoffObstruction,
    )
    from ..physics.relational_sine_two_port_dipole import SineTwoPortDipole
    from ..physics.relational_sine_two_port_inference import SineTwoPortInference
    from ..physics.relational_sine_two_port_probe import SineTwoPortProbe
    from ..physics.relational_sine_two_port_readout import SineTwoPortReadout
    from ..physics.relational_sine_two_port_transit import SineTwoPortTransit
    from ..physics.relational_sine_two_pulse_inference import SineTwoPulseInference
    from ..physics.relational_transit import RelationalTransitCertificate
    from ..research.sine_constitutive_robustness import SineConstitutiveRobustness

    if isinstance(
        report,
        (
            PhaseInformationResponse,
            PhaseMomentInformationAssessment,
            PhaseMomentMotion,
            SineStarMomentClosure,
            ReturnPathStorageGeometryAssessment,
            ReturnPathGeometryResponseAssessment,
            SineConservativeHandoff,
            SineConservativePhaseTransport,
            SineConservativeSourceGeometry,
            SineConservativeWindingEntry,
            SineRegionalStorageBalance,
            SineCycleBarrier,
            SineCycleRetention,
            SineReversiblePreparation,
            SineRegionalOrganization,
            SineRegionalChannelHistory,
            SineBudgetConsensus,
            SineBridgeMemoryAssessment,
            SinePatternComposition,
            SinePreparedComposition,
            SinePreparedEntry,
            SineCollectivePulseBalance,
            SineContactAveraging,
            SineMovingPatternWindow,
            SinePhaseOffsetPartition,
            SinePhaseOffsetState,
            SineSlowCapture,
            SineSlowPhaseBound,
            SineCycleSymmetryAssessment,
            SineInvolutionReduction,
            SineInvolutionState,
            SineBridgeChannelAssessment,
            SineC5LeafSaddle,
            SineSaddleCorridor,
            SineSaddleFormation,
            SineSaddlePreparation,
            SaddleStorageDiscriminator,
            SineConstitutiveRobustness,
            SineSaddleRetentionBand,
            SineSaddleSensitivity,
            SineSaddleMetricForecast,
            SineMetricConnection,
            BridgeStorageFamilyAssessment,
            BridgeFiniteLawDiscrimination,
            BridgeClockLawDiscrimination,
            SineCycleResonance,
            SineModeGain,
            SineRecoveryResonance,
            SineMediatedResponse,
            SinePairPulseAssessment,
            SinePathMemoryAssessment,
            SineRecurrenceAssessment,
            SineCycleIdentityAssessment,
            SineFormIncrementAssessment,
            SineMobilityComparison,
            SineMobilityRelativeBalance,
            SineRegionalTransfer,
            SineMobilityGeometryAssessment,
            SineReplicaScaleAssessment,
            SineReplicaStiffnessTraceCurve,
            SineGlobalPairState,
            SinePairCancellationObservation,
            SinePairFiniteExchange,
            SinePairPersistentResponse,
            SineFormationResponse,
            SineFormedClassContact,
            SineReducedClassPorts,
            SineReducedClassPortState,
            SinePortComposition,
            SinePortCompositionState,
            SinePortRelaxation,
            SinePortFormTracking,
            SineClassMediation,
            SineClassMediatedMemory,
            SineClassMediatedMemoryBound,
            SineClassSuperposition,
            SineClassCubicResponse,
            SineClassSpatialObservation,
            SineClassAmplitudeFeasibility,
            SineClassNonlinearOrganization,
            SineClassNonlinearProtocol,
            SineClassFourHistoryReadout,
            SineTwoPortCompatibility,
            SineTwoPortHandoffObstruction,
            SineTwoPortCapture,
            SineTwoPortDipole,
            SineTwoPortInference,
            SineTwoPortProbe,
            SineTwoPortReadout,
            SineTwoPortTransit,
            SineTwoPulseInference,
            SineClockInference,
            SineCurvatureInference,
            SineClockDriftInference,
            SineApertureBudget,
            SineApertureInference,
            SineApertureReadout,
            SineFormedClassMaintenance,
            SineFormedClassPair,
            SineFormedClassResponse,
            SinePairReceiverConfounding,
            SinePairReceiverDefect,
            SinePairReceiverReadout,
            SinePairReceiverTwoTime,
            SinePairReceiverTwoLaw,
            JointPairObservation,
            PhasePairObservation,
            SineJointPairingProjection,
            SineJointPairingWindowAssessment,
            SineMixedPairStateAssessment,
            SinePairEmissionAssessment,
            SinePairSupportSymmetryAssessment,
            SinePairingMobilityAssessment,
            SinePairingTransitionAssessment,
            SinePairingWindowAssessment,
            SineStatePairingAssessment,
            SineReplicaCapacityAssessment,
            SineReplicaEquilibriaAssessment,
            SineReplicaPersistenceAssessment,
            SineReplicaPulseAssessment,
            SineReplicaPulseFiniteWorkResponse,
            SineReplicaPulseSplitting,
            SineReplicaPulseVariation,
            SineReplicaPulseWorkResponse,
        ),
    ):
        return {
            "schema": "tnfr.relational-report.v1",
            "report_type": type(report).__name__,
            "report": report.to_dict()["report"],
        }

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
