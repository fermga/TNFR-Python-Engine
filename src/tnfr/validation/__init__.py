"""Public adapters for distinct input, execution and observation checks.

Scalar adapters reuse ``input_validation``; grammar exports reuse the operator
owners. ``TNFRValidator`` orchestrates selected checks. ``ValidationConfig``
configures the unified input validator; ``StructuralValidationConfig`` and
``configure_validation`` describe the separate structural word policy.
Neither policy substitutes for a complete model's mathematical admission.

Forecast and measurement adapters retain their own preparation, clock and
evidence premises. Public usage and migration boundaries belong to
``docs/API_CONTRACTS.md``; the theory catalog identifies proof owners.
"""

from __future__ import annotations

from typing import Any

from ..operators import grammar as _grammar
from ..types import Glyph
from .base import SubjectT, ValidationOutcome, Validator  # noqa: F401

# Public views of the shared adjacency policy. Keep this import after grammar
# initialization because the glyph tables translate names through that module.
from .compatibility import (
    CANON_COMPAT,
    CANON_FALLBACK,
    GRADUATED_COMPATIBILITY,
    CompatibilityLevel,
    get_compatibility_level,
)
from .config import ValidationConfig as StructuralValidationConfig  # noqa: F401
from .config import configure_validation, validation_config  # noqa: F401
from .graph import GRAPH_VALIDATORS, run_validators  # noqa: F401

# Value-returning adapters share admission with the configured input pipeline.
from .input_validation import (
    validate_dnfr_value,
    validate_epi_value,
    validate_glyph,
    validate_glyph_factors,
    validate_node_id,
    validate_operator_parameters,
    validate_theta_value,
    validate_tnfr_graph,
    validate_vf_value,
)
from .interface_baselines import (  # noqa: F401
    BASELINE_FORMULAS,
    compute_all_baselines,
    constant_baseline,
    degree_score,
    feature_deviation,
    graph_cut_contribution,
    graph_total_variation,
    label_propagation_residual,
    local_class_entropy,
    local_disagreement,
    mean_neighbour_distance,
    random_baseline,
)
from .invariants import (  # noqa: F401
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant3_DNFRSemantics,
    Invariant4_OperatorClosure,
    Invariant5_ExplicitPhaseChecks,
    Invariant6_NodeBirthCollapse,
    Invariant7_OperationalFractality,
    Invariant8_ControlledDeterminism,
    Invariant9_StructuralMetrics,
    Invariant10_DomainNeutrality,
    InvariantSeverity,
    InvariantViolation,
    TNFRInvariant,
)
from .multichannel_interface import (  # noqa: F401
    MultichannelConfig,
    MultichannelWindowSeries,
    SynchronyDiscrimination,
    amplitude_pressure,
    analytic_phase_amplitude,
    build_coupling_graph,
    evaluate_synchrony_discrimination,
    fft_bandpass,
    kuramoto_order_parameter,
    mean_field_order,
    multichannel_window_series,
    phase_amplitude_matrices,
    phase_locking_matrix,
    phase_offsets,
)
from .nodal_prediction import (  # noqa: F401
    FrozenNodalCalibration,
    NodalCalibrationError,
    NodalForecast,
    NodalForecastScore,
    NodalMeasurementRun,
    calibrate_nodal_prediction,
    forecast_nodal_response,
    score_nodal_forecast,
    write_nodal_forecast,
)
from .p2_transport import (  # noqa: F401
    P2IntervalCalibration,
    P2IntervalComparison,
    P2IntervalForecast,
    P2MeasurementBounds,
    calibrate_p2_transport,
    forecast_p2_transport,
    score_p2_transport,
    write_p2_transport_forecast,
)
from .phase_gate import (  # noqa: F401
    DEFAULT_MIN_COMPLIANCE,
    DEFAULT_PHASE_GATE,
    PhaseGateCompliance,
    PhaseGateOperatorPrescription,
    PhaseGateReport,
    PhaseGateViolation,
    PhaseStressHotspot,
    analyze_phase_gate,
    compare_against_global_baselines,
    compute_edge_gate_compliance,
    detect_phase_gate_violations,
    export_phase_gate_report,
    prescribe_phase_gate_operators,
    rank_phase_stress_hotspots,
)
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr  # noqa: F401
from .runtime import (  # noqa: F401
    GraphCanonicalValidator,
    apply_canonical_clamps,
    validate_canon,
)
from .sequence_validator import SequenceSemanticValidator  # noqa: F401
from .signal_confrontation import (  # noqa: F401
    ModalRootDiagnostic,
    NodalPredictionSkill,
    SignalConfrontation,
    confront_signal,
    diagnose_modal_roots,
    emergent_wave_fraction,
    estimate_quality_factor,
    nodal_prediction_skill,
)
from .soft_filters import (  # noqa: F401
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .structural_interface import (  # noqa: F401
    StructuralInterfaceProblem,
    StructuralInterfaceScore,
    baseline_score_maps,
    build_knn_graph,
    encode_phase_from_binary_state,
    evaluate_interface_scores,
    export_structural_interface_report,
    full_baseline_score_maps,
    interface_score_maps,
    local_state_disagreement,
    render_structural_interface_html,
    render_structural_interface_markdown,
    score_structural_interfaces,
)
from .temporal_interface import (  # noqa: F401
    EarlyWarningComparison,
    ProspectiveWarningComparison,
    TemporalInterfaceConfig,
    TemporalWarningCalibration,
    WindowTetradSeries,
    build_temporal_proximity_graph,
    calibrate_temporal_warning,
    delay_embedding,
    evaluate_early_warning,
    evaluate_prospective_warning,
    hilbert_instantaneous_phase,
    kendall_tau,
    local_structural_pressure,
    rolling_lag1_autocorrelation,
    rolling_variance,
    window_tetrad_series,
)

# Unified validation system exports
from .unified_validation_system import (  # noqa: F401
    TNFRSecurityError,
    TNFRUnifiedValidationSystem,
    ValidationConfig,
    ValidationError,
    ValidationResult,
    get_unified_validation_stats,
    get_unified_validation_system,
    validate_coherence,
    validate_phase_value,
    validate_string_input,
    validate_structural_frequency,
)
from .validator import TNFRValidationError, TNFRValidator  # noqa: F401
from .window import validate_window  # noqa: F401

_GRAMMAR_EXPORTS = tuple(getattr(_grammar, "__all__", ()))

globals().update({name: getattr(_grammar, name) for name in _GRAMMAR_EXPORTS})

_RUNTIME_EXPORTS = (
    "ValidationOutcome",
    "Validator",
    "GraphCanonicalValidator",
    "apply_canonical_clamps",
    "validate_canon",
    "GRAPH_VALIDATORS",
    "run_validators",
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "validate_window",
    "coerce_glyph",
    "get_norm",
    "glyph_fallback",
    "normalized_dnfr",
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
    "NFRValidator",
    "ValidationError",
    "validate_epi_value",
    "validate_vf_value",
    "validate_theta_value",
    "validate_dnfr_value",
    "validate_node_id",
    "validate_glyph",
    "validate_tnfr_graph",
    "validate_glyph_factors",
    "validate_operator_parameters",
    "InvariantSeverity",
    "InvariantViolation",
    "TNFRInvariant",
    "Invariant1_EPIOnlyThroughOperators",
    "Invariant2_VfInHzStr",
    "Invariant3_DNFRSemantics",
    "Invariant4_OperatorClosure",
    "Invariant5_ExplicitPhaseChecks",
    "Invariant6_NodeBirthCollapse",
    "Invariant7_OperationalFractality",
    "Invariant8_ControlledDeterminism",
    "Invariant9_StructuralMetrics",
    "Invariant10_DomainNeutrality",
    "TNFRValidator",
    "TNFRValidationError",
    "SequenceSemanticValidator",
    "ValidationConfig",
    "StructuralValidationConfig",
    "validation_config",
    "configure_validation",
    "DEFAULT_MIN_COMPLIANCE",
    "DEFAULT_PHASE_GATE",
    "PhaseGateCompliance",
    "PhaseGateOperatorPrescription",
    "PhaseGateReport",
    "PhaseGateViolation",
    "PhaseStressHotspot",
    "analyze_phase_gate",
    "compare_against_global_baselines",
    "compute_edge_gate_compliance",
    "detect_phase_gate_violations",
    "export_phase_gate_report",
    "prescribe_phase_gate_operators",
    "rank_phase_stress_hotspots",
    "StructuralInterfaceProblem",
    "StructuralInterfaceScore",
    "build_knn_graph",
    "encode_phase_from_binary_state",
    "local_state_disagreement",
    "score_structural_interfaces",
    "interface_score_maps",
    "baseline_score_maps",
    "full_baseline_score_maps",
    "evaluate_interface_scores",
    "render_structural_interface_markdown",
    "render_structural_interface_html",
    "export_structural_interface_report",
    "BASELINE_FORMULAS",
    "compute_all_baselines",
    "constant_baseline",
    "degree_score",
    "feature_deviation",
    "graph_cut_contribution",
    "graph_total_variation",
    "label_propagation_residual",
    "local_class_entropy",
    "local_disagreement",
    "mean_neighbour_distance",
    "random_baseline",
    "TemporalInterfaceConfig",
    "WindowTetradSeries",
    "EarlyWarningComparison",
    "ProspectiveWarningComparison",
    "TemporalWarningCalibration",
    "calibrate_temporal_warning",
    "evaluate_prospective_warning",
    "hilbert_instantaneous_phase",
    "delay_embedding",
    "local_structural_pressure",
    "build_temporal_proximity_graph",
    "window_tetrad_series",
    "rolling_variance",
    "rolling_lag1_autocorrelation",
    "kendall_tau",
    "evaluate_early_warning",
    "MultichannelConfig",
    "MultichannelWindowSeries",
    "SynchronyDiscrimination",
    "fft_bandpass",
    "analytic_phase_amplitude",
    "phase_amplitude_matrices",
    "mean_field_order",
    "kuramoto_order_parameter",
    "phase_locking_matrix",
    "phase_offsets",
    "amplitude_pressure",
    "build_coupling_graph",
    "multichannel_window_series",
    "evaluate_synchrony_discrimination",
    "SignalConfrontation",
    "ModalRootDiagnostic",
    "diagnose_modal_roots",
    "confront_signal",
    "estimate_quality_factor",
    "emergent_wave_fraction",
    "NodalPredictionSkill",
    "nodal_prediction_skill",
    "NodalMeasurementRun",
    "FrozenNodalCalibration",
    "NodalCalibrationError",
    "NodalForecast",
    "NodalForecastScore",
    "calibrate_nodal_prediction",
    "forecast_nodal_response",
    "score_nodal_forecast",
    "write_nodal_forecast",
    "P2MeasurementBounds",
    "P2IntervalCalibration",
    "P2IntervalForecast",
    "P2IntervalComparison",
    "calibrate_p2_transport",
    "forecast_p2_transport",
    "score_p2_transport",
    "write_p2_transport_forecast",
)

__all__ = _GRAMMAR_EXPORTS + _RUNTIME_EXPORTS

_ENFORCE_CANONICAL_GRAMMAR = _grammar.enforce_canonical_grammar


def enforce_canonical_grammar(
    G: Any,
    n: Any,
    cand: Any,
    ctx: Any | None = None,
) -> Any:
    """Proxy to the canonical grammar enforcement helper preserving Glyph outputs."""

    result = _ENFORCE_CANONICAL_GRAMMAR(G, n, cand, ctx)
    if isinstance(cand, Glyph) and not isinstance(result, Glyph):
        translated = _grammar.function_name_to_glyph(result)
        if translated is None and isinstance(result, str):
            try:
                translated = Glyph(result)
            except (TypeError, ValueError):
                translated = None
        if translated is not None:
            return translated
    return result


def __getattr__(name: str) -> Any:
    if name == "NFRValidator":
        from .spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)
