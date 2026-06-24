"""Unified validation interface consolidating grammar, graph and spectral checks.

RECOMMENDED: Use TNFRValidator for unified validation pipeline
==============================================================

The TNFRValidator class provides a single entry point for all TNFR validation
operations, consolidating input validation, graph validation, invariant checking,
operator preconditions, and runtime validation into one coherent API.

Example Usage::

    from tnfr.validation import TNFRValidator

    validator = TNFRValidator()

    # Comprehensive validation in one call
    result = validator.validate(
        graph=G,
        epi=0.5,
        vf=1.0,
        include_invariants=True,
    )

    if not result['passed']:
        print(f"Validation failed: {result['errors']}")

For detailed migration guide, see UNIFIED_VALIDATION_PIPELINE.md

Legacy API
==========

This package also re-exports individual validation functions for backward
compatibility, but these may be deprecated in future versions. New code should
use TNFRValidator instead.
"""

from __future__ import annotations

from typing import Any

from ..operators import grammar as _grammar
from ..types import Glyph
from .base import SubjectT, ValidationOutcome, Validator  # noqa: F401
from .config import (  # noqa: F401
    ValidationConfig,
    configure_validation,
    validation_config,
)
from .graph import GRAPH_VALIDATORS, run_validators  # noqa: F401
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
    TemporalInterfaceConfig,
    WindowTetradSeries,
    build_temporal_proximity_graph,
    delay_embedding,
    evaluate_early_warning,
    hilbert_instantaneous_phase,
    kendall_tau,
    local_structural_pressure,
    rolling_lag1_autocorrelation,
    rolling_variance,
    window_tetrad_series,
)

# Unified validation system exports
from .unified_validation_system import (  # noqa: F401,F811
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

# Legacy exports mapped to unified system where possible
# validate_dnfr_value, validate_epi_value etc are deprecated


# NOTE: Compatibility module deprecated - grammar emerges from TNFR structural dynamics
# Legacy exports kept for backward compatibility but will be removed in future versions
try:
    from .compatibility import (
        CANON_COMPAT,
        CANON_FALLBACK,
        GRADUATED_COMPATIBILITY,
        CompatibilityLevel,
        get_compatibility_level,
    )

    _COMPAT_AVAILABLE = True
except ImportError:
    # Compatibility module removed - provide stubs for backward compatibility
    _COMPAT_AVAILABLE = False
    CANON_COMPAT = {}
    CANON_FALLBACK = {}

    class CompatibilityLevel:
        EXCELLENT = "excellent"
        GOOD = "good"
        CAUTION = "caution"
        AVOID = "avoid"

    GRADUATED_COMPATIBILITY = {}

    def get_compatibility_level(prev: str, next_op: str) -> str:
        """Deprecated: Use frequency transition validation instead."""
        import warnings

        warnings.warn(
            "get_compatibility_level is deprecated. "
            "Grammar rules now emerge naturally from TNFR structural dynamics. "
            "Use validate_frequency_transition from tnfr.operators.grammar instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "good"


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
