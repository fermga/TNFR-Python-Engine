from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

from ..operators.grammar import (
    GrammarContext,
    MutationPreconditionError,
    RepeatWindowError,
    SequenceSyntaxError,
    SequenceValidationResult,
    StructuralGrammarError,
    TholClosureError,
    TransitionCompatibilityError,
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
    record_grammar_violation,
    validate_sequence,
)
from .compatibility import CANON_COMPAT as CANON_COMPAT
from .compatibility import CANON_FALLBACK as CANON_FALLBACK
from .config import ValidationConfig as StructuralValidationConfig
from .config import configure_validation as configure_validation
from .config import validation_config as validation_config
from .graph import GRAPH_VALIDATORS, run_validators
from .input_validation import validate_dnfr_value as validate_dnfr_value
from .input_validation import validate_epi_value as validate_epi_value
from .input_validation import validate_glyph as validate_glyph
from .input_validation import validate_glyph_factors as validate_glyph_factors
from .input_validation import validate_node_id as validate_node_id
from .input_validation import (
    validate_operator_parameters as validate_operator_parameters,
)
from .input_validation import validate_theta_value as validate_theta_value
from .input_validation import validate_tnfr_graph as validate_tnfr_graph
from .input_validation import validate_vf_value as validate_vf_value
from .nodal_prediction import (
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
from .p2_transport import (
    P2IntervalCalibration,
    P2IntervalComparison,
    P2IntervalForecast,
    P2MeasurementBounds,
    calibrate_p2_transport,
    forecast_p2_transport,
    score_p2_transport,
    write_p2_transport_forecast,
)
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon
from .signal_confrontation import (
    ModalRootDiagnostic,
    NodalPredictionSkill,
    SignalConfrontation,
    confront_signal,
    diagnose_modal_roots,
    emergent_wave_fraction,
    estimate_quality_factor,
    nodal_prediction_skill,
)
from .soft_filters import (
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
from .spectral import NFRValidator
from .temporal_interface import (
    ProspectiveWarningComparison,
    TemporalWarningCalibration,
    calibrate_temporal_warning,
    evaluate_prospective_warning,
)
from .unified_validation_system import ValidationConfig as ValidationConfig
from .window import validate_window

SubjectT = TypeVar("SubjectT")

class ValidationOutcome(Generic[SubjectT]):
    subject: SubjectT
    passed: bool
    summary: Mapping[str, Any]
    artifacts: Mapping[str, Any] | None

class Validator(Protocol[SubjectT]):
    def validate(
        self, subject: SubjectT, /, **kwargs: Any
    ) -> ValidationOutcome[SubjectT]: ...
    def report(self, outcome: ValidationOutcome[SubjectT]) -> str: ...

__all__ = (
    "validate_sequence",
    "GrammarContext",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "record_grammar_violation",
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
    "ValidationConfig",
    "StructuralValidationConfig",
    "configure_validation",
    "validation_config",
    "validate_dnfr_value",
    "validate_epi_value",
    "validate_glyph",
    "validate_glyph_factors",
    "validate_node_id",
    "validate_operator_parameters",
    "validate_theta_value",
    "validate_tnfr_graph",
    "validate_vf_value",
    "P2MeasurementBounds",
    "P2IntervalCalibration",
    "P2IntervalForecast",
    "P2IntervalComparison",
    "calibrate_p2_transport",
    "forecast_p2_transport",
    "score_p2_transport",
    "write_p2_transport_forecast",
    "FrozenNodalCalibration",
    "NodalCalibrationError",
    "NodalForecast",
    "NodalForecastScore",
    "NodalMeasurementRun",
    "calibrate_nodal_prediction",
    "forecast_nodal_response",
    "score_nodal_forecast",
    "write_nodal_forecast",
    "ModalRootDiagnostic",
    "NodalPredictionSkill",
    "SignalConfrontation",
    "confront_signal",
    "diagnose_modal_roots",
    "emergent_wave_fraction",
    "estimate_quality_factor",
    "nodal_prediction_skill",
    "ProspectiveWarningComparison",
    "TemporalWarningCalibration",
    "calibrate_temporal_warning",
    "evaluate_prospective_warning",
)
